"""Dynamic batching by a padded-frame budget, VRAM auto-tuning and LR scaling.

A batch's cost is `len(batch) * longest item` (padded frames), not its clip count. Items are
sorted into length buckets; each epoch shuffles inside buckets, chunks each bucket under the
budget, then shuffles the batches across buckets.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence


def padded_frames(batch: Sequence[int], lengths: Sequence[int]) -> int:
    return len(batch) * max(lengths[i] for i in batch) if batch else 0


def chunk_by_budget(
    order: Sequence[int], lengths: Sequence[int], frame_budget: int, max_items: int | None = None
) -> list[list[int]]:
    """Greedy chunks of `order` with len(batch) * max length <= frame_budget.

    An item longer than the budget on its own forms a single-item batch.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in order:
        length = lengths[index]
        new_longest = max(longest, length)
        too_big = current and new_longest * (len(current) + 1) > frame_budget
        if too_big or (max_items is not None and len(current) >= max_items):
            batches.append(current)
            current, new_longest = [], length
        current.append(index)
        longest = new_longest
    if current:
        batches.append(current)
    return batches


class BucketFrameSampler:
    """Length-bucketed batches under a frame budget, reshuffled every epoch.

    Deterministic for a given (seed, epoch), so a resumed run replays the same schedule.
    `weights` (optional, e.g. temperature-sampling weights) is the expected number of draws of each
    item per epoch: item i appears floor(w) times plus once more with probability frac(w), drawn
    afresh every epoch. Without weights every item appears exactly once.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        frame_budget: int,
        num_buckets: int = 30,
        shuffle: bool = True,
        seed: int = 0,
        max_items: int | None = None,
        weights: Sequence[float] | None = None,
    ):
        if frame_budget <= 0:
            raise ValueError("frame_budget must be positive")
        if weights is not None and len(weights) != len(lengths):
            raise ValueError("weights and lengths differ in length")
        self.weights = None if weights is None else [max(0.0, float(w)) for w in weights]
        self.lengths = list(lengths)
        self.frame_budget = frame_budget
        self.shuffle = shuffle
        self.seed = seed
        self.max_items = max_items
        by_length = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        count = max(1, min(num_buckets, len(by_length)))
        size = math.ceil(len(by_length) / count) if by_length else 0
        self.buckets = [by_length[i : i + size] for i in range(0, len(by_length), size or 1)]
        self._cache: dict[int, list[list[int]]] = {}

    def batches_for_epoch(self, epoch: int) -> list[list[int]]:
        if epoch not in self._cache:
            rng = random.Random(self.seed * 1_000_003 + epoch)
            counts = None
            if self.weights is not None:
                counts = [int(w) + (rng.random() < w - int(w)) for w in self.weights]
            batches: list[list[int]] = []
            for bucket in self.buckets:
                order = list(bucket) if counts is None else [i for i in bucket for _ in range(counts[i])]
                if self.shuffle:
                    rng.shuffle(order)
                batches.extend(
                    chunk_by_budget(order, self.lengths, self.frame_budget, self.max_items)
                )
            if self.shuffle:
                rng.shuffle(batches)
            self._cache = {epoch: batches}  # keep only the current epoch in memory
        return self._cache[epoch]

    def __len__(self) -> int:
        return len(self.batches_for_epoch(0))


class StepSchedule:
    """Iterable of batches over `epochs` epochs, starting at global step `start_step`.

    Used as a DataLoader `batch_sampler` so one persistent loader serves the whole run.
    """

    def __init__(self, sampler: BucketFrameSampler, epochs: int, start_step: int = 0):
        self.sampler = sampler
        self.epochs = epochs
        self.start_step = start_step
        self.epoch_sizes = [len(sampler.batches_for_epoch(e)) for e in range(epochs)]
        self.total_steps = sum(self.epoch_sizes)

    def epoch_of(self, step: int) -> int:
        seen = 0
        for epoch, size in enumerate(self.epoch_sizes):
            seen += size
            if step < seen:
                return epoch
        return self.epochs - 1

    def __iter__(self) -> Iterator[list[int]]:
        step = 0
        for epoch in range(self.epochs):
            size = self.epoch_sizes[epoch]
            if step + size <= self.start_step:
                step += size
                continue
            for batch in self.sampler.batches_for_epoch(epoch):
                if step >= self.start_step:
                    yield batch
                step += 1

    def __len__(self) -> int:
        return max(0, self.total_steps - self.start_step)


def worst_case_batch(lengths: Sequence[int], frame_budget: int) -> list[int]:
    """The longest items that fit the budget: the peak-memory batch for RNN-T and attention."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    return chunk_by_budget(order, lengths, frame_budget)[0]


def autotune_frame_budget(
    measure: Callable[[int], float],
    start: int,
    max_budget: int,
    target: float = 0.85,
    growth: float = 1.3,
    log: Callable[[str], None] = print,
) -> int:
    """Grow the frame budget until a real step's peak memory reaches `target`, then fix it.

    `measure(budget)` runs one forward + backward (+ optimizer step) on a worst-case batch and
    returns peak allocated memory as a fraction of device memory; it may raise
    torch.OutOfMemoryError (any exception whose name contains "OutOfMemory" counts as OOM).
    After the first overshoot, one interpolated budget between the last good and the
    overshooting budget is tried, since memory grows roughly linearly with the budget.
    """
    best: tuple[int, float] | None = None
    budget = start
    overshoot: tuple[int, float] | None = None
    while budget <= max_budget:
        try:
            fraction = measure(budget)
        except Exception as error:
            if "OutOfMemory" not in type(error).__name__:
                raise
            log(f"autotune: {budget} frames -> out of memory")
            overshoot = (budget, float("inf"))
            break
        log(f"autotune: {budget} frames -> peak {100 * fraction:.1f}% of device memory")
        if fraction > target:
            overshoot = (budget, fraction)
            break
        best = (budget, fraction)
        if budget == max_budget:
            break
        budget = min(max_budget, max(budget + 1, int(budget * growth)))
    if best is None:
        raise RuntimeError(
            f"the starting budget of {start} frames already exceeds {100 * target:.0f}% of "
            "device memory; lower --tune-start"
        )
    if overshoot is not None and math.isfinite(overshoot[1]) and overshoot[1] > best[1]:
        good, good_fraction = best
        slope = (overshoot[1] - good_fraction) / (overshoot[0] - good)
        guess = int(good + 0.97 * (target - good_fraction) / slope)
        if good < guess < overshoot[0]:
            try:
                fraction = measure(guess)
                log(f"autotune: {guess} frames -> peak {100 * fraction:.1f}% of device memory")
                if fraction <= target:
                    best = (guess, fraction)
            except Exception as error:
                if "OutOfMemory" not in type(error).__name__:
                    raise
                log(f"autotune: {guess} frames -> out of memory")
    log(f"autotune: fixed frame budget {best[0]} ({100 * best[1]:.1f}% peak)")
    return best[0]


def scale_lr(base_lr: float, effective_frames: int, reference_frames: int) -> float:
    """Square-root LR scaling: base_lr * sqrt(effective batch / reference batch), in frames."""
    return base_lr * math.sqrt(effective_frames / reference_frames)


# ---------------------------------------------------------------- VRAM sizing and OOM backoff


def is_oom(error: BaseException) -> bool:
    """torch.OutOfMemoryError (CUDA), or a RuntimeError such as "MPS backend out of memory"
    (Apple MPS raises a plain RuntimeError)."""
    return "OutOfMemory" in type(error).__name__ or "out of memory" in str(error).lower()


def split_evenly(items: Sequence, parts: int) -> list[list]:
    parts = max(1, min(parts, len(items)))
    size = math.ceil(len(items) / parts)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def vram_capacity(share: float = 1.0) -> int:
    """Bytes this process may use: its share of the device, capped by what is free right now
    (other jobs may already be resident on the same GPU)."""
    import torch

    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    used_by_us = torch.cuda.memory_allocated()
    return int(min(total * share, free + used_by_us))


class OomBackoff:
    """Runs `step(sub_batch, weight)` over a batch split into `self.splits` micro-batches; on
    CUDA or MPS OOM it frees memory, doubles the split count (halving the micro-batch budget) and
    retries the whole batch. The split count persists for the rest of the run. `weight` is the
    micro-batch's share of the rows; the trainer instead normalizes every loss term by the whole
    batch's denominators when it splits, so the accumulated gradient equals the one-batch one."""

    def __init__(self, splits: int = 1, max_splits: int = 64, log: Callable[[str], None] = print):
        self.splits, self.max_splits, self.log = splits, max_splits, log

    def run(self, batch: Sequence, step: Callable[[list, float], None], on_retry: Callable[[], None] | None = None) -> None:
        from indicear.runtime.gpu import empty_device_cache

        while True:
            try:
                chunks = split_evenly(batch, self.splits)
                for chunk in chunks:
                    step(chunk, len(chunk) / len(batch))
                return
            except Exception as error:
                if not is_oom(error) or self.splits >= self.max_splits or len(batch) <= self.splits:
                    raise
                self.splits *= 2
                self.log(f"OOM: retrying with {self.splits} micro-batches per batch")
                if on_retry is not None:
                    on_retry()
            empty_device_cache()  # outside the handler, once the error's traceback (and its tensors) is gone
