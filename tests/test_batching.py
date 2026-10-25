import math

import pytest

from indicear.runtime.batching import (
    BucketFrameSampler,
    OomBackoff,
    StepSchedule,
    autotune_frame_budget,
    chunk_by_budget,
    padded_frames,
    scale_lr,
    split_evenly,
    worst_case_batch,
)

LENGTHS = [(i * 37) % 290 + 10 for i in range(500)] + [1200]  # one item over the budget


def test_frame_budget_respected_and_every_item_once_per_epoch():
    sampler = BucketFrameSampler(LENGTHS, frame_budget=1000, num_buckets=8, seed=3)
    for epoch in range(3):
        batches = sampler.batches_for_epoch(epoch)
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(len(LENGTHS)))
        for batch in batches:
            assert padded_frames(batch, LENGTHS) <= 1000 or len(batch) == 1


def test_buckets_cover_all_items_and_epochs_reshuffle_deterministically():
    sampler = BucketFrameSampler(LENGTHS, 1000, num_buckets=8, seed=3)
    assert sorted(i for b in sampler.buckets for i in b) == list(range(len(LENGTHS)))
    assert sampler.batches_for_epoch(0) != sampler.batches_for_epoch(1)
    again = BucketFrameSampler(LENGTHS, 1000, num_buckets=8, seed=3)
    assert again.batches_for_epoch(1) == sampler.batches_for_epoch(1)


def test_step_schedule_resume_matches_tail():
    sampler = BucketFrameSampler(LENGTHS, 1500, num_buckets=5, seed=1)
    full = list(StepSchedule(sampler, epochs=3))
    tail = list(StepSchedule(sampler, epochs=3, start_step=len(full) // 2 + 3))
    assert tail == full[len(full) // 2 + 3 :]
    schedule = StepSchedule(sampler, epochs=3)
    assert schedule.epoch_of(0) == 0 and schedule.epoch_of(len(full) - 1) == 2


def test_chunk_by_budget_and_worst_case():
    assert chunk_by_budget([0, 1, 2], [10, 10, 10], 20) == [[0, 1], [2]]
    worst = worst_case_batch(LENGTHS, 2400)
    assert LENGTHS.index(1200) in worst and padded_frames(worst, LENGTHS) <= 2400


def test_autotune_grows_to_target_and_stops_at_oom():
    class FakeOOM(Exception):
        pass

    FakeOOM.__name__ = "OutOfMemoryError"
    seen = []

    def measure(budget):
        seen.append(budget)
        if budget > 30_000:
            raise FakeOOM()
        return 0.1 + budget / 40_000

    budget = autotune_frame_budget(measure, 2_000, 100_000, target=0.85, log=lambda _m: None)
    assert 0.1 + budget / 40_000 <= 0.85 and budget >= 25_000
    assert seen == sorted(seen[:-1]) + seen[-1:]


def test_autotune_raises_when_start_too_big():
    with pytest.raises(RuntimeError):
        autotune_frame_budget(lambda b: 0.99, 1000, 5000, log=lambda _m: None)


def test_lr_scaling_is_square_root():
    assert scale_lr(1e-3, 80_000, 20_000) == pytest.approx(2e-3)
    assert scale_lr(1e-3, 20_000, 20_000) == pytest.approx(1e-3)


def test_oom_backoff_halves_and_retries():
    class OOM(Exception):
        pass

    OOM.__name__ = "OutOfMemoryError"
    calls = []

    def step(chunk, weight):
        if len(chunk) > 3:
            raise OOM()
        calls.append((tuple(chunk), weight))

    backoff = OomBackoff(log=lambda _m: None)
    backoff.run(list(range(10)), step)
    assert backoff.splits == 4
    assert sorted(i for c, _ in calls for i in c) == list(range(10))
    assert math.isclose(sum(w for _, w in calls), 1.0)
    assert split_evenly(list(range(5)), 2) == [[0, 1, 2], [3, 4]]
