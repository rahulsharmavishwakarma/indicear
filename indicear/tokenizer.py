"""Aggregate (concatenated) per-script tokenizer for design A's output units.

ID layout: 0 blank, 1 pad, 2 bos, 3 eos, then one SentencePiece unigram model per script in a
disjoint range (order `indicear.text.SCRIPTS`). Every token therefore carries its script, which is
what the two-level output p(script) x p(subword | script) factorizes over. Encoding picks the
sub-tokenizer per word by Unicode script; code-switched text needs no language ID.

Open vocabulary: each script model is trained with its **full character inventory** as
`required_chars` (every assigned letter, mark and digit of the script's Unicode block, plus ASCII
digits), so any word in that script can be spelled even if training text never used a
character. Unigram pieces on Indic text are typically aksharas and akshara sequences; no
constraint forces a piece boundary at akshara boundaries (documented limitation).

    python -m indicear.tokenizer train --manifest train.jsonl --out tok/
"""

from __future__ import annotations

import argparse
import bisect
import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

import sentencepiece as spm

from indicear.text import SCRIPT_RANGES, SCRIPTS, word_script

SPECIALS = ("<blank>", "<pad>", "<bos>", "<eos>")
BLANK, PAD, BOS, EOS = 0, 1, 2, 3
# About 512-1k units per script (a design choice); Devanagari serves Hindi and Marathi.
DEFAULT_VOCAB = {script: 768 for script in SCRIPTS} | {"deva": 1024, "latn": 1024}


def script_inventory(script: str) -> str:
    """Every assigned letter, mark and digit in the script's primary Unicode block + 0-9."""
    if script == "latn":  # a-z plus Latin-1 lowercase letters; normalized text is lowercase
        codes = list(range(0x61, 0x7B)) + list(range(0xDF, 0x100))
    else:
        lo, hi = SCRIPT_RANGES[script][0]
        codes = list(range(lo, hi + 1))
    chars = []
    for code in codes:
        char = chr(code)
        # NFC never produces the non-NFC characters
        if (unicodedata.category(char)[0] in "LMN" and unicodedata.name(char, "")
                and unicodedata.normalize("NFC", char) == char):
            chars.append(char)
    return "".join(dict.fromkeys(chars + list("0123456789")))


def script_runs(text: str, default_script: str = "latn") -> list[tuple[str | None, str]]:
    """Split normalized text into (script, words) runs of consecutive same-script words.

    Words without letters (digits) inherit the previous run's script, else `default_script`.
    Words in a script outside `SCRIPTS` get None: training skips them and encoding falls back
    to the default sub-tokenizer (<unk> pieces, dropped on decode).
    """
    runs: list[tuple[str | None, list[str]]] = []
    for word in text.split():
        script = word_script(word)
        if script is None:
            script = runs[-1][0] if runs else default_script
        elif script not in SCRIPTS:
            script = None
        if runs and runs[-1][0] == script:
            runs[-1][1].append(word)
        else:
            runs.append((script, [word]))
    return [(script, " ".join(words)) for script, words in runs]


def _corpora(texts: Iterable[str], scripts: Iterable[str]) -> dict[str, list[str]]:
    """Training lines per script: the same-script runs of every text."""
    corpora: dict[str, list[str]] = {script: [] for script in scripts}
    for text in texts:
        for script, run in script_runs(text):
            if script in corpora:
                corpora[script].append(run)
    return corpora


def _train_scripts(corpora: dict[str, list[str]], out_dir: Path, vocab_sizes: dict[str, int],
                   threads: int) -> dict[str, dict]:
    """One unigram model per script (`<script>.model`), with the full character inventory."""
    stats = {}
    for script, lines in corpora.items():
        inventory = script_inventory(script)
        # The inventory as one extra line gives every character a nonzero training count.
        lines = lines + [" ".join(inventory)]
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines),
            model_prefix=str(out_dir / script),
            vocab_size=max(vocab_sizes[script], len(inventory) + 8),
            model_type="unigram",
            character_coverage=1.0,
            required_chars=inventory,
            # indicear.text.normalize owns normalization; SentencePiece's default NFKC rule
            # would rewrite compatibility characters and break exact round trips.
            normalization_rule_name="identity",
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            hard_vocab_limit=False,  # small corpora cannot always reach the requested size
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=threads,
            minloglevel=2,
        )
        stats[script] = {"lines": len(lines) - 1, "inventory": len(inventory)}
    return stats


class AggregateTokenizer:
    def __init__(self, models: dict[str, spm.SentencePieceProcessor], order: Sequence[str]):
        missing = [script for script in order if script not in models]
        if missing:
            raise ValueError(f"no sub-tokenizer for {missing}")
        self.models = models
        self.order = tuple(order)
        self.offsets: dict[str, int] = {}
        start = len(SPECIALS)
        for script in self.order:
            self.offsets[script] = start
            start += models[script].get_piece_size()
        self.vocab_size = start
        self._starts = [self.offsets[script] for script in self.order]
        self.directory: Path | None = None  # set by load()

    # ------------------------------------------------------------ construction

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        out_dir: str | Path,
        vocab_sizes: dict[str, int] | None = None,
        threads: int = 8,
    ) -> AggregateTokenizer:
        """Train one unigram model per script on that script's words, then save and load.

        A script with no training text still gets a character-level model built from its
        inventory alone, so the output layer covers all eleven scripts.
        """
        vocab_sizes = dict(vocab_sizes or DEFAULT_VOCAB)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = _train_scripts(_corpora(texts, vocab_sizes), out_dir, vocab_sizes, threads)
        (out_dir / "aggregate.json").write_text(
            json.dumps(
                {"order": list(vocab_sizes), "requested": vocab_sizes, "corpus": stats}, indent=2
            ),
            encoding="utf-8",
        )
        return cls.load(out_dir)

    @classmethod
    def extend(
        cls,
        base_dir: str | Path,
        keep: Sequence[str],
        texts: Iterable[str],
        out_dir: str | Path,
        vocab_sizes: dict[str, int] | None = None,
        threads: int = 8,
    ) -> AggregateTokenizer:
        """A tokenizer for a warm-started run: the scripts in `keep` reuse the base tokenizer's
        SentencePiece models byte for byte (so their token ids map one to one onto the base
        model's output rows, shifted by the new offsets); every other script gets a new model
        trained on `texts`. The script order is the base tokenizer's, so the two-level output's
        script classes keep their meaning (indicear.warmstart maps the head rows).

        aggregate.json records the base directory's aggregate.json sha256 and each kept model's
        sha256 ("extended_from"), which indicear.warmstart checks against the run it starts from."""
        import hashlib
        import shutil

        base_dir, out_dir = Path(base_dir), Path(out_dir)
        base_meta = json.loads((base_dir / "aggregate.json").read_text(encoding="utf-8"))
        order = list(base_meta["order"])
        unknown = sorted(set(keep) - set(order))
        if unknown:
            raise ValueError(f"cannot keep {unknown}: the base tokenizer has {order}")
        requested = {script: int(base_meta.get("requested", {}).get(script, DEFAULT_VOCAB.get(script, 768)))
                     for script in order}
        requested.update({s: v for s, v in (vocab_sizes or DEFAULT_VOCAB).items() if s in order and s not in keep})
        out_dir.mkdir(parents=True, exist_ok=True)
        kept = {}
        for script in keep:
            for suffix in (".model", ".vocab"):
                shutil.copyfile(base_dir / f"{script}{suffix}", out_dir / f"{script}{suffix}")
            kept[script] = hashlib.sha256((out_dir / f"{script}.model").read_bytes()).hexdigest()
        trained = {s: requested[s] for s in order if s not in keep}
        stats = _train_scripts(_corpora(texts, trained), out_dir, trained, threads)
        for script in keep:
            stats[script] = {**base_meta.get("corpus", {}).get(script, {}), "kept": True}
        base_sha = hashlib.sha256((base_dir / "aggregate.json").read_bytes()).hexdigest()
        (out_dir / "aggregate.json").write_text(
            json.dumps(
                {"order": order, "requested": requested, "corpus": {s: stats[s] for s in order},
                 "extended_from": {"aggregate_sha256": base_sha, "kept": kept}},
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls.load(out_dir)

    @classmethod
    def load(cls, directory: str | Path) -> AggregateTokenizer:
        directory = Path(directory)
        meta = json.loads((directory / "aggregate.json").read_text(encoding="utf-8"))
        models = {
            script: spm.SentencePieceProcessor(model_file=str(directory / f"{script}.model"))
            for script in meta["order"]
        }
        tokenizer = cls(models, meta["order"])
        tokenizer.directory = directory
        return tokenizer

    def describe(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "specials": list(SPECIALS),
            "ranges": {script: list(self.script_range(script)) for script in self.order},
        }

    def script_range(self, script: str) -> tuple[int, int]:
        start = self.offsets[script]
        return start, start + self.models[script].get_piece_size()

    def token_scripts(self) -> list[int]:
        """Per token id: 0 for blank, -1 for pad/bos/eos, else 1 + index in `self.order`."""
        out = [0, -1, -1, -1]
        for index, script in enumerate(self.order):
            out.extend([index + 1] * self.models[script].get_piece_size())
        return out

    # ------------------------------------------------------------ encode / decode

    def encode(self, text: str, script_hint: str | None = None) -> list[int]:
        default = script_hint if script_hint in self.models else self.order[-1]
        ids: list[int] = []
        for run_script, run in script_runs(text, default_script=default):
            script = run_script if run_script in self.models else default
            offset = self.offsets[script]
            ids.extend(offset + local for local in self.models[script].encode(run))
        return ids

    def script_of(self, token: int) -> str | None:
        if token < len(SPECIALS) or token >= self.vocab_size:
            return None
        return self.order[bisect.bisect_right(self._starts, token) - 1]

    def piece(self, token: int) -> str:
        script = self.script_of(token)
        if script is None:
            return SPECIALS[token] if 0 <= token < len(SPECIALS) else "<invalid>"
        return self.models[script].id_to_piece(token - self.offsets[script])

    def is_unknown(self, token: int) -> bool:
        script = self.script_of(token)
        return script is not None and self.models[script].is_unknown(token - self.offsets[script])

    def decode(self, ids: Iterable[int]) -> str:
        """IDs -> text. Specials and <unk> pieces are skipped."""
        pieces: list[str] = []
        for token in ids:
            token = int(token)
            if self.script_of(token) is None or self.is_unknown(token):
                continue
            pieces.append(self.piece(token))
        return " ".join("".join(pieces).replace("▁", " ").split())

    def word_starts(self, ids: Sequence[int]) -> list[bool]:
        """Per token: does it start a new word? (used to time words from CTC alignments)."""
        return [self.piece(int(t)).startswith("▁") for t in ids]


def main() -> None:
    from indicear.common import read_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--manifest", nargs="+", required=True, help="training manifests only")
    train.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    texts = [row["text"] for path in args.manifest for row in read_jsonl(path)]
    tokenizer = AggregateTokenizer.train(texts, args.out)
    print(json.dumps(tokenizer.describe(), indent=2))


if __name__ == "__main__":
    main()
