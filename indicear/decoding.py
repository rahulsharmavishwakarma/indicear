"""Decode-time script constraints for the script-factored CTC output (no retraining).

The two-level head scores log p(token) = log p(script | h) + log p(token | script, h) per frame,
and an unconstrained best path can switch scripts anywhere. Error analysis of the pilot model
found two failure modes that these constraints remove at decode time; training losses and the
model's weights are untouched, and with both rules off every decoder gives the old output exactly.

**(a) Word-boundary script rule** (`script_rules`). Every token of the aggregate tokenizer has a
script (its id range) and a kind, from its SentencePiece piece:

- WORD: starts a new word ("▁" + letters); BARE: the piece "▁" alone (scripts without whole-word
  pieces spell a word as "▁" + characters); NEUTRAL: no letters or marks (digits, <unk>);
- MARK: a continuation whose first character is a combining mark (Unicode Mn/Mc/Me: vowel sign,
  virama, anusvara, nukta) or a zero-width joiner; CONT: any other continuation (a full akshara
  or a Latin letter).

The decoder state is the script of the current word (`START` at the start of the utterance and
after a BARE piece; a NEUTRAL piece keeps it). Given the state, a token is allowed when it is
WORD, BARE or NEUTRAL; or it has the state's script (same-script continuation, marks included);
or the state is START or Latin and the token is CONT. So a script may change mid-word only from
Latin into a continuation that starts with a full akshara or letter ("company" + "ಗಳು"), and a
combining mark never follows a piece of another script ("family" + "ಿಗೆ", "royal" + "ो"). Indic
to another Indic script needs a word boundary ("ಪ್ರ" + "ाणियों", "स्थानीय" + "ಯ"). A Latin stem
plus a full Indic akshara ("stande" + "न") is indistinguishable from a legitimate Latin stem plus
an Indic suffix at the token level, so it stays allowed.

Indic to Latin mid-word (`latin_split`, on by default): the model often glues an English word to
the preceding Hindi one without "▁" ("कि" + "etition", "जिससे" + "tization"), a missed word
boundary rather than a broken word. Blocking it drops the English word and cost code-mixed span
accuracy on the pilot eval (IndicContextEval Hindi 80.1 -> 78.8, English spans 71.5 -> 68.3), so
by default a Latin continuation (letters, never a mark) after an Indic piece is allowed and
becomes a new word: the decoder inserts the Latin "▁" piece before it (`insert_breaks`).
`latin_split=False` blocks it, the strict form of the rule.

**(b) Utterance script set** (`lid_script_set`). From the utterance LID posteriors: Latin, the
top-1 language's script, and the scripts of the top-`lid_topk` Indic languages whose probability
exceeds `lid_threshold` (defaults 2 and 0.15). With `lid_frame_share` > 0 the frame head adds the
script of every language that is the frame-wise argmax on at least that share of speech frames.
Every other script is masked. Models without LID heads skip this rule.

**(c) Exclusive scripts** (`exclusive`, default Devanagari/Urdu). Hindi and Urdu speech are one
spoken language, so with both scripts allowed the model can write one utterance partly in each
("आप doctor بن چاہتے ہیں"). Of each exclusive group still allowed after (b), only the script with
the most evidence is kept: the sum over frames of its best token's probability where that beats
blank. Nobody writes one utterance in both scripts, so this never removes a correct mix.

Both rules mask the script and token log-probs (no renormalisation: a hard constraint, like a
lexicon). Greedy CTC becomes a constrained best path: per frame the most likely allowed label
given the last emitted token's word state, tracked across blanks and repeats (a repeat of the
last emitted token is always allowed, so CTC collapsing is unchanged). The search is exact per
frame because the allowed set depends only on the state, the script and the token kind: each
frame is reduced to the best token per (script, kind group) (`TokenTables.reduce`), so long audio
needs only [T, scripts, 3] values. CTC prefix beam search (`indicear.transcribe.ctc_prefix_beam`)
applies the same rule per prefix.

    python -m indicear.decoding eval --model PKG --manifests ROOT/manifests --per-lang 600 \\
        --sets 'kathbath_*' 'vaani_test_hi' --out results/pilot/eval-mac-final-constrained
"""

from __future__ import annotations

import argparse
import math
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from indicear.text import LANG_SCRIPT, char_script

# Token kinds and the three reduction groups. Groups are cumulative per state: a state allows the
# first `level` groups of a script (0: script masked, 1: FREE, 2: + CONT, 3: + MARK).
WORD, BARE, NEUTRAL, CONT, MARK = range(5)
FREE_GROUP, CONT_GROUP, MARK_GROUP = range(3)
_GROUP = {WORD: FREE_GROUP, BARE: FREE_GROUP, NEUTRAL: FREE_GROUP, CONT: CONT_GROUP, MARK: MARK_GROUP}
START = 0  # word-start state; state i >= 1 is the script at index i - 1 of the tokenizer order
LATIN = "latn"
_JOINERS = "‌‍"


def piece_kind(piece: str, special: bool = False) -> int:
    """WORD, BARE, NEUTRAL, CONT or MARK for a SentencePiece piece (see the module docstring)."""
    if special:
        return NEUTRAL
    word = piece.startswith("▁")
    body = piece[1:] if word else piece
    if not body:
        return BARE
    if not any(char_script(c) is not None for c in body):
        return NEUTRAL
    if word:
        return WORD
    if unicodedata.category(body[0]) in ("Mn", "Mc", "Me") or body[0] in _JOINERS:
        return MARK
    return CONT


@dataclass(frozen=True)
class DecodeOptions:
    script_rules: bool = True
    lid_script_set: bool = True
    lid_topk: int = 2
    lid_threshold: float = 0.15
    lid_frame_share: float = 0.0  # > 0: scripts of languages winning this share of speech frames
    latin_split: bool = True  # rule (a): a Latin continuation after an Indic piece starts a new word
    # Rule (c): scripts never mixed within one utterance. Hindi and Urdu speech are one spoken
    # language (Hindustani), so the model can split one utterance across Devanagari and Urdu
    # script; only the script with the most decoded evidence over the utterance is kept.
    exclusive: tuple[tuple[str, ...], ...] = (("deva", "arab"),)

    @property
    def active(self) -> bool:
        return self.script_rules or self.lid_script_set or bool(self.exclusive)


DEFAULT = DecodeOptions()
OFF = DecodeOptions(script_rules=False, lid_script_set=False, exclusive=())


def options_from_flags(script_rules: str = "on", lid_script_set: str = "on", lid_topk: int = 2,
                       lid_threshold: float = 0.15, lid_frame_share: float = 0.0,
                       latin_split: str = "on", exclusive_scripts: str = "on") -> DecodeOptions:
    return DecodeOptions(script_rules == "on", lid_script_set == "on", lid_topk, lid_threshold, lid_frame_share,
                         latin_split == "on", DEFAULT.exclusive if exclusive_scripts == "on" else ())


def add_flags(parser: argparse.ArgumentParser) -> None:
    """The constraint flags shared by the transcribe CLI and the eval command (default: on)."""
    parser.add_argument("--script-rules", choices=("on", "off"), default="on",
                        help="word-boundary script rule: no mid-word script switch except Latin + full akshara")
    parser.add_argument("--lid-script-set", choices=("on", "off"), default="on",
                        help="allow only Latin + the scripts of the likely languages (utterance LID)")
    parser.add_argument("--lid-topk", type=int, default=2, help="Indic languages kept by the LID script set")
    parser.add_argument("--lid-threshold", type=float, default=0.15, help="min. utterance LID probability")
    parser.add_argument("--lid-frame-share", type=float, default=0.0,
                        help="also allow languages that win this share of speech frames in the frame LID (0: off)")
    parser.add_argument("--latin-split", choices=("on", "off"), default="on",
                        help="with --script-rules: a Latin continuation glued to an Indic word becomes a new "
                             "word (a space is inserted) instead of being blocked")
    parser.add_argument("--exclusive-scripts", choices=("on", "off"), default="on",
                        help="one of Devanagari / Urdu script per utterance (Hindi and Urdu speech are one "
                             "spoken language): the script with the most decoded evidence is kept")


def options_from_args(args) -> DecodeOptions:
    return options_from_flags(args.script_rules, args.lid_script_set, args.lid_topk, args.lid_threshold,
                              args.lid_frame_share, args.latin_split, getattr(args, "exclusive_scripts", "on"))


# ---------------------------------------------------------------- token tables


class TokenTables:
    """Per-token script and kind for an aggregate tokenizer, and the per-frame reduction."""

    def __init__(self, order: Sequence[str], token_script: Sequence[int], kinds: Sequence[int]):
        self.order = tuple(order)
        self.vocab_size = len(token_script)
        self.token_script = list(token_script)  # 0 blank, -1 pad/bos/eos, else 1 + script index
        self.kinds = list(kinds)
        self.latin = self.order.index(LATIN) + 1 if LATIN in self.order else -1
        groups: list[list[int]] = [[] for _ in range(3 * len(self.order))]
        for token, (script, kind) in enumerate(zip(self.token_script, self.kinds, strict=True)):
            if script >= 1:
                groups[3 * (script - 1) + _GROUP[kind]].append(token)
        self.bounds, perm = [], []
        for group in groups:
            self.bounds.append((len(perm), len(perm) + len(group)))
            perm.extend(group)
        self.perm = torch.tensor(perm, dtype=torch.long)
        # The group level each state allows per script under rule (a): [states, scripts].
        states = len(self.order) + 1
        levels = torch.ones(states, len(self.order), dtype=torch.long)
        for state in range(states):
            for script in range(1, len(self.order) + 1):
                if state == script:
                    levels[state, script - 1] = 3
                elif state in (START, self.latin):
                    levels[state, script - 1] = 2
        self.rule_levels = levels
        bare = [t for t, (s, k) in enumerate(zip(self.token_script, self.kinds, strict=True))
                if s == self.latin and k == BARE]
        self.latin_bare = bare[0] if bare else None  # inserted as the word break of a Latin split
        self._perm_cache: dict[torch.device, torch.Tensor] = {}

    @classmethod
    def from_tokenizer(cls, tokenizer) -> TokenTables:
        kinds = [NEUTRAL] * 4
        for token in range(4, tokenizer.vocab_size):
            script = tokenizer.script_of(token)
            local = token - tokenizer.offsets[script]
            model = tokenizer.models[script]
            special = model.is_unknown(local) or model.is_control(local) or model.is_unused(local)
            kinds.append(piece_kind(tokenizer.piece(token), special))
        return cls(tokenizer.order, tokenizer.token_scripts(), kinds)

    def check(self, vocab_size: int, script_ranges: Sequence[tuple[int, int]]) -> None:
        ranges = []
        for index in range(1, len(self.order) + 1):
            ids = [t for t, s in enumerate(self.token_script) if s == index]
            ranges.append((ids[0], ids[-1] + 1))
        if vocab_size != self.vocab_size or [tuple(r) for r in script_ranges] != ranges:
            raise ValueError("the tokenizer's script ranges do not match the model's output head")

    def next_state(self, state: int, token: int) -> int:
        kind = self.kinds[token]
        if kind == BARE:
            return START
        if kind == NEUTRAL:
            return state
        return self.token_script[token]

    def allows(self, state: int, token: int, scripts: set[int] | None = None,
               levels: torch.Tensor | None = None) -> bool:
        """Rule (a) for one token in a state (strict, or under `levels` from `levels()`), and
        rule (b) when `scripts`, the allowed script indexes, is given. Blank is always allowed."""
        script = self.token_script[token]
        if script == 0:
            return True
        if script < 0 or (scripts is not None and script not in scripts):
            return False
        levels = self.rule_levels if levels is None else levels
        return _GROUP[self.kinds[token]] < int(levels[state, script - 1])

    def splits(self, options: DecodeOptions) -> bool:
        return options.script_rules and options.latin_split and self.latin_bare is not None and self.latin > 0

    def insert_breaks(self, tokens: list[int], frames: list[int], options: DecodeOptions
                      ) -> tuple[list[int], list[int]]:
        """The Latin split's output: a Latin "▁" before every Latin continuation that follows an
        Indic piece mid-word (the decode allowed it as a new word)."""
        if not self.splits(options):
            return tokens, frames
        out_tokens, out_frames, state = [], [], START
        for token, frame in zip(tokens, frames, strict=True):
            if (state not in (START, self.latin) and self.token_script[token] == self.latin
                    and self.kinds[token] == CONT):
                out_tokens.append(self.latin_bare)
                out_frames.append(frame)
                state = START
            out_tokens.append(token)
            out_frames.append(frame)
            state = self.next_state(state, token)
        return out_tokens, out_frames

    def reduce(self, log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[..., V] -> blank [...], best log-prob [..., scripts, 3] and its token id per
        (script, group); an empty group gets -inf and id 0."""
        device = log_probs.device
        perm = self._perm_cache.get(device)
        if perm is None:
            perm = self._perm_cache[device] = self.perm.to(device)
        lp = log_probs.float().index_select(-1, perm)
        values, ids = [], []
        for start, end in self.bounds:
            if end == start:
                values.append(lp.new_full(lp.shape[:-1], -math.inf))
                ids.append(torch.zeros(lp.shape[:-1], dtype=torch.long, device=device))
                continue
            value, index = lp[..., start:end].max(-1)
            values.append(value)
            ids.append(perm[start + index])
        shape = (*lp.shape[:-1], len(self.order), 3)
        return log_probs[..., 0].float(), torch.stack(values, -1).view(shape), torch.stack(ids, -1).view(shape)

    def levels(self, options: DecodeOptions, scripts: set[int] | None) -> torch.Tensor:
        """[states, scripts] group levels under the active rules."""
        levels = self.rule_levels.clone() if options.script_rules else torch.full_like(self.rule_levels, 3)
        if self.splits(options):  # Indic state -> Latin continuation: allowed, as a new word
            for state in range(1, len(self.order) + 1):
                if state != self.latin:
                    levels[state, self.latin - 1] = max(int(levels[state, self.latin - 1]), 2)
        if scripts is not None:
            for index in range(1, len(self.order) + 1):
                if index not in scripts:
                    levels[:, index - 1] = 0
        return levels


# ---------------------------------------------------------------- rule (b)


def script_set(order: Sequence[str], utt_probs: Mapping[str, float] | None, options: DecodeOptions,
               frame_log_probs: torch.Tensor | None = None) -> set[int] | None:
    """Allowed script indexes (1-based in `order`) from the LID outputs, or None (all scripts)
    when the rule is off or there are no utterance LID probabilities."""
    from indicear.lid import LID_LANGS, NON_SPEECH

    if not options.lid_script_set or not utt_probs:
        return None
    ranked = sorted(utt_probs.items(), key=lambda kv: -kv[1])
    names = {LATIN, LANG_SCRIPT[ranked[0][0]]}
    # Top-k distinct Indic *scripts*: languages sharing a script (hi, mr) take one slot, so they
    # cannot push out a third script that LID also finds likely (hi .40, mr .30, ur .25 keeps arab).
    indic: list[str] = []
    for lang, p in ranked:
        script = LANG_SCRIPT[lang]
        if len(indic) >= max(0, options.lid_topk):
            break
        if script != LATIN and script not in indic and p > options.lid_threshold:
            indic.append(script)
    names |= set(indic)
    if options.lid_frame_share > 0 and frame_log_probs is not None and frame_log_probs.shape[0]:
        path = frame_log_probs.argmax(-1)
        speech = path[path != NON_SPEECH]
        if speech.numel():
            counts = torch.bincount(speech - 1, minlength=len(LID_LANGS)).float() / speech.numel()
            names |= {LANG_SCRIPT[LID_LANGS[i]] for i in torch.nonzero(counts >= options.lid_frame_share).flatten().tolist()}
    return {order.index(name) + 1 for name in names if name in order}


def exclusive_scripts(order: Sequence[str], scripts: set[int] | None, blank: torch.Tensor, cat_lp: torch.Tensor,
                      groups: Sequence[Sequence[str]], utt_probs: Mapping[str, float] | None = None) -> set[int] | None:
    """Rule (c): of each group of mutually exclusive scripts that are all still allowed, keep one:
    the script whose languages the utterance LID finds most likely (summed), else (no LID) the one
    with the most decoded evidence, the sum over frames of its best token's probability where that
    beats blank. `scripts` uses 1-based indexes into `order` (None: every script allowed)."""
    if not groups:
        return scripts
    allowed = set(range(1, len(order) + 1)) if scripts is None else set(scripts)
    best = cat_lp.max(-1).values  # [T, scripts]: each script's best token per frame
    evidence = torch.where(best > blank.unsqueeze(-1), best.exp(), torch.zeros_like(best)).sum(0)
    lid: dict[str, float] = defaultdict(float)
    for lang, p in (utt_probs or {}).items():
        lid[LANG_SCRIPT.get(lang, "")] += p
    for group in groups:
        members = [order.index(name) + 1 for name in group if name in order]
        live = [m for m in members if m in allowed]
        if len(live) > 1:
            keep = max(live, key=lambda m: (lid[order[m - 1]], float(evidence[m - 1])))
            allowed -= set(live) - {keep}
    return allowed if scripts is not None or allowed != set(range(1, len(order) + 1)) else None


def allowed_scripts(order: Sequence[str], utt_probs: Mapping[str, float] | None, options: DecodeOptions,
                    frame_log_probs: torch.Tensor | None, blank: torch.Tensor, cat_lp: torch.Tensor) -> set[int] | None:
    """Rules (b) then (c) for one utterance: the 1-based script indexes the decoder may use (None:
    all). `blank` [T] and `cat_lp` [T, scripts, 3] are `TokenTables.reduce` of its log-probs."""
    scripts = script_set(order, utt_probs, options, frame_log_probs)
    return exclusive_scripts(order, scripts, blank, cat_lp, options.exclusive, utt_probs)


def utt_probs_dict(logits_or_probs: torch.Tensor, logits: bool = True) -> dict[str, float]:
    from indicear.lid import LID_LANGS

    probs = logits_or_probs.float().softmax(-1) if logits else logits_or_probs.float()
    return dict(zip(LID_LANGS, probs.cpu().tolist(), strict=True))


# ---------------------------------------------------------------- decoders


def best_per_state(blank: torch.Tensor, cat_lp: torch.Tensor, cat_id: torch.Tensor,
                   levels: torch.Tensor) -> torch.Tensor:
    """Per frame and state, the most likely allowed label (0: blank wins, ties included):
    blank [T], cat_lp / cat_id [T, scripts, 3], levels [states, scripts] -> labels [T, states]."""
    groups = torch.arange(3)
    bias = torch.where(groups[None, None, :] < levels[:, :, None], 0.0, -math.inf)  # [K, S, 3]
    frames = cat_lp.shape[0]
    scores = (cat_lp[:, None] + bias[None]).reshape(frames, levels.shape[0], -1)  # [T, K, S*3]
    best, index = scores.max(-1)
    labels = cat_id.reshape(frames, -1).gather(1, index)
    return torch.where(blank[:, None] >= best, torch.zeros_like(labels), labels)


def constrained_greedy(tables: TokenTables, blank: torch.Tensor, cat_lp: torch.Tensor, cat_id: torch.Tensor,
                       options: DecodeOptions, scripts: set[int] | None) -> tuple[list[int], list[int]]:
    """Constrained best path over one utterance's reduced frames -> (tokens, emission frames)."""
    labels = best_per_state(blank.cpu(), cat_lp.cpu(), cat_id.cpu(), tables.levels(options, scripts)).tolist()
    tokens, frames = [], []
    state, previous = START, 0
    for frame, row in enumerate(labels):
        label = row[state]
        if label != previous and label != 0:
            tokens.append(label)
            frames.append(frame)
            state = tables.next_state(state, label)
        previous = label
    return tables.insert_breaks(tokens, frames, options)


def beam_candidates(cat_lp: torch.Tensor, cat_id: torch.Tensor, top_lp: torch.Tensor, top_ids: torch.Tensor,
                    tables: TokenTables, scripts: set[int] | None) -> tuple[list[list[float]], list[list[int]]]:
    """Per frame: the frame's top-k tokens in allowed scripts plus the best token of every
    (allowed script, group), so every prefix state keeps its best allowed continuation."""
    out_lp, out_ids = [], []
    for c_lp, c_id, t_lp, t_id in zip(cat_lp.tolist(), cat_id.tolist(), top_lp.tolist(), top_ids.tolist(), strict=True):
        chosen: dict[int, float] = {}
        for lp, token in zip(t_lp, t_id, strict=True):
            script = tables.token_script[token]
            if script >= 1 and (scripts is None or script in scripts):
                chosen[token] = lp
        for index, (lps, ids) in enumerate(zip(c_lp, c_id, strict=True)):
            if scripts is not None and index + 1 not in scripts:
                continue
            for lp, token in zip(lps, ids, strict=True):
                if lp > -math.inf:
                    chosen[token] = lp
        out_lp.append(list(chosen.values()))
        out_ids.append(list(chosen))
    return out_lp, out_ids


def prefix_rule(tables: TokenTables, options: DecodeOptions) -> Callable[[tuple[int, ...], int], bool] | None:
    """`allow(prefix, token)` for `ctc_prefix_beam` under rule (a) (rule (b) is applied to the
    candidates), or None without rule (a)."""
    if not options.script_rules:
        return None
    states: dict[tuple[int, ...], int] = {(): START}

    def state_of(prefix: tuple[int, ...]) -> int:
        state = states.get(prefix)
        if state is None:
            state = states[prefix] = tables.next_state(state_of(prefix[:-1]), prefix[-1])
        return state

    levels = tables.levels(options, None)
    return lambda prefix, token: tables.allows(state_of(prefix), token, levels=levels)


class Constrainer:
    """Rule (a) + rule (b) for one tokenizer: batch greedy decoding for `DesignA.decode`."""

    def __init__(self, tables: TokenTables, options: DecodeOptions = DEFAULT):
        self.tables = tables
        self.options = options

    @classmethod
    def from_tokenizer(cls, tokenizer, options: DecodeOptions = DEFAULT) -> Constrainer:
        return cls(TokenTables.from_tokenizer(tokenizer), options)

    def greedy(self, log_probs: torch.Tensor, lengths: torch.Tensor, utt_logits: torch.Tensor | None = None,
               frame_log_probs: torch.Tensor | None = None) -> tuple[list[tuple[list[int], list[int]]], list]:
        """log_probs [B, T, V] -> ([(tokens, frames)] per row, [allowed script names or None])."""
        blank, cat_lp, cat_id = (x.cpu() for x in self.tables.reduce(log_probs))
        out, allowed = [], []
        for row, length in enumerate(lengths.cpu().tolist()):
            frame_lp = frame_log_probs[row, :length] if frame_log_probs is not None else None
            probs = utt_probs_dict(utt_logits[row]) if utt_logits is not None else None
            scripts = allowed_scripts(self.tables.order, probs, self.options, frame_lp, blank[row, :length],
                                      cat_lp[row, :length])
            out.append(constrained_greedy(self.tables, blank[row, :length], cat_lp[row, :length],
                                          cat_id[row, :length], self.options, scripts))
            allowed.append(None if scripts is None else [self.tables.order[i - 1] for i in sorted(scripts)])
        return out, allowed


# ---------------------------------------------------------------- evaluation


def run_eval(args) -> None:
    """Decode eval subsets with a package once per utterance through `Transcriber`, as both the
    unconstrained (baseline) and the constrained decoder; score with
    `indicear.evaluate.score_transcripts` and write one JSON per set and decoder."""
    from concurrent.futures import ThreadPoolExecutor

    from indicear import evaluate, metrics
    from indicear.common import write_json
    from indicear.encoder import load_audio
    from indicear.transcribe import Transcriber

    options = options_from_args(args)
    configs = {"constrained": options}
    if args.baseline_out:
        configs["unconstrained"] = OFF
    if getattr(args, "hint_out", None):
        configs["hinted"] = options  # decoded with the row's language as the hint (`lang_hint`)
    sets = evaluate.eval_sets(Path(args.manifests).expanduser(), args.sets, max_per_lang=args.per_lang,
                              robust_conditions=())
    summary = []
    with Transcriber(args.model, args.device, args.dtype, verify=not args.no_verify) as transcriber, \
            ThreadPoolExecutor(max_workers=8) as pool:
        transcriber.log(f"{len(sets)} sets on {transcriber.device.type}; constraints {asdict(options)}")
        for eval_set in sets:
            started = time.time()
            items = eval_set.items
            hyps = {name: [] for name in configs}
            detected = {name: [] for name in configs}
            langs, examples = [], []
            for item, audio in zip(items, pool.map(lambda i: load_audio(i["audio"]), items), strict=True):
                outputs = transcriber.frame_outputs(audio, tables=True)
                results = {name: transcriber.transcribe(audio, constraints=opt, outputs=outputs,
                                                        lang_hint=item["lang"] if name == "hinted" else None)
                           for name, opt in configs.items()}
                for name, result in results.items():
                    hyps[name].append(result["text"])
                    detected[name].append(result.get("language", {}).get("top"))
                first = next(iter(results.values()))
                langs.append(first.get("language", {}).get("top"))
                texts = {name: r["text"] for name, r in results.items()}
                if len(set(texts.values())) > 1 and len(examples) < args.examples:
                    examples.append({"key": item["key"], **texts})
            seconds = round(time.time() - started)
            hours = round(sum(float(i.get("seconds") or 0.0) for i in items) / 3600, 2)
            lid = metrics.lid_report([i["lang"] for i in items], langs, [float(i.get("seconds") or 0.0) for i in items])
            for name, config in configs.items():
                scored = evaluate.score_transcripts(items, hyps[name])
                out_dir = Path({"constrained": args.out, "unconstrained": args.baseline_out,
                                "hinted": getattr(args, "hint_out", None)}[name])
                record = {**scored, "lid": lid, "eval_set": eval_set.name, "utterances": len(items), "hours": hours,
                          "seconds": seconds, "model": str(Path(args.model).expanduser()),
                          "decoding": {"decoder": "ctc", "beam": 1, **asdict(config),
                                       "lang_hint": "the row's language" if name == "hinted" else None}}
                if name == "constrained" and "unconstrained" in configs:
                    record["changed_utterances"] = sum(a != b for a, b in zip(hyps["constrained"],
                                                                               hyps["unconstrained"], strict=True))
                    record["changed_examples"] = examples
                write_json(out_dir / f"{eval_set.name}.json", record)
                evaluate.write_rows(out_dir / f"{eval_set.name}.rows.jsonl", items, hyps[name], detected[name])
            summary.append(eval_set.name)
            transcriber.log(f"{eval_set.name}: {len(items)} utterances in {seconds} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="score a package on eval subsets with the constrained decoder")
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--manifests", type=Path, required=True)
    ev.add_argument("--sets", nargs="+", default=["*"], help="glob patterns on eval set names")
    ev.add_argument("--per-lang", type=int, default=600, help="first N utterances per language (0: all)")
    ev.add_argument("--out", type=Path, required=True, help="constrained results directory")
    ev.add_argument("--baseline-out", type=Path, help="also write the unconstrained decode here")
    ev.add_argument("--hint-out", type=Path,
                    help="also write the constrained decode given each row's language (in place of the "
                         "utterance LID, as language-given baselines such as IndicConformer are run) here")
    ev.add_argument("--examples", type=int, default=20, help="changed transcripts kept per set")
    ev.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ev.add_argument("--dtype", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    ev.add_argument("--no-verify", action="store_true")
    add_flags(ev)
    args = parser.parse_args(argv)
    if args.command == "eval":
        run_eval(args)


if __name__ == "__main__":
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
