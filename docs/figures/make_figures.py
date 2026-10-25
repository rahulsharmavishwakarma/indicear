"""Figures for the Vesper-IndicEAR model card and technical report (docs/figures/*.png and *.svg), from
the run records and benchmark results in this repository.

    uv run --locked --group figures python docs/figures/make_figures.py [--only NAME ...]

Style: Vesper's Ember design tokens, light theme (apps/desktop/src/tokens.css in the Vesper repository):
ink #12110f, muted #6b6962, line #e8e6e0, accent #e84d17, typeface Bricolage Grotesque (OFL, fonts/).
Categorical slots, validated for colour-vision deficiency (adjacent pairs): Ember, blue #2a78d6,
aqua #1baf7a, violet #6250d6. Vesper-IndicEAR is always Ember; the language-given variant is Ember
hatching. Every figure is 7.2 in wide (one text width) unless it is a square matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "results" / "runs"

INK, MUTED, LINE, ACCENT = "#12110f", "#6b6962", "#e8e6e0", "#e84d17"
ACCENT_LIGHT, TINT, PANEL, EDGE = "#f29d7f", "#fdede8", "#f6f5f2", "#8a8781"
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#6250d6"
CONTEXT = "#b9b5ad"  # neutral context marks (not a categorical slot)
WIDTH = 7.2
SYSTEMS = {  # key -> (label, face, hatch)
    "auto": ("Vesper-IndicEAR, automatic language", ACCENT, None),
    "hinted": ("Vesper-IndicEAR, language given", "white", "/////"),
    "indicconformer": ("IndicConformer 600M, language given", BLUE, None),
    "sravaani": ("SraVaani-1.0, automatic language", AQUA, None),
}
LANG_NAMES = {"bn": "Bengali", "en": "English", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam",
              "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"}
LANGS = ["hi", "bn", "te", "ml", "mr", "gu", "pa", "or", "ur", "ta", "kn"]
FAMILIES = [
    ("kathbath_clean_test_unknown", "Kathbath clean"),
    ("kathbath_noisy_test_unknown", "Kathbath noisy"),
    ("indicvoices_valid", "IndicVoices"),
    ("vaani_test", "Vaani"),
    ("indiccontext", "IndicContext"),
    ("mucs_test", "MUCS"),
]
EVENTS = [(1079, "Kathbath ml/or/ur/gu zeroed"), (3427, "Hindi share raised")]


# ---------------------------------------------------------------- style


def static_fonts() -> str:
    """Regular and SemiBold text-size instances of the variable font (its default instance is
    ExtraBold at display size), written to a temporary directory and registered with matplotlib."""
    import tempfile

    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    out = Path(tempfile.mkdtemp(prefix="vesper-figures-fonts-"))
    name = None
    for weight in (400, 600):
        font = instancer.instantiateVariableFont(TTFont(HERE / "fonts" / "bricolage-grotesque.ttf"),
                                                 {"wght": weight, "opsz": 12, "wdth": 100}, updateFontNames=True)
        font["OS/2"].usWeightClass = weight
        path = out / f"bricolage-{weight}.ttf"
        font.save(path)
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
    return name


def style() -> None:
    name = static_fonts()
    plt.rcParams.update({
        "font.family": name, "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": 600,
        "mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": name, "mathtext.cal": name,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": EDGE, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.5, "svg.fonttype": "path",  # text as outlines: renders without the font
        "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelpad": 3,
    })


def save(fig, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {name}")


def title(ax, letter: str | None, text: str) -> None:
    ax.set_title(f"{letter}   {text}" if letter else text)


def thousands(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if v == 0 else f"{v / 1000:g}k"))


def no_ticks(ax) -> None:
    ax.tick_params(length=0)


# ---------------------------------------------------------------- data


def read_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    out = []
    for line in lines[1:]:
        values = line.split(",")
        if values == head:  # a resumed run appends a new header
            continue
        row = {}
        for key, value in zip(head, values, strict=False):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        out.append(row)
    return out


def dev_metrics(log: Path) -> tuple[list[tuple[int, float]], dict[int, dict[str, float]]]:
    """(step, macro dev CER) per evaluation, and per-language dev CER keyed by step (the two lines of one
    evaluation share a timestamp; the language line is written first)."""
    macro, stamps, by_stamp = [], {}, {}
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(\d\d:\d\d:\d\d)\]? ?step (\d+): dev metric ([\d.]+)", line)
        if m:
            macro.append((int(m.group(2)), float(m.group(3))))
            stamps[m.group(1)] = int(m.group(2))
            continue
        m = re.search(r"(\d\d:\d\d:\d\d)\]? ?dev CER by language: (.*)", line)
        if m:
            by_stamp[m.group(1)] = {k: float(v) for k, v in (p.split() for p in m.group(2).split(", "))}
    langs = {stamps[s]: v for s, v in by_stamp.items() if s in stamps}
    return sorted(set(macro)), langs


def ema(values: list[float], alpha: float = 0.15) -> list[float]:
    out, state = [], None
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(state)
            continue
        state = value if state is None else alpha * value + (1 - alpha) * state
        out.append(state)
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bench_value(directory: Path, set_name: str, lang: str, metric: str = "cer_script_normalised"):
    path = directory / f"{set_name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" in data:  # baselines: metrics[decoder][lang]
        decoder = next(iter(data["metrics"].values()), {})
        return (decoder.get(lang) or {}).get(metric)
    return (data.get(lang) or {}).get(metric)  # decoding eval: keyed by language


# ---------------------------------------------------------------- architecture


def rbox(ax, x, y, w, h, face, edge, lw=0.8, radius=0.06, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face,
                                edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))


def label(ax, x, y, text, size=7.4, weight=400, color=INK, ha="center", va="center", z=5):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.3, zorder=z)


def arrow(ax, a, b, color=INK, dashed=False, rad=0.0, lw=0.8, z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0, linestyle=(0, (3, 2)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture() -> None:
    """Encoder drawn as its 12 conformer layers; heads tap the layers they read."""
    W, H = WIDTH, 3.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    mid = 1.95  # vertical centre of the main path

    # input
    rbox(ax, 0.04, mid - 0.34, 0.66, 0.68, "white", INK)
    label(ax, 0.37, mid, "Audio\n16 kHz", size=7.2)
    arrow(ax, (0.70, mid), (0.84, mid))
    rbox(ax, 0.84, mid - 0.34, 0.86, 0.68, PANEL, EDGE)
    label(ax, 1.27, mid, "128 log-mel\nconv subsample\n25 frames/s", size=6.6)

    # encoder: 12 layer bars inside a frozen region
    x0, bw, gap, bh = 2.02, 0.15, 0.052, 1.36
    xs = [x0 + i * (bw + gap) for i in range(12)]
    enc_right = xs[-1] + bw
    region = (x0 - 0.1, mid - bh / 2 - 0.50, enc_right - x0 + 0.2, bh + 0.64)
    rbox(ax, *region, "white", LINE, lw=0.8, radius=0.08, z=1)
    arrow(ax, (1.70, mid), (region[0], mid))
    for i, x in enumerate(xs):
        rbox(ax, x, mid - bh / 2, bw, bh, PANEL, EDGE, lw=0.6, radius=0.03)
        label(ax, x + bw / 2, mid - bh / 2 + 0.11, str(i + 1), size=6.0, color=MUTED, z=4)
        if i >= 6:  # adapter after the block's second feed-forward module
            rbox(ax, x + 0.022, mid + bh / 2 - 0.31, bw - 0.044, 0.25, ACCENT, ACCENT, lw=0.6, radius=0.02, z=3)
    for i in range(11):
        arrow(ax, (xs[i] + bw, mid), (xs[i + 1], mid), color=EDGE, lw=0.5)
    label(ax, x0 - 0.02, region[1] + 0.23, "EmbeddingGemma 2 audio\ntower · frozen · 305M", size=6.1,
          color=MUTED, ha="left")

    # phonetic CTC above: reads layers 6 and 9, feeds back into the next layer (self-conditioning)
    ptop = region[1] + region[3] + 0.42
    px0, px1 = xs[4] - 0.05, xs[10] + bw + 0.05
    rbox(ax, px0, ptop, px1 - px0, 0.40, TINT, ACCENT)
    label(ax, (px0 + px1) / 2, ptop + 0.20, "Phonetic CTC · 82 shared units", size=6.8)
    for i in (5, 8):
        cx = xs[i] + bw / 2
        arrow(ax, (cx, mid + bh / 2), (cx, ptop))
        nx = xs[i + 1] + bw / 2
        arrow(ax, (nx, ptop), (nx, mid + bh / 2 + 0.005), color=ACCENT, dashed=True)
    label(ax, px1 + 0.08, ptop + 0.20, "dashed: self-conditioning\ninto the next layer", size=6.1, color=ACCENT,
          ha="left")

    # language ID below: reads layers 6 and 12
    ly = 0.12
    lx0, lx1 = xs[5] - 0.12, enc_right + 0.12
    rbox(ax, lx0, ly, lx1 - lx0, 0.40, TINT, ACCENT)
    label(ax, (lx0 + lx1) / 2, ly + 0.20, "Language ID · utterance + frame", size=6.8)
    for i in (5, 11):
        cx = xs[i] + bw / 2
        arrow(ax, (cx, mid - bh / 2), (cx, ly + 0.40))

    # script-factored CTC, then decode-time rules
    hx, hw = region[0] + region[2] + 0.2, 1.2
    arrow(ax, (region[0] + region[2], mid), (hx, mid))
    rbox(ax, hx, mid - 0.56, hw, 1.12, TINT, ACCENT)
    label(ax, hx + hw / 2, mid + 0.30, "Script-factored\nCTC", size=7.0, weight=600)
    label(ax, hx + hw / 2, mid - 0.25, "p(script) ×\np(piece | script)\n11 scripts · 8,964 pieces", size=6.2)
    rx = hx + hw + 0.2
    rw = W - rx - 0.04
    arrow(ax, (hx + hw, mid), (rx, mid))
    rbox(ax, rx, mid - 0.56, rw, 1.12, "white", INK)
    label(ax, rx + rw / 2, mid + 0.27, "Decode-time\nscript rules", size=7.0, weight=600)
    label(ax, rx + rw / 2, mid - 0.27, "text, language,\ncode-switch\nspans", size=6.2)
    arrow(ax, (lx1, ly + 0.20), (rx + rw / 2, mid - 0.56), color=MUTED, dashed=True, rad=0.25)
    label(ax, rx + rw / 2 + 0.05, 0.62, "language ID\nlimits scripts", size=6.1, color=MUTED, ha="center")

    # legend
    items = ((TINT, ACCENT, "trained head"), (ACCENT, ACCENT, "trained adapter (rank 128)"), (PANEL, EDGE, "frozen"))
    x = 0.04
    for face, edge, text in items:
        rbox(ax, x, H - 0.23, 0.2, 0.13, face, edge, lw=0.7, radius=0.02)
        label(ax, x + 0.27, H - 0.165, text, size=6.8, ha="left")
        x += 0.27 + 0.055 * len(text) + 0.25
    label(ax, W - 0.04, H - 0.165, "11.8M trained parameters, 45 MB", size=6.8, color=MUTED, ha="right")
    save(fig, "fig01_architecture")


# ---------------------------------------------------------------- training


def fig_training() -> None:
    pilot, full = read_csv(RUNS / "pilot-v4lid" / "train.csv"), read_csv(RUNS / "full_warm" / "train.csv")
    pilot_dev, _ = dev_metrics(RUNS / "pilot-v4lid" / "train.log")
    full_dev, _ = dev_metrics(RUNS / "full_warm" / "train.log")
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 5.6), sharex="col", sharey="row",
                             gridspec_kw={"hspace": 0.42, "wspace": 0.08, "width_ratios": [7423, 13394]})
    columns = ((pilot, pilot_dev, "Pilot: 4 languages, A100", []),
               (full, full_dev, "Full run: 12 languages, warm start, RTX PRO 6000", EVENTS))
    for col, (rows, dev, heading, events) in enumerate(columns):
        steps = [r["step"] for r in rows]
        ctc = [r["ctc"] for r in rows]
        a, b, c = axes[0, col], axes[1, col], axes[2, col]
        a.plot(steps, ctc, color=ACCENT_LIGHT, linewidth=0.6, alpha=0.7)
        a.plot(steps, ema(ctc), color=ACCENT, linewidth=1.5)
        a.set_ylim(0, 4)
        title(a, "ab"[col], heading)
        b.plot([s for s, _ in dev], [v for _, v in dev], color=ACCENT, marker="o", markersize=3.6,
               markeredgecolor="white", markeredgewidth=0.7)
        b.annotate(f"{dev[-1][1]:.1f}", dev[-1], textcoords="offset points", xytext=(0, -11), ha="center",
                   fontsize=7, color=INK)
        b.annotate(f"{dev[0][1]:.1f}", dev[0], textcoords="offset points", xytext=(0, 6), ha="center",
                   fontsize=7, color=INK)
        b.set_ylim(0, 40)
        title(b, "cd"[col], ("Dev CER, mean of 4 languages", "Dev CER, mean of 12 languages")[col])
        c.plot(steps, [r["lr"] * 1e3 for r in rows], color=INK, linewidth=1.2)
        c.set_ylim(0, 2.0)
        c.set_xlabel("Optimizer step")
        title(c, "ef"[col], "Learning rate")
        for ax in (a, b, c):
            ax.set_xlim(0, max(steps) * 1.02)
            ax.xaxis.set_major_locator(MultipleLocator(2000))
            thousands(ax)
            for step, _ in events:
                ax.axvline(step, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        for n, (step, text) in enumerate(events):
            b.text(step + 180, 9.5 - 5.0 * n, f"{text} (step {step:,})", fontsize=6.3, color=MUTED, va="center",
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    axes[0, 0].set_ylabel("CTC loss (train)")
    axes[1, 0].set_ylabel("CER (%)")
    axes[2, 0].set_ylabel(r"LR ($\times10^{-3}$)")
    axes[0, 0].text(0.98, 0.95, "thin: per 25 steps\nthick: moving average", transform=axes[0, 0].transAxes,
                    ha="right", va="top", fontsize=6.3, color=MUTED)
    save(fig, "fig02_training_curves")


def fig_dev_by_language() -> None:
    _, langs = dev_metrics(RUNS / "full_warm" / "train.log")
    steps = sorted(langs)
    order = ["hi", "ur", "bn", "pa", "gu", "mr", "or", "te", "kn", "ta", "ml", "en"]
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 4.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.5, "wspace": 0.1})
    for ax, lang in zip(axes.flat, order, strict=True):
        for other in order:
            if other != lang:
                ax.plot(steps, [langs[s][other] for s in steps], color=LINE, linewidth=0.7, zorder=1)
        values = [langs[s][lang] for s in steps]
        ax.plot(steps, values, color=ACCENT, linewidth=1.5, zorder=3)
        ax.scatter([steps[-1]], [values[-1]], color=ACCENT, s=12, zorder=4, edgecolor="white", linewidth=0.6)
        ax.set_title(f"{LANG_NAMES[lang]}  ·  {values[-1]:.1f}", fontsize=7.8)
        ax.axvline(3427, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=2)
        ax.set_ylim(0, 55)
        ax.set_xlim(1500, 13500)
        ax.set_xticks([2000, 7000, 12000])
        thousands(ax)
    axes[0, 0].text(3427 + 300, 52, "Hindi share\nraised", fontsize=6, color=MUTED, va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dev CER (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    save(fig, "fig03_dev_cer_by_language")


def fig_throughput() -> None:
    rows = read_csv(RUNS / "full_warm" / "train.csv")
    gpu = read_csv(RUNS / "full_warm" / "gpu.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    steps = [r["step"] for r in rows]
    fps = [r["frames_per_s"] / 1000 for r in rows]
    ax.plot(steps, fps, color=ACCENT_LIGHT, linewidth=0.6, alpha=0.7)
    ax.plot(steps, ema(fps, 0.2), color=ACCENT, linewidth=1.5)
    ax.axvline(1079, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(1079 + 250, 7, "input-pipeline\nfixes complete", fontsize=6.3, color=MUTED)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Thousand frames / s")
    ax.set_ylim(0, 42)
    ax.set_xlim(0, 13700)
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    thousands(ax)
    title(ax, "a", "Training throughput")
    t0 = gpu[0]["time"]
    hours = [(g["time"] - t0) / 3600 for g in gpu]
    util = [g["util_percent"] for g in gpu]
    window = 24  # 2 minutes of 5 s samples
    smooth = [sum(util[max(0, i - window):i + 1]) / len(util[max(0, i - window):i + 1]) for i in range(len(util))]
    ax = axes[1]
    for a, b in zip(hours, hours[1:]):
        if b - a > 0.05:  # a gap of more than 3 min: the run was stopped and resumed
            ax.axvspan(a, b, color=LINE, linewidth=0)
    segments, start = [], 0
    for i in range(1, len(hours)):
        if hours[i] - hours[i - 1] > 0.05:
            segments.append((start, i))
            start = i
    segments.append((start, len(hours)))
    for a, b in segments:  # no line across a stop
        ax.plot(hours[a:b], smooth[a:b], color=ACCENT, linewidth=1.0)
    ax.set_xlabel("Hours since launch")
    ax.set_ylabel("GPU busy (%), 2-min mean")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(hours))
    ax.text(0.98, 0.06, "grey: run stopped and resumed", transform=ax.transAxes, ha="right", fontsize=6.3, color=MUTED)
    title(ax, "b", "GPU utilisation")
    save(fig, "fig04_throughput_gpu")


# ---------------------------------------------------------------- results


def heldout_means(bench: Path) -> dict[str, list]:
    dirs = {"auto": bench / "vesper-indicear-auto", "hinted": bench / "vesper-indicear-hinted",
            "indicconformer": bench / "indicconformer-ctc", "sravaani": bench / "sravaani"}
    means = {k: [] for k in dirs}
    for family, _ in FAMILIES:
        per = {k: [] for k in dirs}
        for lang in LANGS:
            values = {k: bench_value(d, f"{family}_{lang}", lang) for k, d in dirs.items()}
            if None in values.values():
                continue  # only languages every system has (Urdu: SraVaani has none)
            for k, v in values.items():
                per[k].append(v)
        for k in dirs:
            means[k].append(mean(per[k]))
    return means


def fig_heldout(bench: Path) -> None:
    means = heldout_means(bench)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    width = 0.19
    for i, (key, (name, face, hatch)) in enumerate(SYSTEMS.items()):
        xs = [j + (i - 1.5) * (width + 0.015) for j in range(len(FAMILIES))]
        plt.rcParams["hatch.color"] = ACCENT
        plt.rcParams["hatch.linewidth"] = 0.8
        bars = ax.bar(xs, [m or 0 for m in means[key]], width, label=name, color=face, hatch=hatch,
                      edgecolor=ACCENT if key == "hinted" else face, linewidth=0.8)
        for bar, value in zip(bars, means[key], strict=True):
            if value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=5.6,
                        color=INK)
    ax.set_xticks(range(len(FAMILIES)), [t for _, t in FAMILIES])
    no_ticks(ax)
    ax.set_ylabel("Script-normalised CER (%)")
    ax.set_ylim(0, 55)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=1.6, handlelength=1.6)
    save(fig, "fig05_heldout_comparison")


def ember_map():
    return LinearSegmentedColormap.from_list("ember", ["#fff7f3", "#f9c7b2", ACCENT, "#7a2507"])


def fig_heatmap(bench: Path) -> None:
    directory = bench / "vesper-indicear-auto"
    grid = [[bench_value(directory, f"{f}_{lang}", lang) for f, _ in FAMILIES] for lang in LANGS]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    data = [[math.nan if v is None else v for v in row] for row in grid]
    image = ax.imshow(data, cmap=ember_map(), vmin=0, vmax=60, aspect="auto")
    ax.grid(False)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value is None:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color=CONTEXT)
            else:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 36 else INK)
    ax.set_xticks(range(len(FAMILIES)), [t for _, t in FAMILIES], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(LANGS)), [LANG_NAMES[lang] for lang in LANGS])
    no_ticks(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("Script-normalised CER (%)")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    save(fig, "fig06_per_language_heatmap")


def fig_robustness() -> None:
    rows = json.loads((REPO / "results" / "full" / "eval-g4" / "eval_summary.json").read_text())["rows"]
    want = [("fleurs_dev", "Clean"), ("robust_noise_10", "Noise, 10 dB SNR"), ("robust_noise_5", "Noise, 5 dB SNR"),
            ("robust_babble_10", "Babble, 10 dB SNR"), ("robust_reverb_0.6", "Reverb, RT60 0.6 s"),
            ("robust_reverb_0.9", "Reverb, RT60 0.9 s"), ("robust_reverb_0.6+pointnoise_10", "Reverb 0.6 s + noise")]
    values = []
    for name, text in want:
        match = [r for r in rows if r["set"] == name and r["decoder"] == "ctc_constrained" and r["group"] == "all"]
        if match:
            values.append((text, match[0]["cer"]))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    ys = list(range(len(values)))[::-1]
    clean = values[0][1]
    ax.axvline(clean, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    for y, (text, value) in zip(ys, values, strict=True):
        ax.plot([clean, value], [y, y], color=ACCENT_LIGHT, linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([value], [y], color=INK if y == ys[0] else ACCENT, s=26, zorder=3, edgecolor="white", linewidth=0.7)
        ax.text(value + 1.0, y, f"{value:.1f}", va="center", fontsize=7)
    ax.set_yticks(ys, [text for text, _ in values])
    no_ticks(ax)
    ax.set_xlabel("CER (%), FLEURS dev, 12 languages")
    ax.set_xlim(0, 60)
    ax.grid(axis="y", visible=False)
    save(fig, "fig07_robustness")


def fig_lid() -> None:
    data = json.loads((REPO / "results" / "full" / "eval-g4" / "v4+lid" / "fleurs_dev.json").read_text())
    confusion = data["lid"]["utterance"]["confusion"]
    langs = ["en", "hi", "ur", "pa", "gu", "mr", "bn", "or", "te", "kn", "ta", "ml"]
    matrix = []
    for truth in langs:
        row = confusion.get(truth, {})
        total = sum(row.values()) or 1
        matrix.append([100 * row.get(pred, 0) / total for pred in langs])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    image = ax.imshow(matrix, cmap=ember_map(), vmin=0, vmax=100)
    ax.grid(False)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value >= 0.5:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.4,
                        color="white" if value > 55 else INK)
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(langs)), langs)
    no_ticks(ax)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("Utterances (% of row)")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    save(fig, "fig08_lid_confusion")


def fig_mixture() -> None:
    mixture = json.loads((REPO / "indicear" / "mixtures" / "full_warm.json").read_text())
    share = mixture["sampling"]["max_share"]
    hours: dict[str, dict[str, float]] = {}
    for stream in mixture["train"]:
        group = stream["source"]
        group = ("English corpora" if group in ("librispeech", "peoples_speech", "voxpopuli")
                 else group.split("_")[0])
        hours.setdefault(stream["lang"], {}).setdefault(group, 0)
        hours[stream["lang"]][group] += stream.get("hours", 0)
    order = sorted([k for k in share if k != "cs"], key=lambda k: (-share[k], k))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.5]})
    ax = axes[0]
    ys = list(range(len(order) + 1))[::-1]
    labels = [LANG_NAMES[k] for k in order] + ["Code-mixed rows"]
    values = [100 * share[k] for k in order] + [100 * share["cs"]]
    ax.barh(ys, values, color=[ACCENT] * len(order) + [ACCENT_LIGHT], height=0.62)
    for y, v in zip(ys, values, strict=True):
        ax.text(v + 0.3, y, f"{v:g}%", va="center", fontsize=6.8)
    ax.set_yticks(ys, labels)
    no_ticks(ax)
    ax.set_xlim(0, 17.5)
    ax.set_xlabel("Share of drawn rows (%)")
    ax.grid(axis="y", visible=False)
    title(ax, "a", "Sampling shares")
    ax = axes[1]
    groups = [("indicvoices", "IndicVoices", ACCENT), ("kathbath", "Kathbath", BLUE), ("vaani", "Vaani", AQUA),
              ("English corpora", "LibriSpeech, People's Speech, VoxPopuli", VIOLET)]
    ypos = list(range(len(order)))[::-1]
    left = [0.0] * len(order)
    for key, name, colour in groups:
        vals = [hours.get(lang, {}).get(key, 0) for lang in order]
        ax.barh(ypos, vals, left=left, color=colour, height=0.62, label=name, edgecolor="white", linewidth=0.8)
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_yticks(ypos, [LANG_NAMES[k] for k in order])
    no_ticks(ax)
    ax.set_xlabel("Planned hours (before speaker exclusion)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 700)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=6.6, handlelength=1.0,
              handleheight=0.8, columnspacing=1.2)
    title(ax, "b", "Training pools by source")
    save(fig, "fig09_data_mixture")


def fig_vesper(vesper: Path) -> None:
    path = vesper / "scores.json"
    if not path.exists():
        print("skip fig10: no Vesper benchmark scores")
        return
    scores = json.loads(path.read_text())
    sets = [("fleurs_hi", "FLEURS Hindi"), ("mucs_hi_en", "MUCS Hindi-English"), ("svarah", "Svarah, Indian English"),
            ("fleurs_en", "FLEURS US English")]
    sets = [s for s in sets if s[0] in scores]
    fig, mosaic = plt.subplot_mosaic([["fleurs_hi", "fleurs_en"], ["mucs_hi_en", "fleurs_en"], ["svarah", "fleurs_en"]],
                                     figsize=(WIDTH, 5.0), gridspec_kw={"wspace": 1.0, "hspace": 0.7})
    for name, heading in sets:
        ax = mosaic[name]
        rows = [(m, s) for m, s in scores[name].items()
                if not (name.startswith("fleurs_en") or name == "svarah") or "language given" not in m]
        rows.sort(key=lambda kv: kv[1]["cer"])
        ys = list(range(len(rows)))[::-1]
        top = max(s["cer"] for _, s in rows)
        for y, (model, s) in zip(ys, rows, strict=True):
            ours = model.startswith("Vesper-IndicEAR")
            ax.plot([0, s["cer"]], [y, y], color=ACCENT_LIGHT if ours else LINE, linewidth=1.6,
                    solid_capstyle="round", zorder=1)
            ax.scatter([s["cer"]], [y], color=ACCENT if ours else CONTEXT, s=22, zorder=3, edgecolor="white",
                       linewidth=0.7)
            ax.text(s["cer"] + top * 0.03, y, f"{s['cer']:.1f}", va="center", fontsize=6.6)
        ax.set_yticks(ys, [m.replace(" (language given)", ", language given") for m, _ in rows], fontsize=6.9)
        no_ticks(ax)
        for tick in ax.get_yticklabels():
            if tick.get_text().startswith("Vesper-IndicEAR"):
                tick.set_color(ACCENT)
        ax.set_xlabel("CER (%)")
        ax.set_xlim(0, top * 1.22)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="y", visible=False)
        title(ax, None, f"{heading} · {rows[0][1]['clips']} clips")
    save(fig, "fig10_vesper_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=REPO / "results" / "bench")
    parser.add_argument("--vesper", type=Path, default=REPO / "results" / "vesper-benchmark")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    style()
    figures = {"architecture": fig_architecture, "training": fig_training, "language": fig_dev_by_language,
               "throughput": fig_throughput, "heldout": lambda: fig_heldout(args.bench),
               "heatmap": lambda: fig_heatmap(args.bench), "robustness": fig_robustness, "lid": fig_lid,
               "mixture": fig_mixture, "vesper": lambda: fig_vesper(args.vesper)}
    for name, make in figures.items():
        if not args.only or name in args.only:
            make()


if __name__ == "__main__":
    main()
