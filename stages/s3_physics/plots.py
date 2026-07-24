# S3 plots: the multimodal surface the hypothesis agent reads; the figures ARE its evidence. each
# per-trial figure stacks three views on one time axis (leg angles, interleg swap signal, anchor
# timeline). labels are drawn as the BACKGROUND, so physics-vs-label disagreement is visible (Section 10).

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg") # headless: write PNGs, never open a window
import matplotlib.pyplot as plt # noqa: E402
import numpy as np # noqa: E402

from stages.s2_ml.dataset import FEATURES, LABEL_COL, STAND, WALK, HUMAN_UNKNOWN, TIME_COL, Trial
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import (
    ANCHOR_NAMES, SWAP_DELTA_DEG, STANDING, AMBIGUOUS, WALKING, trial_anchors,
)

# human label -> (facecolor, name) for the background band
LABEL_STYLE = {
    STAND: ("#c7c7c7", "STAND"),
    WALK: ("#b6e2b6", "WALK"),
    HUMAN_UNKNOWN: ("#f2c9c9", "UNKNOWN"),
}
# swap-rule verdict -> marker colour (physics track over the label band)
VERDICT_COLOR = {STANDING: "#7a7a7a", AMBIGUOUS: "#e08a1e", WALKING: "#1a8a1a"}


# contiguous (t0, t1, label) runs in seconds, so a background band is one span per run
def _label_spans(t_s: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, int]]:
    if t_s.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(labels) != 0) + 1
    bounds = np.concatenate(([0], cuts, [t_s.size]))
    spans = []
    for i in range(bounds.size - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        spans.append((float(t_s[a]), float(t_s[min(b, t_s.size - 1)]), int(labels[a])))
    return spans


# shade the human-label bands behind an axis
def _shade_labels(ax, t_s: np.ndarray, labels: np.ndarray) -> None:
    for t0, t1, lab in _label_spans(t_s, labels):
        face, _ = LABEL_STYLE.get(lab, ("#ffffff", "?"))
        ax.axvspan(t0, t1, facecolor=face, alpha=0.5, linewidth=0)


# one 3-panel figure for a trial; returns the PNG path. segments drawn separately so no line
# bridges a gap (Section 3.1).
def plot_trial(trial: Trial, out_dir: Path, spec: WindowSpec | None = None) -> Path:
    spec = spec or WindowSpec()
    frame = trial.frame
    t0_ms = float(frame[TIME_COL].min())
    anchors = trial_anchors(trial, spec)
    wt = (anchors["t_start_ms"] - t0_ms) / 1000.0 if not anchors.empty else np.array([])

    # flag an untrusted rest zero (file does not begin at rest, Section 10.5) so the agent discounts it
    rest_note = ("  [!] rest zero untrusted (file does not begin at rest)"
                 if not anchors.empty and not bool(anchors["rest_offset_trusted"].iloc[0])
                 else "")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    fig.suptitle(f"{trial.rev} trial {trial.trial} ({trial.split}) - physics anchors vs "
                 f"human label{rest_note}", fontsize=12)

    for _seg, seg in frame.groupby("segment", sort=True):
        t_s = (seg[TIME_COL].to_numpy(float) - t0_ms) / 1000.0
        labels = seg[LABEL_COL].to_numpy()
        for ax in (ax1, ax2):
            _shade_labels(ax, t_s, labels)
        ax1.plot(t_s, seg[FEATURES[0]], color="#1f77b4", lw=0.8, label="L_ang")
        ax1.plot(t_s, seg[FEATURES[1]], color="#d62728", lw=0.8, label="R_ang")
        ax2.plot(t_s, seg[FEATURES[0]].to_numpy(float) - seg[FEATURES[1]].to_numpy(float),
                 color="#333333", lw=0.8)

    # panel 1: leg angles, label band behind
    ax1.set_ylabel("leg angle (deg)")
    handles = [plt.Line2D([], [], color="#1f77b4", label="L_ang"),
               plt.Line2D([], [], color="#d62728", label="R_ang")]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.5, label=n)
                for c, n in LABEL_STYLE.values()]
    ax1.legend(handles=handles, loc="upper right", ncol=5, fontsize=8, framealpha=0.9)

    # panel 2: swap signal L_ang - R_ang with +/-1 deg commit bands and the stride-adaptive verdict
    # as dots at the top (Section 10.6). this is the verdict the disagreement ranking is built on.
    ax2.axhline(SWAP_DELTA_DEG, color="#888", ls="--", lw=0.7)
    ax2.axhline(-SWAP_DELTA_DEG, color="#888", ls="--", lw=0.7)
    ax2.axhline(0, color="#bbb", lw=0.5)
    if not anchors.empty:
        ytop = ax2.get_ylim()[1]
        for verdict, color in VERDICT_COLOR.items():
            m = anchors["swap_verdict_adaptive"] == verdict
            ax2.scatter(wt[m], np.full(int(m.sum()), ytop), s=14, c=color,
                        marker="s", label=verdict, clip_on=False)
        ax2.legend(loc="lower right", ncol=3, fontsize=8, framealpha=0.9,
                   title="swap verdict (stride-adaptive)")
    ax2.set_ylabel("L_ang - R_ang (deg)")

    # panel 3: bounded anchor timeline
    if not anchors.empty:
        ax3.plot(wt, anchors["antiphase"], color="#9467bd", lw=1.0, label="antiphase")
        ax3.plot(wt, anchors["periodicity"], color="#2ca02c", lw=1.0, label="periodicity")
        ax3.plot(wt, anchors["grav_stab"], color="#8c564b", lw=1.0, label="grav_stab")
        ax3.axhline(0, color="#ddd", lw=0.5)
        ax3.set_ylim(-1.05, 1.05)
        ax3.set_ylabel("bounded anchor")
        ax3.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    ax3.set_xlabel("time (s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trial_{trial.rev}_t{trial.trial}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# corpus bar chart of the rate audit: per-anchor change vs tolerance, coloured by verdict
def plot_rate_audit(report: dict[str, dict], out_dir: Path) -> Path:
    anchors = list(ANCHOR_NAMES)
    deltas = [report[a]["median_delta"] for a in anchors]
    tol = report[anchors[0]]["tol"]
    colors = ["#1a8a1a" if report[a]["verdict"] == "invariant" else "#c0392b" for a in anchors]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(anchors, deltas, color=colors)
    ax.axhline(tol, color="#333", ls="--", lw=1.0, label=f"tol {tol:g}")
    for a, bar in zip(anchors, bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f" {report[a]['metric']}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("median change under 100->50 Hz decimation")
    ax.set_title("Rate-invariance audit (walking windows)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rate_audit.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# every per-trial figure + the audit bar; returns all paths written
def plot_all(trials: list[Trial], report: dict[str, dict], out_dir: Path,
             spec: WindowSpec | None = None) -> list[Path]:
    paths = [plot_trial(t, out_dir, spec) for t in trials]
    paths.append(plot_rate_audit(report, out_dir))
    return paths
