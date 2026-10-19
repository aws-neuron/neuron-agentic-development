"""§4-output: roofline + Pareto + prefill figures from a plan() result.

Consumes the plan() dict (and the model bundle for HW constants); runs no cost math and never
re-invokes cell(). Every value plotted is already on the decode_grid rows (otps, step_ms,
cores, tput_per_node, ai, the `pareto`/`pareto_pn` flags) or the selected colocated target.
matplotlib-only -- kept OUT of the core so `python -m planner` stays
stdlib-only; imported lazily by __main__ under --visualize.

Four PNGs, written to the project's ``agent_artifacts/tmp`` (see ``emit.artifacts_dir``):
  - roofline_<model>.png        : decode cells on the TE/HBM roofline (AI x ceiling).
  - pareto_<model>.png          : per-bucket 3-axis Pareto (otps x ITL, area=cores).
  - pareto_per_node_<model>.png : per-bucket DP-normalized Pareto (per-node otps x ITL).
  - prefill_footprint_<model>.png : per-bucket prefill TTFT vs footprint (bs=1) + SLA ceiling.
"""

from __future__ import annotations

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from entitlement_formulas import roofline_envelope  # noqa: E402
from planner.emit import artifacts_dir  # noqa: E402

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
AREA_PER_CORE = 1.6


def _hw_name(b):
    return b.raw.get("hardware_name", "hw")


def _model_slug(b):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", b.raw.get("model", "model").lower())).strip("_")


def _outpath(b, prefix, outdir):
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, "%s_%s.png" % (prefix, _model_slug(b)))


def _label(r):
    return ("TP%d" % r["TP"]) + (" CP%d" % r["CP"] if r.get("CP", 1) > 1 else "") \
        + (" DP%d" % r["DP"] if r.get("DP", 1) > 1 else "")


def _facets(n):
    cols = 3 if n > 4 else 2
    rows = (n + cols - 1) // cols
    return rows, cols


# --- roofline ------------------------------------------------------------------

def roofline(b, p, outdir):
    hw = b.hw
    peak_tflops = hw.te_peak_flops / 1e12
    ridge = hw.te_peak_flops / hw.hbm_bw_bytes_per_s
    decode = [r for S in p["buckets"] for r in p["decode_grid"][S]]

    merged = {}
    for c in decode:
        ai = c.get("ai", 0.0)
        ceil_tflops = roofline_envelope(hw, ai) / 1e12
        key = (round(ai, 3), round(ceil_tflops, 3))
        m = merged.setdefault(key, {"ai": ai, "ceil": ceil_tflops, "tags": set()})
        m["tags"].add("%dk/bs%d" % (c["bucket"] // 1024, c["max_bs"]))
    pts = [q for q in merged.values() if q["ai"] > 0]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ais = [q["ai"] for q in pts] or [1.0]
    ai_min, ai_max = max(2.0, min(ais) / 3), max(ais) * 3
    xs = np.logspace(np.log10(ai_min), np.log10(ai_max), 400)
    roof = np.minimum(hw.hbm_bw_bytes_per_s / 1e12 * xs, peak_tflops)
    ax.plot(xs, roof, color="black", lw=2.5, zorder=5)
    ax.axvline(ridge, color="gray", ls="--", lw=1.5, zorder=2)
    ax.text(ridge * 0.9, peak_tflops * 1.18, "memory\nbound", ha="right", va="top",
            fontsize=13, color="gray", style="italic")
    ax.text(ridge * 1.12, peak_tflops * 1.18, "compute\nbound", ha="left", va="top",
            fontsize=13, color="gray", style="italic")
    ax.text(ai_max * 0.95, peak_tflops * 1.03, "TE peak %.0f TFLOP/s" % peak_tflops,
            ha="right", va="bottom", fontsize=12)
    for i, q in enumerate(sorted(pts, key=lambda z: z["ai"])):
        ax.scatter([q["ai"]], [q["ceil"]], s=200, zorder=10,
                   color=_PALETTE[i % len(_PALETTE)], edgecolor="black", linewidth=0.6)
        ax.annotate(" ".join(sorted(q["tags"])[:2]) + ("…" if len(q["tags"]) > 2 else ""),
                    (q["ai"], q["ceil"]), textcoords="offset points",
                    xytext=(7, -14), fontsize=7.5, color="#333")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte from HBM)")
    ax.set_ylabel("Throughput ceiling (TFLOP/s)")
    ax.set_title("Roofline (decode @ max-feasible bs) -- %s on %s" % (b.raw.get("model"), _hw_name(b)),
                 fontsize=15, pad=14)
    ax.set_ylim(peak_tflops / 12, peak_tflops * 1.45)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = _outpath(b, "roofline", outdir)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- per-bucket Pareto helpers -------------------------------------------------

def _pareto_facets(b, p, outdir, xkey, flag, prefix, title, xlabel):
    buckets = p["buckets"]
    rows_, cols = _facets(len(buckets))
    fig, axes = plt.subplots(rows_, cols, figsize=(6.2 * cols, 5.0 * rows_))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, S in zip(axes_flat, buckets):
        fit = p["decode_grid"][S]
        par = sorted([c for c in fit if c.get(flag)], key=lambda c: c[xkey])
        dom = [c for c in fit if not c.get(flag)]
        if par:
            ax.plot([c[xkey] for c in par], [c["step_ms"] for c in par],
                    color="#1f77b4", lw=1.5, ls="--", alpha=0.6, zorder=2)
        if dom:
            ax.scatter([c[xkey] for c in dom], [c["step_ms"] for c in dom],
                       s=[c["cores"] * AREA_PER_CORE for c in dom],
                       c="lightgray", edgecolor="gray", alpha=0.6, zorder=3)
        if par:
            ax.scatter([c[xkey] for c in par], [c["step_ms"] for c in par],
                       s=[c["cores"] * AREA_PER_CORE for c in par],
                       c="#1f77b4", edgecolor="black", linewidth=0.7, alpha=0.9, zorder=4)
        par_itls = [c["step_ms"] for c in par] or [c["step_ms"] for c in fit]
        if fit:
            lo = min(c["step_ms"] for c in fit)
            hi = max(par_itls) * 1.5
            pad = (hi - lo) * 0.15 or hi * 0.15
            ax.set_ylim(hi + pad, lo - pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ITL (ms)  ->  better (down)")
        ax.set_title("bucket=%dk  (%d configs)" % (S // 1024, len(fit)), fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
    for ax in list(axes_flat)[len(buckets):]:
        ax.set_visible(False)
    fig.suptitle("%s -- %s on %s" % (title, b.raw.get("model"), _hw_name(b)), fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = _outpath(b, prefix, outdir)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def pareto(b, p, outdir):
    return _pareto_facets(b, p, outdir, "otps", "pareto", "pareto",
                          "Parallelism Pareto (decode @ max bs)",
                          "output throughput (otps)  ->  better")


def pareto_per_node(b, p, outdir):
    """Interactivity Pareto: per-node throughput vs ITL, swept BELOW max batch. Consumes the
    decode_bs_grid (each config's bs ladder), draws each config's sweep as a recessive gray
    line, and the DP-normalized `pareto_pn` frontier bold with points colored per config and
    labeled by batch size. This is the below-max-batch companion to the throughput-capacity
    `pareto` plot -- it traces the ITL/throughput tradeoff the max-batch landscape (ITL ~flat)
    cannot show. Falls back to the max-bs decode_grid if decode_bs_grid is absent."""
    buckets = p["buckets"]
    bs_grid = p.get("decode_bs_grid")
    rows_, cols = _facets(len(buckets))
    fig, axes = plt.subplots(rows_, cols, figsize=(6.2 * cols, 5.0 * rows_))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    # GLOBAL color map: assign a stable hue per config across ALL facets, so the same config
    # is the same color in every bucket. Ordered by first appearance on any bucket's frontier.
    frontier_labels = []
    for S in buckets:
        fit = bs_grid[S] if bs_grid else p["decode_grid"][S]
        for c in sorted([c for c in fit if c.get("pareto_pn")], key=lambda c: c["step_ms"]):
            lbl = _label(c)
            if lbl not in frontier_labels:
                frontier_labels.append(lbl)
    cmap = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(frontier_labels)}

    for ax, S in zip(axes_flat, buckets):
        fit = bs_grid[S] if bs_grid else p["decode_grid"][S]
        # group the sweep by config for the recessive per-config lines
        by_cfg = {}
        for c in fit:
            by_cfg.setdefault(_label(c), []).append(c)
        for i, (lbl, cs) in enumerate(sorted(by_cfg.items())):
            cs = sorted(cs, key=lambda c: c["bs"])
            ax.plot([c["tput_per_node"] for c in cs], [c["step_ms"] for c in cs],
                    color="#cccbc4", lw=1.0, alpha=0.55, marker="o", ms=2.5,
                    markeredgecolor="none", zorder=2)
            ax.annotate(lbl, (cs[-1]["tput_per_node"], cs[-1]["step_ms"]),
                        textcoords="offset points", xytext=(4, 1), fontsize=6,
                        color="#a5a49e", zorder=3)
        par = sorted([c for c in fit if c.get("pareto_pn")], key=lambda c: c["step_ms"])
        if par:
            ax.plot([c["tput_per_node"] for c in par], [c["step_ms"] for c in par],
                    color="#4a4a48", lw=1.8, zorder=5, solid_capstyle="round")
            for c in par:
                ax.scatter([c["tput_per_node"]], [c["step_ms"]], s=70,
                           facecolors=cmap[_label(c)], edgecolor="white", linewidth=1.0,
                           zorder=7)
                ax.annotate("bs%d" % c["bs"], (c["tput_per_node"], c["step_ms"]),
                            textcoords="offset points", xytext=(5, -9), fontsize=6.5,
                            color="#52514e", zorder=8)
        itls = [c["step_ms"] for c in fit]
        if itls:
            lo, hi = min(itls), max(itls)
            pad = (hi - lo) * 0.1 or hi * 0.1
            ax.set_ylim(hi + pad, max(0.0, lo - pad))   # inverted: low ITL at top
        ax.set_xlabel("per-node otps  ->  better")
        ax.set_ylabel("ITL (ms)  ->  better (low)")
        ax.set_title("bucket=%dk  (%d configs)" % (S // 1024, len(by_cfg)), fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
    for ax in list(axes_flat)[len(buckets):]:
        ax.set_visible(False)

    # Figure-level legend (frontier point color -> config), top-left above the plots.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=9, markerfacecolor=cmap[lbl],
                      markeredgecolor="white", label=lbl) for lbl in frontier_labels]
    handles.append(Line2D([0], [0], color="#cccbc4", lw=1.2, marker="o", ms=3,
                          label="other configs (batch sweep)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1,
               fontsize=9, framealpha=0.95, title="frontier point -> config")
    fig.suptitle("Interactivity Pareto (decode, per node, swept below max bs) -- %s on %s"
                 % (b.raw.get("model"), _hw_name(b)), fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.10, 0.03, 1, 0.95])
    out = _outpath(b, "pareto_per_node", outdir)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- prefill TTFT vs footprint -------------------------------------------------

def prefill_footprint(b, p, outdir):
    buckets = p["buckets"]
    rows_, cols = _facets(len(buckets))
    fig, axes = plt.subplots(rows_, cols, figsize=(6.2 * cols, 5.0 * rows_))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, S in zip(axes_flat, buckets):
        target = p["selected"].get(S)
        if target is not None:
            ax.scatter([target["cores"]], [target["ttft_ms"]], s=420, facecolors="none",
                       edgecolors="#b00020", linewidth=2.4, zorder=5)
            pc = target.get("prefix_cache", {})
            seqs = pc.get("cacheable_full_ctx_seqs", 0.0)
            ax.annotate("selected\n%.1f seqs cached" % seqs, (target["cores"], target["ttft_ms"]),
                        textcoords="offset points", xytext=(16, 14), fontsize=8.5,
                        fontweight="bold", color="#b00020")
            ceil = target.get("ttft_ceiling_ms")
            if ceil:
                ax.axhline(ceil, color="#b00020", ls=":", lw=1.6, zorder=3)
                ax.text(target["cores"], ceil, " SLA %.0f ms" % ceil, color="#b00020",
                        fontsize=8, va="bottom", ha="left")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("footprint (cores)  ->  smaller is cheaper")
        ax.set_ylabel("TTFT (ms)  ->  better (down)")
        ax.set_title("bucket=%dk" % (S // 1024), fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    for ax in list(axes_flat)[len(buckets):]:
        ax.set_visible(False)
    note = ("Selected single-node colocated config per bucket (prefill at bs=1). "
            "Serving is colocated on one node; there is no separate prefill pool.")
    fig.suptitle("Prefill TTFT vs footprint (bs=1) -- %s on %s" % (b.raw.get("model"), _hw_name(b)),
                 fontsize=16, y=0.98)
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="gray", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = _outpath(b, "prefill_footprint", outdir)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize(b, p, outdir: str = None) -> list:
    """Write all four figures from the plan dict `p`. Returns the paths written."""
    outdir = artifacts_dir() if outdir is None else outdir
    return [roofline(b, p, outdir), pareto(b, p, outdir),
            pareto_per_node(b, p, outdir), prefill_footprint(b, p, outdir)]
