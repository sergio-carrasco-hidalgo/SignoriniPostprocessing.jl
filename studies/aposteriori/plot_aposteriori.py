#!/usr/bin/env python3
"""
plot_aposteriori.py -- the built-in estimator eta = ||grad eps_h|| as an a posteriori
estimator, Signorini against the same problem without contact condition.

Reads the CSVs of estimator_study.jl and writes, per benchmark,

    <pdfdir>/aposteriori_<bench>.pdf   (default <pdfdir> = results/pdf/03_aposteriori)
        left : effectivity index eta / E,  E^2 = ||grad_h(u-nu_h)||^2 + ||q-q_h||_*^2
        right: saturation constant delta = ||grad_h(u-theta_h)|| / ||grad_h(u-nu_h)||
               (Assumption 3.5 needs delta < 1)

Colour and marker identify the degree k; Signorini is line + filled marker,
the linear problem a hollow ring of the same colour (ring on marker = same value).

usage (from the root of SignoriniPostprocessing.jl):

    python3 studies/aposteriori/plot_aposteriori.py [--csvdir results/csv] [--pdfdir results/pdf/03_aposteriori]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "convergence"))
import plot_convergence as pc          # shared style: rcParams, inks, read_csv

# Categorical slots 1-4 of the reference palette, in order (lines: adjacent pairs).
KSTYLE = {0: ("#2a78d6", "o"), 1: ("#eb6834", "s"), 2: ("#1baf7a", "^"), 3: ("#eda100", "D")}
BENCH_TITLE = {"smooth":   r"smooth benchmark, $u=-r^{11/2}\sin(\frac{11}{2}\theta)+y^2$",
               "singular": r"singular benchmark, $u=-r^{3/2}\sin(\frac{3}{2}\theta)+y^2$"}


def plot_bench(bench, csvdir, outdir, ext="pdf"):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for k, (color, marker) in KSTYLE.items():
        for problem in ("signorini", "linear"):
            path = os.path.join(csvdir, f"aposteriori_{bench}_{problem}_k{k}.csv")
            if not os.path.exists(path):
                continue
            d = pc.read_csv(path)
            for a, col in ((ax, "eff"), (bx, "delta")):
                if problem == "signorini":
                    a.semilogx(d["ndof"], d[col], color=color, marker=marker, lw=1.5, ms=6,
                               mec=pc.SURFACE, mew=0.8, zorder=3)
                else:
                    a.semilogx(d["ndof"], d[col], ls="none", marker="o", ms=11, mfc="none",
                               mec=color, mew=1.1, zorder=4)

    for a, title, ylabel in ((ax, r"effectivity index $\tilde\eta\,/\,E$", r"$\tilde\eta\,/\,E$"),
                             (bx, r"saturation constant $\delta$ (Assumption 3.5)", r"$\delta$")):
        a.set_ylim(0, 1.08)
        a.set_title(title, color=pc.INK)
        a.set_xlabel(r"degrees of freedom of $(\sigma_h, u_h)$")
        a.set_ylabel(ylabel)
        a.grid(True, which="major");  a.grid(False, which="minor")
        a.margins(x=0.06)
    bx.axhline(1.0, color=pc.MUTED, lw=0.8)
    bx.text(bx.get_xlim()[0] * 1.15, 1.0 / 1.02, r"$\delta = 1$", ha="left", va="top",
            fontsize=7.5, color=pc.INK2)

    handles = [Line2D([], [], color=c, marker=m, lw=1.5, ms=6, mec=pc.SURFACE, label=f"$k={k}$")
               for k, (c, m) in KSTYLE.items()]
    handles += [Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=6, mec=pc.SURFACE,
                       label="Signorini"),
                Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=pc.INK2, mew=1.1,
                       label="linear (Dirichlet on all of $\\partial\\Omega$)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.06), columnspacing=1.3)
    fig.suptitle(BENCH_TITLE[bench], color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = os.path.join(outdir, f"aposteriori_{bench}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(csvdir, pdfdir, ext="pdf"):
    os.makedirs(pdfdir, exist_ok=True)
    out = [plot_bench(b, csvdir, pdfdir, ext) for b in ("smooth", "singular")]
    for o in out:
        print("  wrote", os.path.relpath(o, ROOT))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csvdir", default="results/csv")
    ap.add_argument("--pdfdir", default="results/pdf/03_aposteriori")
    a = ap.parse_args()
    absol = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main(absol(a.csvdir), absol(a.pdfdir))
