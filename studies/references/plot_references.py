#!/usr/bin/env python3
"""
plot_references.py -- plots of the reference examples (reference_examples.jl).

Examples with a known solution (fhs42, bgk72):
    <pdfdir>/<ex>_k<k>.pdf          ||sigma - sigma_h||, ||u - u_h||, ||u - nu_h|| and the
                                    estimator eta against ndof: adaptive (line + filled
                                    marker) vs uniform (thin line + hollow marker);
                                    triangles: optimal adaptive orders
    <pdfdir>/<ex>_effectivity.pdf   eta / E and the saturation constant delta, all k
Examples with an unknown solution (fhs43, bgk731):
    <pdfdir>/<ex>_estimator.pdf     eta, adaptive vs uniform, one panel per k
All examples:
    <pdfdir>/<ex>_meshes.pdf        final adaptive mesh per k

usage (from the root of SignoriniPostprocessing.jl):

    python3 studies/references/plot_references.py [--csvdir results/csv/references]
                                                  [--pdfdir results/pdf/05_references]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "convergence"))
sys.path.insert(0, os.path.join(HERE, "..", "adaptivity"))
import plot_convergence as pc          # shared style: rcParams, inks, colours, read_csv
import plot_adaptive as pa             # slope triangles, k colours

EXAMPLES = {
    "fhs42":  dict(title=r"Führer–Heuer–Stephan §4.2: $-\Delta u+u=f$, Signorini on $\partial\Omega$",
                   box=(-1, 1, 0, 1), exact=True,
                   # u is a polynomial of degree <= 5 on each side of x = 0: k = 3
                   # reproduces it to round-off (errors ~1e-11), nothing to measure
                   skip_k=[3]),
    "fhs43":  dict(title=r"Führer–Heuer–Stephan §4.3: L-shape, $-\Delta u+u=f$, Signorini on $\partial\Omega$",
                   box=(-1, 1, -1, 1), exact=False),
    "bgk72":  dict(title=r"Bartels–Gudi–Kaltenbach §7.2: $u=-10\,\psi(r)\,r^{3/2}\sin(\frac{3}{2}\theta)$",
                   box=(0, 1, 0, 1), exact=True),
    "bgk731": dict(title=r"Bartels–Gudi–Kaltenbach §7.3.1: obstacle $\chi$, $\Gamma_D$, $\Gamma_N$, $\Gamma_C$",
                   box=(-1, 1, -1, 1), exact=False),
}
# (style key, column, label, optimal adaptive order r(k): e ~ N^{-r/2}; None = no triangle)
CURVES = [("flux", "err_sigma_L2", r"$\Vert\sigma-\sigma_h\Vert_{L^2}$", lambda k: k + 2),
          ("u",    "err_u_L2",     r"$\Vert u-u_h\Vert_{L^2}$",           None),
          ("nu",   "err_nu_L2",    r"$\Vert u-\nu_h\Vert_{L^2}$",         lambda k: k + 3)]


def csv(csvdir, ex, kind, k):
    return pc.read_csv(os.path.join(csvdir, f"{ex}_{kind}_k{k}.csv"))


def legend_styles():
    return [Line2D([], [], color=pc.INK2, lw=1.0, label=r"$\tilde\eta$"),
            Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                   label="adaptive (Dörfler, NVB)"),
            Line2D([], [], color=pc.INK2, marker="o", lw=0.8, ms=6, mfc="none", label="uniform")]


def plot_errors(ex, k, csvdir, outdir, ext):
    ad, un = csv(csvdir, ex, "adaptive", k), csv(csvdir, ex, "uniform", k)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for key, col, label, order in CURVES:
        color, marker = pc.STYLE[key]
        ax.loglog(ad["ndof"], ad[col], color=color, marker=marker, lw=1.5, ms=4.5,
                  mec=pc.SURFACE, mew=0.6, zorder=3)
        ax.loglog(un["ndof"], un[col], color=color, marker=marker, lw=0.8, ms=7,
                  mfc="none", mec=color, mew=1.0, zorder=2)
        if order is not None:
            pa.triangle(ax, ad["ndof"], ad[col], order(k))
    ax.loglog(ad["ndof"], ad["eta"], color=pc.INK2, lw=1.0, zorder=3)
    ax.loglog(un["ndof"], un["eta"], color=pc.INK2, lw=0.6, zorder=2)
    ax.set_title(EXAMPLES[ex]["title"] + "\n" + rf"$k={k}$", color=pc.INK, fontsize=9)
    ax.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$")
    ax.set_ylabel("error / estimator")
    ax.grid(True, which="major");  ax.grid(False, which="minor")
    ax.margins(x=0.06, y=0.1)
    handles = [Line2D([], [], color=pc.STYLE[key][0], marker=pc.STYLE[key][1], lw=1.5, ms=5,
                      mec=pc.SURFACE, label=label) for key, _, label, _ in CURVES] + legend_styles()
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, labelcolor=pc.INK, columnspacing=1.2)
    out = os.path.join(outdir, f"{ex}_k{k}.{ext}")
    fig.savefig(out, bbox_inches="tight");  plt.close(fig)
    return out


def plot_effectivity(ex, ks, csvdir, outdir, ext):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for k in ks:
        color, marker = pa.KSTYLE[k]
        for kind, lw, filled in (("adaptive", 1.5, True), ("uniform", 0.8, False)):
            d = csv(csvdir, ex, kind, k)
            for a, col in ((ax, "eff"), (bx, "delta")):
                a.semilogx(d["ndof"], d[col], color=color, marker=marker, lw=lw,
                           ms=4 if filled else 6, mfc=color if filled else "none",
                           mec=pc.SURFACE if filled else color, mew=0.5 if filled else 0.9,
                           zorder=3 if filled else 2)
    for a, title, ylab in ((ax, r"effectivity index $\tilde\eta/E$", r"$\tilde\eta\,/\,E$"),
                           (bx, r"saturation constant $\delta$ (Assumption 3.5)", r"$\delta$")):
        a.set_ylim(0, 1.08);  a.set_title(title, color=pc.INK)
        a.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$");  a.set_ylabel(ylab)
        a.grid(True, which="major");  a.grid(False, which="minor")
    bx.axhline(1.0, color=pc.MUTED, lw=0.8)
    handles = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, mec=pc.SURFACE, label=f"$k={k}$")
               for k, (c, m) in pa.KSTYLE.items() if k in ks] + legend_styles()[1:]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(EXAMPLES[ex]["title"], color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(outdir, f"{ex}_effectivity.{ext}")
    fig.savefig(out, bbox_inches="tight");  plt.close(fig)
    return out


def plot_estimator(ex, ks, csvdir, outdir, ext):
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.3), squeeze=False)
    for ax, k in zip(axs[0], ks):
        ad, un = csv(csvdir, ex, "adaptive", k), csv(csvdir, ex, "uniform", k)
        ax.loglog(ad["ndof"], ad["eta"], color=pc.INK2, marker="o", lw=1.5, ms=4,
                  mec=pc.SURFACE, mew=0.5, zorder=3)
        ax.loglog(un["ndof"], un["eta"], color=pc.INK2, marker="o", lw=0.8, ms=6,
                  mfc="none", mec=pc.INK2, mew=0.9, zorder=2)
        pa.triangle(ax, ad["ndof"], ad["eta"], k + 2)
        ax.set_title(rf"$k={k}$", color=pc.INK);  ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel(r"$\tilde\eta$")
    fig.legend(handles=legend_styles()[1:], loc="lower center", ncol=2, frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(EXAMPLES[ex]["title"] + r" (unknown $u$): estimator $\tilde\eta$",
                 color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(outdir, f"{ex}_estimator.{ext}")
    fig.savefig(out, bbox_inches="tight");  plt.close(fig)
    return out


def plot_meshes(ex, ks, csvdir, outdir, ext):
    x0, x1, y0, y1 = EXAMPLES[ex]["box"]
    w = 2.6 * (x1 - x0) / (y1 - y0) if (x1 - x0) > (y1 - y0) else 2.6
    fig, axs = plt.subplots(1, len(ks), figsize=(w * len(ks) * 0.55 + 0.4, 2.9), squeeze=False)
    for ax, k in zip(axs[0], ks):
        tris = pa.read_mesh(os.path.join(csvdir, f"{ex}_adaptive_k{k}_mesh.csv"))
        ax.add_collection(PolyCollection(tris, facecolors="none", edgecolors=pc.INK2, linewidths=0.12))
        ax.set_xlim(x0, x1);  ax.set_ylim(y0, y1);  ax.set_aspect("equal")
        ax.set_xticks([]);  ax.set_yticks([]);  ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"$k={k}$: {len(tris)} triangles", color=pc.INK, fontsize=9)
    fig.suptitle(EXAMPLES[ex]["title"] + r": final adaptive meshes ($\tilde\eta$-driven)",
                 color=pc.INK, fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, f"{ex}_meshes.{ext}")
    fig.savefig(out, bbox_inches="tight");  plt.close(fig)
    return out


def main(csvdir, pdfdir, ext="pdf"):
    os.makedirs(pdfdir, exist_ok=True)
    out = []
    for ex, spec in EXAMPLES.items():
        ks = [k for k in range(4) if os.path.exists(os.path.join(csvdir, f"{ex}_adaptive_k{k}.csv"))
              and k not in spec.get("skip_k", [])]
        if not ks:
            continue
        if spec["exact"]:
            out += [plot_errors(ex, k, csvdir, pdfdir, ext) for k in ks]
            out.append(plot_effectivity(ex, ks, csvdir, pdfdir, ext))
        else:
            out.append(plot_estimator(ex, ks, csvdir, pdfdir, ext))
        out.append(plot_meshes(ex, ks, csvdir, pdfdir, ext))
    for o in out:
        print("  wrote", os.path.relpath(o, ROOT))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csvdir", default="results/csv/references")
    ap.add_argument("--pdfdir", default="results/pdf/05_references")
    a = ap.parse_args()
    absol = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main(absol(a.csvdir), absol(a.pdfdir))
