#!/usr/bin/env python3
"""
plot_adaptive.py -- adaptive vs uniform refinement on the singular Signorini
benchmark u = -r^{3/2} sin(3 theta/2) + y^2, from the CSVs of adaptive_study.jl.
<bench> = singular (transition point x0 = 0, a mesh vertex) or offset
(x0 = 1/(2 pi), never a mesh vertex); every bench whose CSVs exist is plotted.

    <pdfdir>/adaptive_<bench>_k<k>.pdf     errors and estimator against ndof:
                                           adaptive (line + filled marker) vs
                                           uniform (thin line + hollow marker);
                                           triangles: optimal adaptive orders
    <pdfdir>/adaptive_<bench>_meshes.pdf   final adaptive mesh per k, whole domain
                                           and a zoom on the contact/separation point
    <pdfdir>/adaptive_[offset_]signorini_vs_linear_errors.pdf, ..._eff.pdf
                                           if adaptive_study.jl --problem=linear was run:
                                           the same loop without contact condition
    <pdfdir>/adaptive_<bench>_sigma.pdf    ||sigma - sigma_h||: Signorini vs linear,
                                           adaptive vs uniform, one panel per k

usage (from the root of SignoriniPostprocessing.jl):

    python3 studies/adaptivity/plot_adaptive.py [--csvdir results/csv] [--pdfdir results/pdf/04_adaptivity]
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
import plot_convergence as pc          # shared style: rcParams, inks, colours, read_csv

BENCH = {"singular": (0.0, r"x_0=0"), "offset": (1 / (2 * np.pi), r"x_0=1/(2\pi)")}

# (style key, column, label, optimal order r(k) in h-units, i.e. N^{-r/2})
CURVES = [("u",    "err_u_L2",       r"$\Vert u-u_h\Vert_{L^2}$",             lambda k: k + 1),
          ("nu",   "err_nu_L2",      r"$\Vert u-\nu_h\Vert_{L^2}$",           lambda k: k + 3),
          ("flux", "err_sigma_L2",   r"$\Vert\sigma-\sigma_h\Vert_{L^2}$",     lambda k: k + 2)]


def triangle(ax, x, y, r, side="below"):
    """Slope -r/2 in ndof (= O(N^{-r/2})) along the last decade of the adaptive curve,
    drawn under it (side="below") or over it (side="above")."""
    x2 = x[-1];  x1 = x2 / 10
    xa, xb = x1 * 10 ** 0.25, x1 * 10 ** 0.75
    at = lambda s: np.exp(np.interp(np.log(s), np.log(x), np.log(y)))
    expo = f"{r // 2}" if r % 2 == 0 else f"{r}/2"
    txt = rf"$\mathcal{{O}}(N^{{-{expo}}})$"
    if side == "above":
        ya = at(xa) * 2.4;  yb = ya * (xb / xa) ** (-r / 2)
        ax.plot([xa, xb, xb, xa], [ya, ya, yb, ya], color=pc.MUTED, lw=0.8)
        ax.text(np.sqrt(xa * xb), ya * 1.15, txt, ha="center", va="bottom", fontsize=7.5,
                color=pc.INK2)
    else:
        ya = at(xa) * 0.45;  yb = ya * (xb / xa) ** (-r / 2)
        ax.plot([xa, xb, xa, xa], [ya, yb, yb, ya], color=pc.MUTED, lw=0.8)
        ax.text(np.sqrt(xa * xb), yb / 1.15, txt, ha="center", va="top", fontsize=7.5,
                color=pc.INK2)


def plot_errors(k, csvdir, outdir, ext, bench="singular"):
    ad = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_k{k}.csv"))
    un = pc.read_csv(os.path.join(csvdir, f"uniform_{bench}_k{k}.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for key, col, label, order in CURVES:
        color, marker = pc.STYLE[key]
        ax.loglog(ad["ndof"], ad[col], color=color, marker=marker, lw=1.5, ms=4.5,
                  mec=pc.SURFACE, mew=0.6, zorder=3)
        ax.loglog(un["ndof"], un[col], color=color, lw=0.8, marker=marker, ms=7,
                  mfc="none", mec=color, mew=1.0, zorder=2)
        if key != "u":       # triangles only for the postprocessed quantities
            triangle(ax, ad["ndof"], ad[col], order(k))
    ax.loglog(ad["ndof"], ad["eta"], color=pc.INK2, lw=1.0, zorder=3)

    ax.set_title(rf"Signorini, $u\in H^{{5/2-\varepsilon}}$, ${BENCH[bench][1]}$" + "\n" +
                 rf"$k={k}$: adaptive vs uniform", color=pc.INK)
    ax.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$")
    ax.set_ylabel("error")
    ax.grid(True, which="major");  ax.grid(False, which="minor")
    ax.margins(x=0.06, y=0.1)

    handles = [Line2D([], [], color=pc.STYLE[key][0], marker=pc.STYLE[key][1], lw=1.5, ms=5,
                      mec=pc.SURFACE, label=label) for key, _, label, _ in CURVES]
    handles += [Line2D([], [], color=pc.INK2, lw=1.0, label=r"$\tilde\eta$ (adaptive)"),
                Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                       label="adaptive (Dörfler, NVB)"),
                Line2D([], [], color=pc.INK2, marker="o", lw=0.8, ms=6, mfc="none",
                       label="uniform")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, labelcolor=pc.INK, columnspacing=1.2)
    out = os.path.join(outdir, f"adaptive_{bench}_k{k}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def read_mesh(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a.reshape(-1, 3, 2)


def zoom_for(tris):
    """Half-width of the zoom window: 15 times the smallest cell diameter, so the
    grading towards the contact/separation point is visible at every degree."""
    diam = np.max(np.linalg.norm(tris[:, [0, 1, 2], :] - tris[:, [1, 2, 0], :], axis=2), axis=1)
    return float(np.clip(15 * diam.min(), 1e-6, 0.1))


def plot_meshes(ks, csvdir, outdir, ext, bench="singular"):
    x0 = BENCH[bench][0]
    fig, axs = plt.subplots(2, len(ks), figsize=(2.6 * len(ks), 4.6), squeeze=False)
    for j, k in enumerate(ks):
        tris = read_mesh(os.path.join(csvdir, f"adaptive_{bench}_k{k}_mesh.csv"))
        zoom = zoom_for(tris)
        for i, (lim, title) in enumerate(((None, f"$k={k}$: {len(tris)} triangles"),
                                          (zoom, rf"zoom $|x-x_0|,y\leq$ {zoom:.1e}"))):
            ax = axs[i, j]
            ax.add_collection(PolyCollection(tris, facecolors="none", edgecolors=pc.INK2,
                                             linewidths=0.15 if i == 0 else 0.3))
            # Gamma_S: separation (x < x0) and contact (x > x0)
            lw = 2.0 if i == 0 else 1.2
            ax.plot([-0.5, x0], [0, 0], color=pc.STYLE["u"][0], lw=lw, solid_capstyle="butt")
            ax.plot([x0, 0.5], [0, 0], color=pc.STYLE["flux"][0], lw=lw, solid_capstyle="butt")
            if lim is None:
                ax.set_xlim(-0.5, 0.5);  ax.set_ylim(0, 1)
            else:
                ax.set_xlim(x0 - lim, x0 + lim);  ax.set_ylim(0, lim)
            ax.set_aspect("equal");  ax.set_xticks([]);  ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, color=pc.INK, fontsize=9)
    handles = [Line2D([], [], color=pc.STYLE["flux"][0], lw=2, label=r"$\Gamma_S$ contact ($x>x_0$)"),
               Line2D([], [], color=pc.STYLE["u"][0], lw=2, label=r"$\Gamma_S$ separation ($x<x_0$)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(r"final adaptive meshes driven by $\tilde\eta_K=\Vert\nabla\varepsilon_K\Vert_K$, "
                 rf"${BENCH[bench][1]}$", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(outdir, f"adaptive_{bench}_meshes.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


KSTYLE = {0: ("#2a78d6", "o"), 1: ("#eb6834", "s"), 2: ("#1baf7a", "^"), 3: ("#eda100", "D")}


def plot_vs_linear(ks, csvdir, outdir, ext, bench="singular"):
    """Same adaptive loop, Signorini (line + filled marker) against the linear problem
    with the same u and Dirichlet data on all of the boundary (thin line + hollow marker)."""
    out = []
    pre = "" if bench == "singular" else f"{bench}_"
    # errors of the postprocessed solution, one panel per degree
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.3), squeeze=False)
    for ax, k in zip(axs[0], ks):
        s = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_k{k}.csv"))
        l = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_linear_k{k}.csv"))
        for key, col in (("nu", "err_nu_L2"), ("flux", "err_sigma_L2")):
            color, marker = pc.STYLE[key]
            ax.loglog(s["ndof"], s[col], color=color, marker=marker, lw=1.5, ms=4, mec=pc.SURFACE,
                      mew=0.5, zorder=3)
            ax.loglog(l["ndof"], l[col], color=color, marker=marker, lw=0.7, ms=5, mfc="none",
                      mec=color, mew=0.8, zorder=2)
        ax.set_title(rf"$k={k}$", color=pc.INK)
        ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel("error")
    handles = [Line2D([], [], color=pc.STYLE["nu"][0], marker=pc.STYLE["nu"][1], lw=1.5, ms=5,
                      mec=pc.SURFACE, label=r"$\Vert u-\nu_h\Vert_{L^2}$"),
               Line2D([], [], color=pc.STYLE["flux"][0], marker=pc.STYLE["flux"][1], lw=1.5, ms=5,
                      mec=pc.SURFACE, label=r"$\Vert\sigma-\sigma_h\Vert_{L^2}$"),
               Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                      label="Signorini"),
               Line2D([], [], color=pc.INK2, marker="o", lw=0.7, ms=5, mfc="none",
                      label=r"linear (Dirichlet on all of $\partial\Omega$)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(r"adaptive refinement, $u\in H^{5/2-\varepsilon}$, " + rf"${BENCH[bench][1]}$" +
                 ": Signorini vs linear", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.append(os.path.join(outdir, f"adaptive_{pre}signorini_vs_linear_errors.{ext}"))
    fig.savefig(out[-1], bbox_inches="tight");  plt.close(fig)

    # effectivity index along the adaptive loop
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for k in ks:
        color, marker = KSTYLE[k]
        s = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_k{k}.csv"))
        l = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_linear_k{k}.csv"))
        ax.semilogx(s["ndof"], s["eff"], color=color, marker=marker, lw=1.5, ms=4,
                    mec=pc.SURFACE, mew=0.5, zorder=3)
        ax.semilogx(l["ndof"], l["eff"], color=color, marker=marker, lw=0.7, ms=5, mfc="none",
                    mec=color, mew=0.8, zorder=2)
    ax.set_ylim(0, 1.0)
    ax.set_title(r"effectivity index $\tilde\eta/E$ along the adaptive loop, " +
                 rf"${BENCH[bench][1]}$", color=pc.INK)
    ax.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$")
    ax.set_ylabel(r"$\tilde\eta\,/\,E$")
    ax.grid(True, which="major");  ax.grid(False, which="minor")
    handles = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, mec=pc.SURFACE, label=f"$k={k}$")
               for k, (c, m) in KSTYLE.items() if k in ks]
    handles += [Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                       label="Signorini"),
                Line2D([], [], color=pc.INK2, marker="o", lw=0.7, ms=5, mfc="none", label="linear")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, labelcolor=pc.INK)
    out.append(os.path.join(outdir, f"adaptive_{pre}signorini_vs_linear_eff.{ext}"))
    fig.savefig(out[-1], bbox_inches="tight");  plt.close(fig)
    return out


def plot_sigma(ks, csvdir, outdir, ext, bench="singular"):
    """||sigma - sigma_h|| per degree: Signorini (thick line, filled) vs linear (thin line,
    hollow), adaptive (circles) vs uniform (squares); triangle: optimal adaptive order."""
    color = pc.STYLE["flux"][0]
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.3), squeeze=False)
    for ax, k in zip(axs[0], ks):
        for ref, marker in (("adaptive", "o"), ("uniform", "s")):
            for prob, lw, filled in (("", 1.5, True), ("_linear", 0.7, False)):
                d = pc.read_csv(os.path.join(csvdir, f"{ref}_{bench}{prob}_k{k}.csv"))
                ax.loglog(d["ndof"], d["err_sigma_L2"], color=color, marker=marker, lw=lw,
                          ms=4 if filled else 5, mfc=color if filled else "none",
                          mec=pc.SURFACE if filled else color, mew=0.5 if filled else 0.8,
                          zorder=3 if filled else 2)
        ad = pc.read_csv(os.path.join(csvdir, f"adaptive_{bench}_k{k}.csv"))
        triangle(ax, ad["ndof"], ad["err_sigma_L2"], k + 2)
        ax.set_title(rf"$k={k}$", color=pc.INK)
        ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel(r"$\Vert\sigma-\sigma_h\Vert_{L^2}$")
    handles = [Line2D([], [], color=color, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                      label="adaptive, Signorini"),
               Line2D([], [], color=color, marker="o", lw=0.7, ms=5, mfc="none",
                      label="adaptive, linear"),
               Line2D([], [], color=color, marker="s", lw=1.5, ms=5, mec=pc.SURFACE,
                      label="uniform, Signorini"),
               Line2D([], [], color=color, marker="s", lw=0.7, ms=5, mfc="none",
                      label="uniform, linear")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(r"$\Vert\sigma-\sigma_h\Vert_{L^2}$, $u\in H^{5/2-\varepsilon}$, " +
                 rf"${BENCH[bench][1]}$: Signorini vs linear", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(outdir, f"adaptive_{bench}_sigma.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(csvdir, pdfdir, ext="pdf"):
    os.makedirs(pdfdir, exist_ok=True)
    out = []
    exists = lambda name: os.path.exists(os.path.join(csvdir, name))
    for bench in BENCH:
        ks = [k for k in range(4) if exists(f"adaptive_{bench}_k{k}.csv")]
        out += [plot_errors(k, csvdir, pdfdir, ext, bench) for k in ks]
        if ks:
            out.append(plot_meshes(ks, csvdir, pdfdir, ext, bench))
        kl = [k for k in ks if exists(f"adaptive_{bench}_linear_k{k}.csv")]
        if kl:
            out += plot_vs_linear(kl, csvdir, pdfdir, ext, bench)
            out.append(plot_sigma(kl, csvdir, pdfdir, ext, bench))
    for o in out:
        print("  wrote", os.path.relpath(o, ROOT))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csvdir", default="results/csv")
    ap.add_argument("--pdfdir", default="results/pdf/04_adaptivity")
    a = ap.parse_args()
    absol = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main(absol(a.csvdir), absol(a.pdfdir))
