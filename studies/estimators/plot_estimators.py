#!/usr/bin/env python3
"""
plot_estimators.py -- full estimator eta (Estimators.contact_estimator) vs the
built-in eta_tilde = ||grad eps_h||, from the CSVs of compare_estimators.jl.

Examples with a known solution (singular, offset, bgk72, fhs42):
    <ex>_effectivity.pdf   eta / E_s (strong norm, Thm 3.9) and eta_tilde / E (weak norm, Thm 3.6)
                           along the eta-driven adaptive loop and under uniform refinement
    <ex>_k<k>.pdf          ||sigma - sigma_h||, ||u - nu_h||, E_s, eta and eta_tilde against ndof
    <ex>_drivers.pdf       adaptivity driven by eta (filled) vs driven by eta_tilde (hollow)
Examples with an unknown solution (fhs43, bgk731):
    <ex>_estimators.pdf    eta and eta_tilde, adaptive (eta-driven) vs uniform
Contact-term variants (contact_terms.jl, f constant so only the contact can be missed):
    contact_variants_<ex>.pdf  eta / E_s for the variants A, B, Bf, Bs, C of the term on Gamma_S
    contact_ratio.pdf          contact term of eta / boundary error on the exact contact zone
    linear_vs_signorini.pdf    eta / E_s: Signorini vs the same u without contact (linear_check.jl)
    bgk72_oscillation.pdf      eta / E_s and osc / E_s for Bartels §7.2 (f not in P_k)
All examples:
    <ex>_components.pdf    share of each term in eta^2 along the adaptive loop
    <ex>_meshes.pdf        final eta-driven adaptive meshes

usage (from the root of SignoriniPostprocessing.jl):

    python3 studies/estimators/plot_estimators.py [--csvdir results/csv/estimators]
                                                  [--pdfdir results/pdf/06_estimators]
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
import plot_adaptive as pa             # slope triangles, k colours, read_mesh

EX = {
    "singular": dict(title=r"$u=-r^{3/2}\sin(\frac{3}{2}\theta)+y^2$, $x_0=0$", box=(-0.5, 0.5, 0, 1),
                     exact=True, eta_tilde_csv="results/csv/adaptive_singular_k{k}.csv"),
    "offset":   dict(title=r"$u=-r^{3/2}\sin(\frac{3}{2}\theta)+y^2$, $x_0=1/(2\pi)$", box=(-0.5, 0.5, 0, 1),
                     exact=True, eta_tilde_csv="results/csv/adaptive_offset_k{k}.csv"),
    "bgk72":    dict(title=r"Bartels–Gudi–Kaltenbach §7.2", box=(0, 1, 0, 1),
                     exact=True, eta_tilde_csv="results/csv/references/bgk72_adaptive_k{k}.csv"),
    "fhs42":    dict(title=r"Führer–Heuer–Stephan §4.2", box=(-1, 1, 0, 1),
                     exact=True, eta_tilde_csv="results/csv/references/fhs42_adaptive_k{k}.csv"),
    "fhs43":    dict(title=r"Führer–Heuer–Stephan §4.3 (L-shape)", box=(-1, 1, -1, 1), exact=False),
    "bgk731":   dict(title=r"Bartels–Gudi–Kaltenbach §7.3.1 (obstacle)", box=(-1, 1, -1, 1), exact=False),
}
INK_E = pc.INK                     # the strong error E_s: a quantity of its own, in ink
PARTS = [("part_tilde", r"$\Vert\nabla\varepsilon_h\Vert^2$", "#2a78d6", "o"),
         ("part_flux", r"$\Vert\sigma_h-\nabla\nu_h\Vert^2$", "#eb6834", "s"),
         ("part_jump", r"jumps $h_F^{-1}\Vert[\![\nu_h]\!]\Vert^2$", "#1baf7a", "^"),
         ("part_dirichlet", r"$\Gamma_D$: $h_F^{-1}\Vert u_D-\nu_h\Vert^2$", "#eda100", "D"),
         ("part_contact", r"$\Gamma_S$: contact terms", "#e87ba4", "v")]


def csv(csvdir, ex, kind, k):
    return pc.read_csv(os.path.join(csvdir, f"{ex}_{kind}_k{k}.csv"))


def ks_of(csvdir, ex):
    return [k for k in range(4) if os.path.exists(os.path.join(csvdir, f"{ex}_adaptive_k{k}.csv"))]


def style_handles(ks):
    h = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, mec=pc.SURFACE, label=f"$k={k}$")
         for k, (c, m) in pa.KSTYLE.items() if k in ks]
    return h + [Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                       label=r"adaptive ($\eta$-driven)"),
                Line2D([], [], color=pc.INK2, marker="o", lw=0.8, ms=6, mfc="none", label="uniform")]


def finish(fig, out):
    fig.savefig(out, bbox_inches="tight");  plt.close(fig)
    return out


def plot_effectivity(ex, ks, csvdir, outdir, ext):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    top = 1.1
    for k in ks:
        color, marker = pa.KSTYLE[k]
        for kind, lw, filled in (("adaptive", 1.5, True), ("uniform", 0.8, False)):
            d = csv(csvdir, ex, kind, k)
            for a, col in ((ax, "eff"), (bx, "eff_tilde")):
                a.semilogx(d["ndof"], d[col], color=color, marker=marker, lw=lw,
                           ms=4 if filled else 6, mfc=color if filled else "none",
                           mec=pc.SURFACE if filled else color, mew=0.5 if filled else 0.9,
                           zorder=3 if filled else 2)
                top = max(top, 1.05 * np.nanmax(d[col]))
    ax.set_title(r"estimator $\eta$ (B): $\eta\,/\,E_s$ (norm of Thm 3.9)", color=pc.INK)
    bx.set_title(r"built-in: $\tilde\eta\,/\,E$ (norm of Thm 3.6)", color=pc.INK)
    for a in (ax, bx):
        a.set_ylim(0, top);  a.axhline(1.0, color=pc.MUTED, lw=0.8)
        a.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$")
        a.grid(True, which="major");  a.grid(False, which="minor")
    ax.set_ylabel("effectivity index")
    fig.legend(handles=style_handles(ks), loc="lower center", ncol=6, frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(EX[ex]["title"], color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish(fig, os.path.join(outdir, f"{ex}_effectivity.{ext}"))


def plot_errors(ex, k, csvdir, outdir, ext):
    ad, un = csv(csvdir, ex, "adaptive", k), csv(csvdir, ex, "uniform", k)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    curves = [(pc.STYLE["flux"][0], "o", "err_sigma_L2", r"$\Vert\sigma-\sigma_h\Vert_{L^2}$", k + 2),
              (pc.STYLE["nu"][0], "^", "err_nu_L2", r"$\Vert u-\nu_h\Vert_{L^2}$", k + 3),
              (INK_E, "D", "err_strong", r"$E_s$ (strong norm)", None)]
    for color, marker, col, label, order in curves:
        ax.loglog(ad["ndof"], ad[col], color=color, marker=marker, lw=1.5, ms=4.5,
                  mec=pc.SURFACE, mew=0.6, zorder=3)
        ax.loglog(un["ndof"], un[col], color=color, marker=marker, lw=0.8, ms=6.5,
                  mfc="none", mec=color, mew=0.9, zorder=2)
        if order is not None:
            pa.triangle(ax, ad["ndof"], ad[col], order)
    ax.loglog(ad["ndof"], ad["eta"], color=pc.INK2, lw=1.4, zorder=3)
    ax.loglog(ad["ndof"], ad["eta_tilde"], color=pc.MUTED, lw=0.9, zorder=3)
    ax.set_title(EX[ex]["title"] + "\n" + rf"$k={k}$, adaptive driven by $\eta$", color=pc.INK, fontsize=9)
    ax.set_xlabel(r"degrees of freedom $N$ of $(\sigma_h, u_h)$")
    ax.set_ylabel("error / estimator")
    ax.grid(True, which="major");  ax.grid(False, which="minor")
    ax.margins(x=0.06, y=0.1)
    handles = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, mec=pc.SURFACE, label=l)
               for c, m, _, l, _ in curves]
    handles += [Line2D([], [], color=pc.INK2, lw=1.4, label=r"$\eta$ (full)"),
                Line2D([], [], color=pc.MUTED, lw=0.9, label=r"$\tilde\eta$ (built-in)"),
                Line2D([], [], color=pc.INK2, marker="o", lw=0.8, ms=6, mfc="none", label="uniform")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, labelcolor=pc.INK, columnspacing=1.2)
    return finish(fig, os.path.join(outdir, f"{ex}_k{k}.{ext}"))


def plot_drivers(ex, ks, csvdir, outdir, ext):
    """Same example and degree, adaptivity driven by eta (filled) or by eta_tilde (hollow)."""
    ks = [k for k in ks if os.path.exists(os.path.join(ROOT, EX[ex]["eta_tilde_csv"].format(k=k)))]
    if not ks:
        return None
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.3), squeeze=False)
    for ax, k in zip(axs[0], ks):
        full = csv(csvdir, ex, "adaptive", k)
        tild = pc.read_csv(os.path.join(ROOT, EX[ex]["eta_tilde_csv"].format(k=k)))
        for key, col in (("flux", "err_sigma_L2"), ("nu", "err_nu_L2")):
            color, marker = pc.STYLE[key]
            ax.loglog(full["ndof"], full[col], color=color, marker=marker, lw=1.5, ms=4,
                      mec=pc.SURFACE, mew=0.5, zorder=3)
            ax.loglog(tild["ndof"], tild[col], color=color, marker=marker, lw=0.7, ms=5,
                      mfc="none", mec=color, mew=0.8, zorder=2)
        ax.set_title(rf"$k={k}$", color=pc.INK);  ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel("error")
    handles = [Line2D([], [], color=pc.STYLE["flux"][0], marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                      label=r"$\Vert\sigma-\sigma_h\Vert_{L^2}$"),
               Line2D([], [], color=pc.STYLE["nu"][0], marker="^", lw=1.5, ms=5, mec=pc.SURFACE,
                      label=r"$\Vert u-\nu_h\Vert_{L^2}$"),
               Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE,
                      label=r"driven by $\eta$ (full)"),
               Line2D([], [], color=pc.INK2, marker="o", lw=0.7, ms=5, mfc="none",
                      label=r"driven by $\tilde\eta$ (built-in)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(EX[ex]["title"] + ": which estimator refines better", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish(fig, os.path.join(outdir, f"{ex}_drivers.{ext}"))


def plot_unknown(ex, ks, csvdir, outdir, ext):
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.3), squeeze=False)
    for ax, k in zip(axs[0], ks):
        ad, un = csv(csvdir, ex, "adaptive", k), csv(csvdir, ex, "uniform", k)
        for col, color, lw in (("eta", pc.INK2, 1.5), ("eta_tilde", pc.MUTED, 1.0)):
            ax.loglog(ad["ndof"], ad[col], color=color, marker="o", lw=lw, ms=3.5,
                      mec=pc.SURFACE, mew=0.4, zorder=3)
            ax.loglog(un["ndof"], un[col], color=color, marker="o", lw=0.7, ms=5.5,
                      mfc="none", mec=color, mew=0.8, zorder=2)
        pa.triangle(ax, ad["ndof"], ad["eta"], k + 2)
        ax.set_title(rf"$k={k}$", color=pc.INK);  ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel("estimator")
    handles = [Line2D([], [], color=pc.INK2, lw=1.5, label=r"$\eta$ (full)"),
               Line2D([], [], color=pc.MUTED, lw=1.0, label=r"$\tilde\eta$ (built-in)")] + style_handles([])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(EX[ex]["title"] + r" (unknown $u$)", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish(fig, os.path.join(outdir, f"{ex}_estimators.{ext}"))


def plot_components(ex, ks, csvdir, outdir, ext):
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 3.1), squeeze=False, sharey=True)
    used = set()
    for ax, k in zip(axs[0], ks):
        d = csv(csvdir, ex, "adaptive", k)
        tot = sum(d[c] for c, *_ in PARTS)
        for col, label, color, marker in PARTS:
            if np.nanmax(d[col]) <= 0:
                continue
            used.add(col)
            ax.semilogx(d["ndof"], d[col] / tot, color=color, marker=marker, lw=1.3, ms=3.5,
                        mec=pc.SURFACE, mew=0.4)
        ax.set_ylim(0, 1);  ax.set_title(rf"$k={k}$", color=pc.INK);  ax.set_xlabel(r"$N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0, 0].set_ylabel(r"share of $\eta^2$")
    handles = [Line2D([], [], color=c, marker=m, lw=1.3, ms=4, mec=pc.SURFACE, label=l)
               for col, l, c, m in PARTS if col in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(EX[ex]["title"] + r": contributions to $\eta^2$ ($\eta$-driven adaptive loop)",
                 color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return finish(fig, os.path.join(outdir, f"{ex}_components.{ext}"))


def plot_meshes(ex, ks, csvdir, outdir, ext):
    x0, x1, y0, y1 = EX[ex]["box"]
    aspect = (x1 - x0) / (y1 - y0)
    fig, axs = plt.subplots(1, len(ks), figsize=(1.0 + 2.4 * aspect * len(ks), 2.8), squeeze=False)
    for ax, k in zip(axs[0], ks):
        tris = pa.read_mesh(os.path.join(csvdir, f"{ex}_adaptive_k{k}_mesh.csv"))
        ax.add_collection(PolyCollection(tris, facecolors="none", edgecolors=pc.INK2, linewidths=0.12))
        ax.set_xlim(x0, x1);  ax.set_ylim(y0, y1);  ax.set_aspect("equal")
        ax.set_xticks([]);  ax.set_yticks([]);  ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"$k={k}$: {len(tris)} triangles", color=pc.INK, fontsize=9)
    fig.suptitle(EX[ex]["title"] + r": final $\eta$-driven meshes", color=pc.INK, fontsize=10)
    fig.tight_layout()
    return finish(fig, os.path.join(outdir, f"{ex}_meshes.{ext}"))


VARIANTS = [("eff_A",  "A: no contact term",                 pc.MUTED,  "o", 1.0),
            ("eff_B",  "B: active + inactive, Bernstein β (chosen)", "#2a78d6", "s", 1.8),
            ("eff_Bf", "Bf: facet-wise classification",      "#eb6834", "^", 1.0),
            ("eff_Bs", r"Bs: classification by $\sigma_h\cdot n>0$", "#1baf7a", "D", 1.0),
            ("eff_C",  r"C: B + $|\langle\sigma_h\cdot n,\nu_h-\chi\rangle|$", "#eda100", "v", 1.0)]


def plot_contact_variants(ex, csvdir, outdir, ext, ymax=2.0):
    """Effectivity eta_variant / E_s of the contact-term variants of contact_terms.jl."""
    ks = [k for k in range(4) if os.path.exists(os.path.join(csvdir, f"{ex}_adaptive_k{k}.csv"))]
    if not ks:
        return None
    fig, axs = plt.subplots(2, len(ks), figsize=(3.0 * len(ks), 5.6), squeeze=False, sharey=True)
    for j, k in enumerate(ks):
        for i, kind in enumerate(("adaptive", "uniform")):
            ax = axs[i, j];  d = csv(csvdir, ex, kind, k)
            for col, _, color, marker, lw in VARIANTS:
                ax.semilogx(d["ndof"], np.minimum(d[col], ymax), color=color, marker=marker,
                            lw=lw, ms=3.5, mec=pc.SURFACE, mew=0.4, zorder=3 if col == "eff_B" else 2)
            ax.axhline(1.0, color=pc.MUTED, lw=0.6)
            ax.set_ylim(0, ymax);  ax.grid(True, which="major");  ax.grid(False, which="minor")
            ax.set_title(rf"{kind}, $k={k}$", color=pc.INK, fontsize=9)
            if i == 1:
                ax.set_xlabel(r"$N$")
        axs[0, 0].set_ylabel(r"$\eta\,/\,E_s$");  axs[1, 0].set_ylabel(r"$\eta\,/\,E_s$")
    handles = [Line2D([], [], color=c, marker=m, lw=lw, ms=4, mec=pc.SURFACE, label=l)
               for _, l, c, m, lw in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(EX[ex]["title"] + r": contact-term variants (values above 2 clipped; "
                 r"D = B + $h_F\Vert\sigma_h\cdot n\Vert^2_{\{\nu_h>\chi\}}$ is $10^2$–$10^7$, not shown)",
                 color=pc.INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return finish(fig, os.path.join(outdir, f"contact_variants_{ex}.{ext}"))


def _k_series(ax, files, col, ks, ylog=False):
    """One curve per k: adaptive (line + filled marker) and uniform (thin line + hollow)."""
    for k in ks:
        color, marker = pa.KSTYLE[k]
        for kind, lw, filled in (("adaptive", 1.5, True), ("uniform", 0.8, False)):
            path = files(kind, k)
            if not os.path.exists(path):
                continue
            d = pc.read_csv(path)
            y = col(d) if callable(col) else d[col]
            plot = ax.loglog if ylog else ax.semilogx
            plot(d["ndof"], y, color=color, marker=marker, lw=lw, ms=4 if filled else 6,
                 mfc=color if filled else "none", mec=pc.SURFACE if filled else color,
                 mew=0.5 if filled else 0.9, zorder=3 if filled else 2)


def plot_linear_check(outdir, ext, ctdir, lindir):
    """eta / E_s per level: Signorini (variant B) against the same u without contact."""
    fig, axs = plt.subplots(2, 2, figsize=(8.8, 6.2), sharey=True)
    for j, (bench, lab) in enumerate((("singular", r"transition at a vertex ($x_0=0$)"),
                                      ("offset", r"transition inside a facet ($x_0=1/(2\pi)$)"))):
        for i, kind in enumerate(("uniform", "adaptive")):
            ax = axs[i, j]
            for k in range(4):
                color, marker = pa.KSTYLE[k]
                s = os.path.join(ctdir, f"{bench}_{kind}_k{k}.csv")
                l = os.path.join(lindir, f"{bench}_{kind}_k{k}.csv")
                if os.path.exists(s):
                    d = pc.read_csv(s)
                    ax.semilogx(d["ndof"], d["eff_B"], color=color, marker=marker, lw=1.5, ms=4,
                                mec=pc.SURFACE, mew=0.5, zorder=3)
                if os.path.exists(l):
                    d = pc.read_csv(l)
                    ax.semilogx(d["ndof"], d["eff"], color=color, marker=marker, lw=0.8, ms=6,
                                mfc="none", mec=color, mew=0.9, zorder=2)
            ax.set_ylim(0, 1.1);  ax.axhline(1.0, color=pc.MUTED, lw=0.6)
            ax.set_title(f"{lab}, {kind}", color=pc.INK, fontsize=9)
            ax.grid(True, which="major");  ax.grid(False, which="minor")
            if i == 1:
                ax.set_xlabel(r"degrees of freedom $N$")
        axs[0, 0].set_ylabel(r"$\eta\,/\,E_s$");  axs[1, 0].set_ylabel(r"$\eta\,/\,E_s$")
    handles = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, mec=pc.SURFACE, label=f"$k={k}$")
               for k, (c, m) in pa.KSTYLE.items()]
    handles += [Line2D([], [], color=pc.INK2, marker="o", lw=1.5, ms=5, mec=pc.SURFACE, label="Signorini"),
                Line2D([], [], color=pc.INK2, marker="o", lw=0.8, ms=6, mfc="none",
                       label=r"same $u$, no contact (Dirichlet on $\partial\Omega$)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, labelcolor=pc.INK,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(r"estimator $\eta$ (B): Signorini vs the same problem without contact",
                 color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return finish(fig, os.path.join(outdir, f"linear_vs_signorini.{ext}"))


def plot_contact_ratio(outdir, ext, ctdir):
    """Contact term of eta against the boundary error on the contact zone."""
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=True)
    for ax, (bench, lab) in zip(axs, (("singular", r"$x_0=0$"), ("offset", r"$x_0=1/(2\pi)$"))):
        _k_series(ax, lambda kind, k: os.path.join(ctdir, f"{bench}_{kind}_k{k}.csv"),
                  "ratio_contact", range(4))
        ax.set_ylim(0, 3);  ax.axhline(1.0, color=pc.MUTED, lw=0.6)
        ax.set_title(lab, color=pc.INK);  ax.set_xlabel(r"degrees of freedom $N$")
        ax.grid(True, which="major");  ax.grid(False, which="minor")
    axs[0].set_ylabel(r"contact term / $(\sum_{F\subset\mathrm{contact}} h_F^{-1}\Vert u-\nu_h\Vert_F^2)^{1/2}$",
                      fontsize=8)
    fig.legend(handles=style_handles(range(4)), loc="lower center", ncol=6, frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(r"contact term of $\eta$ against the boundary error on the exact contact zone",
                 color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish(fig, os.path.join(outdir, f"contact_ratio.{ext}"))


def plot_oscillation(outdir, ext, csvdir, ex="bgk72"):
    """eta / E_s and the relative oscillation osc / E_s, osc = (h_K/pi)||f - Pi_k f||."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.3))
    files = lambda kind, k: os.path.join(csvdir, f"{ex}_{kind}_k{k}.csv")
    _k_series(ax, files, "eff", range(4))
    _k_series(bx, files, lambda d: d["osc"] / d["err_strong"], range(4))
    ax.set_ylim(0, 1.1);  ax.set_title(r"$\eta\,/\,E_s$", color=pc.INK)
    bx.set_ylim(0, 4.5);  bx.set_title(r"$\mathrm{osc}\,/\,E_s$, $\mathrm{osc}=(h_K/\pi)\Vert f-\Pi_k f\Vert_K$",
                                       color=pc.INK)
    for a in (ax, bx):
        a.axhline(1.0, color=pc.MUTED, lw=0.6);  a.set_xlabel(r"degrees of freedom $N$")
        a.grid(True, which="major");  a.grid(False, which="minor")
    fig.legend(handles=style_handles(range(4)), loc="lower center", ncol=6, frameon=False,
               labelcolor=pc.INK, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(EX[ex]["title"] + r" ($f\notin\mathrm{P}_k$)", color=pc.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish(fig, os.path.join(outdir, f"{ex}_oscillation.{ext}"))


def main(csvdir, pdfdir, ext="pdf", ctdir=None):
    os.makedirs(pdfdir, exist_ok=True)
    out = []
    ctdir = ctdir or os.path.join(ROOT, "results", "csv", "contact_terms")
    lindir = os.path.join(ROOT, "results", "csv", "linear_check")
    for ex in ("singular", "offset"):
        out.append(plot_contact_variants(ex, ctdir, pdfdir, ext))
    out.append(plot_contact_ratio(pdfdir, ext, ctdir))
    if os.path.isdir(lindir):
        out.append(plot_linear_check(pdfdir, ext, ctdir, lindir))
    if os.path.exists(os.path.join(csvdir, "bgk72_adaptive_k0.csv")):
        out.append(plot_oscillation(pdfdir, ext, csvdir))
    for ex, spec in EX.items():
        ks = ks_of(csvdir, ex)
        if not ks:
            continue
        if spec["exact"]:
            out.append(plot_effectivity(ex, ks, csvdir, pdfdir, ext))
            out += [plot_errors(ex, k, csvdir, pdfdir, ext) for k in ks]
            out.append(plot_drivers(ex, ks, csvdir, pdfdir, ext))
        else:
            out.append(plot_unknown(ex, ks, csvdir, pdfdir, ext))
        out.append(plot_components(ex, ks, csvdir, pdfdir, ext))
        out.append(plot_meshes(ex, ks, csvdir, pdfdir, ext))
    for o in out:
        if o:
            print("  wrote", os.path.relpath(o, ROOT))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csvdir", default="results/csv/estimators")
    ap.add_argument("--pdfdir", default="results/pdf/06_estimators")
    a = ap.parse_args()
    absol = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main(absol(a.csvdir), absol(a.pdfdir))
