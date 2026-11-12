#!/usr/bin/env python3
"""
plot_convergence.py -- log-log convergence plots from the CSVs of export_csv.jl.

One PDF per study and polynomial degree, error against the degrees of freedom
of the mixed system (sigma_h, u_h):

    <pdfdir>/01_convergence/signorini_k<k>.pdf     ||sigma - sigma_h||, ||u - u_h||
    <pdfdir>/01_convergence/postprocess_k<k>.pdf   ||u - u_h||, ||u - nu_h||, ||sigma - sigma_h||
    <pdfdir>/02_stenberg/stenberg_k<k>.pdf         Proposition 3.1: nu_h of (3.2) against
                                    utilde_h of (2.10) -- errors overlaid (left)
                                    and their relative difference (right)

Each curve carries a reference triangle with its theoretical order O(h^r)
(drawn with slope -r/2 in ndof, since ndof ~ h^-2 in 2D) and the observed
rate on the finest pair of meshes in the legend.

usage (from the root of SignoriniPostprocessing.jl):

    python3 studies/convergence/plot_convergence.py
    python3 studies/convergence/plot_convergence.py --csvdir results/csv --pdfdir results/pdf

The Stenberg comparison keeps ||grad_h(u - nu_h)|| against ||grad_h(u - utilde_h)||:
sigma_h is the same for both postprocessings, so ||sigma - sigma_h|| cannot compare them.

Needs numpy and matplotlib only.
"""
import argparse, csv, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Reference palette: categorical slots 1-3 (validated all-pairs, light mode),
# neutral inks, hairline grid.  Marker shape is the secondary encoding.
# Colour follows the quantity, identically in every figure:
#   "flux"  ||sigma - sigma_h||  (and ||grad_h(u - nu_h)|| in the Stenberg comparison)
#   "u"     ||u - u_h||
#   "nu"    ||u - nu_h||
STYLE = {"flux": ("#2a78d6", "o"), "u": ("#eb6834", "s"), "nu": ("#1baf7a", "^")}
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

# (style key, error column, rate column, legend label, theoretical order r(k) in h)
STUDIES = {
    "signorini": dict(
        title=lambda k: rf"Scalar Signorini, $k={k}$: $\mathrm{{BDM}}_{{{k+1}}}\times\mathrm{{P}}_{{{k}}}$",
        curves=[("flux", "err_sigma_L2", "rate_sigma_L2", r"$\Vert\sigma-\sigma_h\Vert_{L^2(\Omega)}$", lambda k: k + 2),
                ("u",    "err_u_L2",     "rate_u_L2",     r"$\Vert u-u_h\Vert_{L^2(\Omega)}$",           lambda k: k + 1)]),
    "postprocess": dict(
        title=lambda k: rf"Postprocessing (3.2), $k={k}$: $u_h\in\mathrm{{P}}_{{{k}}}$, $\nu_h\in\mathrm{{P}}_{{{k+2}}}$",
        curves=[("u",    "err_u_L2",       "rate_u_L2",       r"$\Vert u-u_h\Vert_{L^2(\Omega)}$",             lambda k: k + 1),
                ("nu",   "err_nu_L2",      "rate_nu_L2",      r"$\Vert u-\nu_h\Vert_{L^2(\Omega)}$",           lambda k: k + 3),
                ("flux", "err_sigma_L2",   "rate_sigma_L2",   r"$\Vert\sigma-\sigma_h\Vert_{L^2(\Omega)}$",    lambda k: k + 2)]),
}


def read_csv(path):
    """Columns of a CSV as float arrays; empty cells (no rate on the first row) are NaN."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows]) for c in rows[0]}


def slope_triangle(ax, x, y, r, color, label="left"):
    """Triangle of slope -r/2 (i.e. O(h^r)) under the last segment of the curve (x, y).

    `label` places the O(h^r) text left of the vertical leg ("left") or under the
    horizontal leg ("below").
    """
    x1, x2 = x[-2], x[-1]
    xa, xb = x1 * (x2 / x1) ** 0.25, x1 * (x2 / x1) ** 0.75
    # value of the curve at xa (linear in log-log), pushed down below it
    ya = np.exp(np.interp(np.log(xa), np.log(x[-2:]), np.log(y[-2:]))) * 0.5
    yb = ya * (xb / xa) ** (-r / 2)
    ax.plot([xa, xb, xa, xa], [ya, yb, yb, ya], color=color, lw=0.8, solid_joinstyle="miter")
    txt = rf"$\mathcal{{O}}(h^{{{r}}})$"
    if label == "below":
        ax.text(np.sqrt(xa * xb), yb / 1.15, txt, ha="center", va="top", fontsize=7.5, color=INK2)
    else:
        ax.text(xa / 1.06, np.sqrt(ya * yb), txt, ha="right", va="center", fontsize=7.5, color=INK2)


def plot_study(path, study, outdir, ext="pdf"):
    d = read_csv(path)
    k = int(d["k"][0])
    spec = STUDIES[study]
    x = d["ndof"]

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for key, ecol, rcol, label, order in spec["curves"]:
        color, marker = STYLE[key]
        y = d[ecol]
        rate = d[rcol][-1]
        ax.loglog(x, y, color=color, marker=marker, lw=1.5, ms=6,
                  mec=SURFACE, mew=0.8, label=f"{label}, rate {rate:.2f}", zorder=3)
        slope_triangle(ax, x, y, order(k), color)

    ax.set_title(spec["title"](k), color=INK)
    ax.set_xlabel(r"degrees of freedom of $(\sigma_h, u_h)$")
    ax.set_ylabel("error")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    ax.legend(loc="lower left", frameon=False, labelcolor=INK)
    ax.margins(x=0.08, y=0.12)

    out = os.path.join(outdir, f"{study}_k{k}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stenberg(path, outdir, ext="pdf"):
    """Proposition 3.1: nu_h (3.2) and utilde_h (2.10) should be the same function.

    Left: the error curves of both, nu_h as line + filled marker, utilde_h as a
    hollow ring of the same colour around it (coincident rings = same values).
    Right: the relative difference between the two, against machine epsilon.
    """
    d = read_csv(path)
    k = int(d["k"][0])
    x = d["ndof"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.4))

    pairs = [("nu",   "err_nu_L2",      "err_utilde_L2",      "rate_nu_L2",      "rate_utilde_L2",
              r"$\Vert u-\nu_h\Vert_{L^2}$", r"$\Vert u-\tilde u_h\Vert_{L^2}$", k + 3),
             ("flux", "err_grad_nu_L2", "err_grad_utilde_L2", "rate_grad_nu_L2", "rate_grad_utilde_L2",
              r"$\Vert\nabla_h(u-\nu_h)\Vert_{L^2}$", r"$\Vert\nabla_h(u-\tilde u_h)\Vert_{L^2}$", k + 2)]
    for key, eν, eũ, rν, rũ, lν, lũ, order in pairs:
        color, marker = STYLE[key]
        ax.loglog(x, d[eν], color=color, marker=marker, lw=1.5, ms=6, mec=SURFACE, mew=0.8,
                  label=f"{lν}  (3.2), rate {d[rν][-1]:.2f}", zorder=3)
        ax.loglog(x, d[eũ], ls="none", marker="o", ms=11, mfc="none", mec=color, mew=1.1,
                  label=f"{lũ}  (2.10), rate {d[rũ][-1]:.2f}", zorder=4)
        slope_triangle(ax, x, d[eν], order, color, label="below")
    ax.set_title("errors of both postprocessings", color=INK)
    ax.set_xlabel(r"degrees of freedom of $(\sigma_h, u_h)$")
    ax.set_ylabel("error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False,
              labelcolor=INK, columnspacing=1.2)
    ax.margins(x=0.08, y=0.12)

    eps = np.finfo(float).eps
    for key, col, lab in [("nu",   "reldiff_L2",   r"$\Vert\nu_h-\tilde u_h\Vert_{L^2}\,/\,\Vert\nu_h\Vert_{L^2}$"),
                          ("flux", "reldiff_grad", r"$\Vert\nabla_h(\nu_h-\tilde u_h)\Vert_{L^2}\,/\,\Vert\nabla_h\nu_h\Vert_{L^2}$")]:
        color, marker = STYLE[key]
        bx.loglog(x, d[col], color=color, marker=marker, lw=1.5, ms=6, mec=SURFACE, mew=0.8,
                  label=lab, zorder=3)
    bx.axhline(eps, color=MUTED, lw=0.8)
    bx.text(x[-1], eps / 1.3, r"machine $\varepsilon$", ha="right", va="top",
            fontsize=7.5, color=INK2)
    lo = min(np.nanmin(d["reldiff_L2"]), eps) / 4
    hi = max(np.nanmax(d["reldiff_grad"]), eps) * 30
    bx.set_ylim(lo, hi)
    bx.set_title("relative difference (3.2) vs (2.10)", color=INK)
    bx.set_xlabel(r"degrees of freedom of $(\sigma_h, u_h)$")
    bx.set_ylabel("relative difference")
    bx.legend(loc="upper left", frameon=False, labelcolor=INK)
    bx.margins(x=0.08)

    for a in (ax, bx):
        a.grid(True, which="major")
        a.grid(False, which="minor")
    fig.suptitle(rf"Proposition 3.1, $k={k}$: $\nu_h$ of (3.2) vs $\tilde u_h$ of (2.10), "
                 rf"both in $\mathrm{{P}}_{{{k+2}}}$", color=INK, fontsize=10)
    fig.tight_layout()

    out = os.path.join(outdir, f"stenberg_k{k}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _by_degree(csvdir, study):
    return sorted(glob.glob(os.path.join(csvdir, f"{study}_k*.csv")),
                  key=lambda f: int(re.search(r"_k(\d+)\.csv$", f).group(1)))


SUBDIR = {"signorini": "01_convergence", "postprocess": "01_convergence", "stenberg": "02_stenberg"}


def main(csvdir, pdfdir, ext="pdf"):
    for d in set(SUBDIR.values()):
        os.makedirs(os.path.join(pdfdir, d), exist_ok=True)
    written = []
    sub = lambda s: os.path.join(pdfdir, SUBDIR[s])
    jobs = [(s, lambda f, s=s: plot_study(f, s, sub(s), ext)) for s in STUDIES]
    jobs.append(("stenberg", lambda f: plot_stenberg(f, sub("stenberg"), ext)))
    for study, plot in jobs:
        files = _by_degree(csvdir, study)
        if not files:
            print(f"  no {study}_k*.csv in {csvdir} (run export_csv.jl first)")
        for f in files:
            written.append(plot(f))
            print("  wrote", os.path.relpath(written[-1], ROOT))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csvdir", default="results/csv", help="input directory (default results/csv)")
    ap.add_argument("--pdfdir", default="results/pdf",
                    help="output root (default results/pdf; files go to 01_convergence/, 02_stenberg/)")
    a = ap.parse_args()
    absol = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main(absol(a.csvdir), absol(a.pdfdir))
