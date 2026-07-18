"""Figure 1 - pyMNPBEM simulation pipeline & compute backend.

Horizontal data-flow (geometry/materials -> ComParticle -> Green/BEM assembly
-> solve -> sig -> observables) with the solver-variant fan-out on top and a
per-stage compute-backend matrix (CPU / GPU-fp64 / GPU-fp32 / multi-GPU) below.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import to_rgba

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})

# ----------------------------------------------------------------- palette
C = {
    "geo_fc": "#dbe9f6", "geo_ec": "#2f6aa8",
    "mat_fc": "#d8efe4", "mat_ec": "#2f8f6b",
    "cp_fc":  "#e9e1f4", "cp_ec":  "#6a4fa3",
    "ker_fc": "#fde6cf", "ker_ec": "#d9820f",
    "sol_fc": "#fbdcd4", "sol_ec": "#cf513c",
    "sig_fc": "#ededed", "sig_ec": "#555555",
    "obs_fc": "#d9efd6", "obs_ec": "#3a9a4e",
    "exc_fc": "#e7e2f1", "exc_ec": "#6f63ad",
    "grn_fc": "#eaf0f6", "grn_ec": "#5d7d9c",
    "slv_fc": "#eef2f8", "slv_ec": "#94a4be",
    "var_fc": "#f6f7fb", "var_ec": "#8a8fa3",
    "chip":   "#ffffff",
    "cpu":   "#9ecae1",
    "fp64":  "#a1d99b",
    "fp32":  "#fdae6b",
    "mgpu":  "#bcbddc",
    "empty": "#f0f0f0",
}

fig, ax = plt.subplots(figsize=(15.5, 9.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(cx, cy, w, h, title, sub=None, fc="#fff", ec="#333",
        tfs=11.5, sfs=8.6, tcol=None, rounding=1.4, lw=1.8, zorder=2, ls="-"):
    tcol = tcol or ec
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.15,rounding_size=%g" % rounding,
                       fc=fc, ec=ec, lw=lw, zorder=zorder, linestyle=ls)
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + h / 2 - 2.1, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=tcol, zorder=zorder + 1)
        ax.text(cx, cy + h / 2 - 4.9, sub, ha="center", va="top",
                fontsize=sfs, color="#222", linespacing=1.45, zorder=zorder + 1)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=tcol, zorder=zorder + 1)


def arrow(x0, y0, x1, y1, color="#3a3a3a", lw=2.3, style="-|>", ls="-", z=1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=18), zorder=z)


def chip(cx, cy, w, h, text, fs=8.6, ec="#555", fc=C["chip"], bold=False):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.8",
                       fc=fc, ec=ec, lw=1.3, zorder=4)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="#222", fontweight="bold" if bold else "normal", zorder=5)


# ============================================================ title
ax.text(1.5, 99.0, "pyMNPBEM  —  simulation pipeline & compute backend",
        ha="left", va="top", fontsize=14, fontweight="bold", color="#1b1b1b")

# ============================================================ (A) variant fan-out panel
vx0, vx1, vy0, vy1 = 30.5, 87.5, 78.9, 95.8
box((vx0 + vx1) / 2, (vy0 + vy1) / 2, vx1 - vx0, vy1 - vy0,
    "", None, fc=C["var_fc"], ec=C["var_ec"], lw=1.5, rounding=1.2)
ax.text((vx0 + vx1) / 2, vy1 - 1.4, "Solver-variant dispatch   (solver_factory → BemBase.find)",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#4a4a4a", zorder=5)
# rows of chips
rowy = [vy1 - 5.6, vy1 - 8.8, vy1 - 12.0]
ax.text(vx0 + 1.4, rowy[0], "sim", ha="left", va="center", fontsize=8.8, style="italic", color="#555", zorder=5)
ax.text(vx0 + 1.4, rowy[1], "medium", ha="left", va="center", fontsize=8.8, style="italic", color="#555", zorder=5)
ax.text(vx0 + 1.4, rowy[2], "method", ha="left", va="center", fontsize=8.8, style="italic", color="#555", zorder=5)
chip(46, rowy[0], 11, 2.8, "quasistatic", ec=C["ker_ec"])
chip(58.5, rowy[0], 8.5, 2.8, "retarded", ec=C["ker_ec"])
chip(45, rowy[1], 8.5, 2.8, "free", ec=C["geo_ec"])
chip(57, rowy[1], 13, 2.8, "substrate layer", ec=C["geo_ec"])
chip(70, rowy[1], 8, 2.8, "mirror", ec=C["geo_ec"])
chip(47, rowy[2], 12, 2.8, "direct  (LU)", ec=C["sol_ec"])
chip(67, rowy[2], 19, 2.8, "iterative  (GMRES + H-matrix)", ec=C["sol_ec"])
ax.text((vx0 + vx1) / 2, vy0 + 1.5,
        "→  BEMStat · BEMRet · BEMRetLayer · BEMStatIter · BEMRetMirror · BEMStatEig · …",
        ha="center", va="center", fontsize=8.2, color="#444", zorder=5)
# dashed connectors from panel down to the solver (Green eval / assembly / solve)
arrow(40, vy0, 40, 78.1, color=C["var_ec"], lw=1.2, style="-", ls=(0, (4, 3)))
arrow(55, vy0, 55, 78.1, color=C["var_ec"], lw=1.2, style="-", ls=(0, (4, 3)))
arrow(68.5, vy0, 68.5, 76.4, color=C["var_ec"], lw=1.2, style="-", ls=(0, (4, 3)))

# ============================================================ (B) main pipeline row
ymain = 70
# --- BEM solver (Green eval + assembly + solve) live in mnpbem/bem/ --------
solver_panel = FancyBboxPatch((31.5, 61.2), 44.0, 17.4,
                              boxstyle="round,pad=0.2,rounding_size=1.4",
                              fc=C["slv_fc"], ec=C["slv_ec"], lw=1.4,
                              linestyle=(0, (5, 3)), zorder=0.5)
ax.add_patch(solver_panel)
ax.text(32.6, ymain, "BEM solver", rotation=90, ha="center", va="center",
        fontsize=9.0, fontweight="bold", color="#6f7f9c", zorder=1)
# inputs: geometry + materials
box(8, 73.2, 13, 7.4, "geometry",
    "particle · mesh gen\nedge profile · layer", fc=C["geo_fc"], ec=C["geo_ec"], tfs=10, sfs=7.6)
box(8, 65.0, 13, 7.4, "materials",
    "ε(λ): Drude / table\n/ nonlocal / fun", fc=C["mat_fc"], ec=C["mat_ec"], tfs=10, sfs=7.6)
# comparticle
box(22.5, ymain, 12.5, 15.5, "ComParticle (p)",
    "inout ε-index\nconnectivity\nclosed surfaces", fc=C["cp_fc"], ec=C["cp_ec"], tfs=10.5, sfs=8.0)
# --- inside solver: Green eval -> BEM assembly -> solve --------------------
box(40, ymain, 13, 16, "Green functions",
    "G, F,\nH$_{in}$, H$_{out}$\n(greenfun)\nstat/ret · layer",
    fc=C["grn_fc"], ec=C["grn_ec"], tfs=10.5, sfs=8.2)
box(55, ymain, 13, 16, "Assemble matrix",
    "LU factorization",
    fc=C["ker_fc"], ec=C["ker_ec"], tfs=10.5, sfs=8.4)
box(68.5, ymain, 11, 12.5, "Solve",
    "A x = b\n→  σ", fc=C["sol_fc"], ec=C["sol_ec"], tfs=11, sfs=8.6)
# sig
box(81, ymain, 7.5, 10.5, "σ",
    "charge", fc=C["sig_fc"], ec=C["sig_ec"], tfs=13, sfs=8.4)
# observables
box(93, ymain, 12, 17, "Observables",
    "spectra\ndecay rate\nEELS loss\nnear-field",
    fc=C["obs_fc"], ec=C["obs_ec"], tfs=10.8, sfs=8.6)

# pipeline arrows
arrow(14.5, 72.6, 16.0, 71.2)   # geometry -> comparticle
arrow(14.5, 65.6, 16.0, 68.8)   # materials -> comparticle
arrow(28.75, ymain, 33.2, ymain)   # comparticle -> green eval
arrow(46.5, ymain, 48.3, ymain)    # green eval -> assembly
arrow(61.5, ymain, 62.8, ymain)    # assembly -> solve
arrow(74.0, ymain, 77.0, ymain)    # solve -> sig
arrow(84.75, ymain, 86.8, ymain)   # sig -> observables

# ============================================================ (C) excitation (RHS) -> solve
box(68.5, 50.5, 16, 12, "Excitation",
    "plane wave · dipole · EELS\n→  RHS  (φ, A)",
    fc=C["exc_fc"], ec=C["exc_ec"], tfs=10.8, sfs=8.4)
arrow(68.5, 56.5, 68.5, 63.8, color=C["exc_ec"], lw=2.2)
ax.text(69.9, 60.0, "RHS", ha="left", va="center", fontsize=8.2,
        style="italic", color=C["exc_ec"])

# callout: dominant cost
co = FancyBboxPatch((78.5, 44.0), 20.5, 12.5,
                    boxstyle="round,pad=0.2,rounding_size=1.0",
                    fc="#fff7ef", ec=C["fp32"], lw=1.6, zorder=2)
ax.add_patch(co)
ax.text(88.7, 55.0, "dominant cost", ha="center", va="top", fontsize=9.2,
        fontweight="bold", color="#b5651d", zorder=3)
ax.text(88.7, 52.2,
        "Σ assembly + LU ≈ 90% of\nper-λ runtime.\nfp32: ~6× vs fp64,\n~21× vs CPU\n(spectrum err < 1.2×10$^{-3}$,\n no iter. refinement)",
        ha="center", va="top", fontsize=7.8, color="#333", linespacing=1.35, zorder=3)

# ============================================================ (D) compute-backend matrix
ax.text(1.5, 39.0, "Compute backend per stage", ha="left", va="top",
        fontsize=11, fontweight="bold", color="#1b1b1b")

rows = [("CPU  (NumPy/SciPy + numba)", C["cpu"]),
        ("GPU fp64  (CuPy, complex128)", C["fp64"]),
        ("GPU fp32  (CuPy, complex64)", C["fp32"]),
        ("multi-GPU  (cuSolverMg)", C["mgpu"])]
cols = ["Geometry\n& ε(λ)", "ComParticle", "Green\neval",
        "BEM assemble\n(Σ, GEMM, LU)", "Solve\n(σ)", "Observables\n/ field"]
# support matrix: 1 = accelerated, 2 = partial (△), 0 = not on this backend
M = [
    [1, 1, 1, 1, 1, 1],   # CPU
    [0, 0, 2, 1, 1, 1],   # GPU fp64   (Green eval: layer/ACA only -> partial)
    [0, 0, 0, 1, 1, 1],   # GPU fp32
    [0, 0, 0, 1, 1, 0],   # multi-GPU
]

gx0 = 23.0           # left edge of first cell
cw = 12.6            # cell width
ch = 5.2             # cell height
gy_top = 30.5        # y center of first (top) row
col_cx = [gx0 + cw * (i + 0.5) for i in range(6)]
row_cy = [gy_top - ch * 1.12 * j for j in range(4)]

# column headers
for i, c in enumerate(cols):
    ax.text(col_cx[i], gy_top + ch * 0.5 + 1.3, c, ha="center", va="bottom",
            fontsize=7.8, color="#333", linespacing=1.15, fontweight="bold")

# row labels + cells
for j, (rlab, rcol) in enumerate(rows):
    ax.text(gx0 - 1.6, row_cy[j], rlab, ha="right", va="center",
            fontsize=8.4, color="#222")
    for i in range(6):
        state = M[j][i]
        if state == 1:
            cell_fc, cell_ec, cell_lw, mark, mcol = rcol, "#9a9a9a", 1.1, "✓", "#1d1d1d"
        elif state == 2:
            cell_fc, cell_ec, cell_lw, mark, mcol = to_rgba(rcol, 0.40), "#9a9a9a", 1.0, "△", "#555"
        else:
            cell_fc, cell_ec, cell_lw, mark, mcol = C["empty"], "#dcdcdc", 0.8, None, None
        r = Rectangle((col_cx[i] - cw / 2 + 0.4, row_cy[j] - ch / 2 + 0.3),
                      cw - 0.8, ch - 0.6, fc=cell_fc, ec=cell_ec, lw=cell_lw, zorder=2)
        ax.add_patch(r)
        if mark:
            ax.text(col_cx[i], row_cy[j], mark, ha="center", va="center",
                    fontsize=10.5, color=mcol, zorder=3)

# footnote
ax.text(1.5, 8.4,
        "△ Green eval on GPU only for layer (Sommerfeld Bessel/Hankel) & ACA / H-matrix block fill (fp64); "
        "dense free-space fill stays on CPU (numba).   "
        "CPU-resident: mesh & ε eval, refinement quadrature, ACA tree, GMRES residual loop.",
        ha="left", va="center", fontsize=7.0, color="#555", style="italic")

# legend (compute modes selection) - two lines
ax.text(1.5, 4.7,
        "mode select:   CPU = default   |   GPU-fp64 = MNPBEM_GPU=1   |   "
        "GPU-fp32 = + MNPBEM_GPU_LOWPREC=1",
        ha="left", va="center", fontsize=7.7, color="#444", fontweight="bold")
ax.text(1.5, 2.2,
        "multi-GPU = MNPBEM_VRAM_SHARE_GPUS=N        "
        "(pymnpbem config:  compute.gpu_precision = fp32 / fp64)",
        ha="left", va="center", fontsize=7.7, color="#444")

plt.tight_layout(pad=0.6)
_here = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(_here, "fig1_pipeline.pdf")
out_png = os.environ.get("FIG1_PNG", "/tmp/fig1_pipeline.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=160, bbox_inches="tight")
print("saved:", out_pdf, out_png)
