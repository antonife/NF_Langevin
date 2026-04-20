"""
Publication-quality matplotlib style for NF_Langevin figures.
Copied from Landau_Theory/scripts/pub_style.py with adjusted figure directory.

Usage:
    from pub_style import apply_style, save_pub, COLORS, panel_label
    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save_pub(fig, 'my_figure')
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Colour palette — harmonised, colour-blind-safe, print-friendly
# ---------------------------------------------------------------------------
COLORS = {
    # Primary physical quantities
    "order_param": "#2166AC",  # rich blue — xi
    "field": "#4DAF4A",  # green — h(t)
    "potential": "#984EA3",  # purple — G, A
    "entropy": "#E41A1C",  # red — S, EPR
    "curvature": "#FF7F00",  # orange — kappa
    "work": "#A65628",  # brown — W
    # Branch styling
    "stable_branch": "#222222",  # near-black
    "unstable_branch": "#999999",  # grey
    "metastable_fill": "#C6DBEF",  # light blue fill
    "jump_arrow": "#E41A1C",  # red
    # Temperature gradient (cold -> hot)
    "t_cold": "#08306B",  # dark navy
    "t_warm": "#FD8D3C",  # orange
    # EPR decomposition
    "epr_total": "#222222",  # black
    "epr_cg": "#E41A1C",  # red
    "epr_nf": "#2166AC",  # blue — NF EPR (new)
    "epr_hidden_fill": "#DEEBF7",  # very light blue
    "epr_cg_fill": "#FEE0D2",  # very light red
    "epr_nf_fill": "#C6DBEF",  # light blue fill for NF
    # NF-specific
    "flow_density": "#2166AC",  # blue
    "boltzmann": "#222222",  # black
    "residual": "#E41A1C",  # red
    "neural_ode": "#FF7F00",  # orange
    # Misc
    "bound_line": "#E41A1C",  # red dashed bound
    "langevin_cycle": "#2166AC",  # blue
    "fit_line": "#E41A1C",  # red
    "data_points": "#222222",  # black squares
}


# ---------------------------------------------------------------------------
# Apply unified rcParams
# ---------------------------------------------------------------------------
def apply_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update(
        {
            # Font — serif family matching LaTeX Computer Modern
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
                "Times New Roman",
            ],
            "mathtext.fontset": "cm",
            "text.usetex": False,  # safe fallback
            # Font sizes
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            # Axes
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": True,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# ---------------------------------------------------------------------------
# Save helper — dual output PDF + PNG
# ---------------------------------------------------------------------------
_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def save_pub(fig, name, fig_dir=None):
    """Save figure as both PDF (vector) and PNG (raster) at publication DPI."""
    d = fig_dir or _FIG_DIR
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(d, f"{name}.{ext}")
        fig.savefig(path, format=ext)
        print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Panel label helper
# ---------------------------------------------------------------------------
def panel_label(ax, label, x=-0.12, y=1.06):
    """Place a bold panel label like (a), (b) at top-left of axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    apply_style()
    print("pub_style loaded successfully.")
    print(f"Figure directory: {os.path.abspath(_FIG_DIR)}")
