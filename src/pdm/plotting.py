"""Shared chart style: one palette and one set of mark rules for every figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pdm.config import FIGURES_DIR, ensure_dirs

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

# Colour follows the entity: a model keeps its colour in every figure.
FAMILY_COLORS = {"logreg": BLUE, "xgboost": ORANGE, "lightgbm": AQUA}
HEALTHY_COLOR = BLUE
FAILED_COLOR = ORANGE

SEQUENTIAL = LinearSegmentedColormap.from_list("pdm_blue", [SURFACE, "#cde2fb", BLUE, "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("pdm_div", [BLUE, "#f0efec", "#e34948"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def save(fig, name: str) -> Path:
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
