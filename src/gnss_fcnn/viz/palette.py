"""Colour palette matching the original paper (standard MS-Office palette).

Use these constants and ``apply_paper_style()`` so regenerated figures look
native to Xu et al. (2024). Figure 13 is the key visual: a C/N0-vs-elevation
scatter with LOS in blue and NLOS in orange.
"""

from __future__ import annotations

# Core categorical colours (Office default).
BLUE = "#4472C4"   # classification / LOS / primary series
ORANGE = "#ED7D31"  # regression / NLOS / secondary series
GRAY = "#A5A5A5"   # "both" / neutral
YELLOW = "#FFC000"
GREEN = "#70AD47"

# Table styling used throughout the paper.
TABLE_FILL = "#A9D08E"      # sage-green cell fill
TABLE_HEADER = "#375623"    # dark-green header text/fill
TABLE_HEADER_BLACK = "#000000"

# Semantic aliases.
LOS = BLUE
NLOS = ORANGE
CLASSIFICATION = BLUE
REGRESSION = ORANGE
BOTH = GRAY

CATEGORICAL = [BLUE, ORANGE, GRAY, YELLOW, GREEN]


def apply_paper_style() -> None:
    """Set matplotlib rcParams to approximate the paper's figure aesthetic."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
