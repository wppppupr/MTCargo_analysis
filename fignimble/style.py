"""The look every figure of this project shares: font, text size, line widths and panel labels.
Change it here and every figure follows."""
try:
    from fignimble import *
    _HAS_FIGNIMBLE = True
except ImportError:
    _HAS_FIGNIMBLE = False

TOL_MUTED = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MY_SET = ["#0072B2", "#56B4E9", "#009E73", "#999933", "#E59F00", "#D55E00"]

PALETTES = {
    "tol_muted": TOL_MUTED,
    "okabe-ito": OKABE_ITO,
    "my_set": MY_SET,
}

# Switch palette here: "tol_muted" or "okabe-ito"
PALETTE_NAME = "my_set"
ACTIVE_PALETTE = PALETTES[PALETTE_NAME]

BEAD_STYLE_BY_NAME = {
    "beads06um": (ACTIVE_PALETTE[0], "o"),
    "beads1um":  (ACTIVE_PALETTE[1], "^"),
    "beads3um":  (ACTIVE_PALETTE[2], "s"),
    "beads5um":  (ACTIVE_PALETTE[3], "p"),
    "beads7um":  (ACTIVE_PALETTE[4], "h"),
    "beads20um": (ACTIVE_PALETTE[5], "D"),
}

BEAD_STYLE_BY_DIAM = {
    0.63: (ACTIVE_PALETTE[0], "o"),
    1.18: (ACTIVE_PALETTE[1], "^"),
    3.37: (ACTIVE_PALETTE[2], "s"),
    5.00: (ACTIVE_PALETTE[3], "p"),
    7.24: (ACTIVE_PALETTE[4], "h"),
    20.0: (ACTIVE_PALETTE[5], "D"),
}


def get_bead_style(d_um=None, bead_name=None):
    """Return (color, marker) matching the active palette and bead marker convention."""
    if bead_name is not None and bead_name in BEAD_STYLE_BY_NAME:
        return BEAD_STYLE_BY_NAME[bead_name]
    if d_um is not None:
        d_val = float(d_um)
        closest = min(BEAD_STYLE_BY_DIAM.keys(), key=lambda k: abs(k - d_val))
        return BEAD_STYLE_BY_DIAM[closest]
    return ACTIVE_PALETTE[0], "o"


if _HAS_FIGNIMBLE:
    STYLE = Style(
        font="Arial",
        size=7,
        linewidth=0.6,
        labels=Labels(format="({})", case="lower", size=8, weight="bold"),
        palette=ACTIVE_PALETTE, despine=True,
    )
