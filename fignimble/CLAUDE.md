<!-- written by Fignimble; removed when Fignimble closes the project -->
# Editing figures in this project with Fignimble

This folder is a Fignimble project: publication figures with several panels, each figure written in Python. The user
sees the figures in Fignimble, a figure editor, while you edit the code here, and edits them by hand there too (moving,
resizing, retyping). Fignimble re-renders a figure about a second after you save a file. (This file was written by
Fignimble and is not part of the project; don't edit it.)

## How a project is laid out

- **`fignimble.toml`** marks the project's top folder and records the Fignimble version that made it. Leave it alone.
- **Figure files** are the top-level `.py` files that make a `Figure` (`fig1.py`, `fig2.py`, ...). Each places
  elements on its figure, one call per element. Units are **millimeters**, from the figure's **top-left corner, y
  pointing down**.
- **Panels** are drawn by Python files, usually in `panels/`. A panel file has a function `draw(fig, ...)` that gets
  a matplotlib `Figure` of exactly the panel's size (`fig.add_subplot()`, `fig.subplots(2, 2)`, `fig.colorbar(...)`
  all work). `fig.panel("panels/x.py", function="other", args={"gene": "CD3E"})` calls `other(fig, gene="CD3E")`,
  so one file can draw several panels.
- **`style.py`** holds the `Style` every figure imports: font, text size, line widths, and the `Labels` (a, b, c...).

```python
from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)
fig.panel("panels/expression.py", x=2, y=2, w=88, h=55)           # labeled a automatically
fig.image("images/gel.png", x=93, y=2, w=88)                       # pictures are panels too (b); h follows aspect
fig.text("Treated", x=40, y=60, size=7, weight="bold", align="center")
fig.arrow(60, 70, 80, 70)                       # also fig.line, fig.rect, fig.ellipse, fig.bracket(x1, x2, y, "**")
```

Panels can also be laid out **patchwork style**: the layout gives each panel its place, lines up their plot areas
across rows and columns, and can collect the legends they share into one:

```python
a, b, c, d = Panel("panels/a.py"), Panel("panels/b.py"), Panel("panels/c.py"), Image("images/gel.png")
fig.layout((a | b | c) / (d | Spacer()), x=0, y=0, w=183, h=120, heights=[2, 1], gap=4)     # | beside, / below
fig.layout(Grid(a, b, c, d, ncol=2), x=0, y=0, w=183, h=120)                               # + wraps into a grid
fig.layout(Design("""
    AAB
    CDB
""", A=a, B=b, C=c, D=d), x=0, y=0, w=183, h=120, widths=[1, 1, 1.5], guides="collect", legend="bottom")
fig.inset(Panel("panels/zoom.py"), within=a, x=0.6, y=0.05, w=0.35, h=0.35)   # fractions of a's box
fig.legend({"WT": "#999999", "KO": "#d1495b"}, x=150, y=5)                   # one legend shared by panels
```

`Row(..., widths=[2, 1])`, `Column(..., heights=[...])`, `Grid(..., ncol=, nrow=)` give a part of a layout its own
sizes. Panels on the figure are lined up the same way whenever they share an edge (`Style(align_axes=False)` turns
this off). Prefer a layout for regular grids of plots, and free `fig.panel(...)` for figures arranged by hand.

Every figure also has a **dark version** for talks (`fignimble snapshot fig1.py --dark`): fignimble turns the style's
text and axes white and the background black. Colors panel code sets itself stay as they are, so in panel code
use `"white" if is_dark() else "black"` (from fignimble) wherever you would write black.

Annotations: `fig.text("**bold** *italic* 10^{3} CD8_{a}", width=60)` (markup; `width` wraps), `fig.arrow(...,
curve=0.3, tip="stealth")`, `fig.path(points, smooth=True, closed=True, fill=...)`, `fig.scalebar(picture, length=50,
unit="µm")` (the pixel size from the picture's metadata, or `pixel_size=0.325` from the microscope), `fig.zoom(picture, inset, region=(x, y, w, h))` (fractions; the inset of the same
picture is cropped to the region). Pictures take `rotation=` and `flip="h"`. `Style(palette={"WT": "#999999", ...})`
sets the colors panels plot with (or a named one: `palette="okabe-ito"`, color-blind safe); use `color("WT")` in
panel code rather than repeating hex codes, and keep color-blind readers in mind (`fignimble info` warns about colors
they can't tell apart). A panel's `texts={"ax0.xlabel": "..."}`, `text_style={"ax0.legend0": {"size": 8, "color": "#...", "weight": "bold"}}`
and `colors={"ax0.line1": "#..."}` hold texts, text settings and colors the user changed on the canvas: leave them alone, or move them into the panel's code and remove the entries. `locked=True` / `hidden=True` are the user's layer settings: keep them.

Panels may import any package. List those beyond matplotlib and numpy in the project's `requirements.txt` (pip's
format), and Fignimble offers to install them when the project opens. Slow work several panels share (loading and
cleaning data, fitting a model) goes in a function decorated `@cached("data/table.csv")` (from fignimble): it runs once
and is kept until its file, its arguments or the listed data change. A rectangle can carry a panel letter
(`fig.rect(..., label="auto")`), for schematics drawn from shapes and pictures.

**Tables of the paper** (Table 1, Supplementary Table 2) are files of their own, not figures: `table1.py` makes one
`Table`, which lays out its title, cells and notes and sizes its page (no positions, no panel letters):

```python
from fignimble import *
from style import STYLE

tab = Table("tables/genes.csv", style=STYLE, title="Differentially expressed genes",
            decimals=[None, 0, 2, 3], notes=["FDR, Benjamini-Hochberg.", "n, cells per gene."])
```

The cells come from a CSV/TSV in `tables/`, `rows=[[...], ...]`, or a function returning rows or a pandas DataFrame
(`Table(genes, ...)`, with `@cached` for slow work), so a table computed from data stays tied to it: prefer that to
pasting numbers. Headers carry units; `decimals=` rounds each column alike. Journals want tables as text:
`fignimble export table1.py table1.docx` writes a Word table with the title and notes (also .xlsx, .csv, .tsv). To
show a table in a figure or poster, place it whole: `fig.figure("table1.py", ...)`. Only a small table that is part of
a figure goes in the figure itself: `fig.table("tables/counts.csv", x=4, y=80)`.

**Figures inside figures, and posters.** `fig.figure("fig2.py", x=20, y=130, w=360)` places another figure of the
project whole, drawn from its code (scaled to `w`, text and lines with it; `letters=False` drops its panel letters,
`style=` redraws it with another Style, `label="auto"` letters it as one panel). Never copy a figure's panels into
another figure to show it again: place the figure. A poster is a figure: `Figure(poster="a0")` (also "48x36", "72x48",
"24x36", "a1", ...: `fignimble journals` lists them; `landscape=True`) sets its size and `Style.poster`, whose type sizes are for reading from
a meter or two; preflight checks text at the size it is printed, including inside placed figures, and keeps a margin
along the edges clear. Text boxes (`width=`) lay out lines starting with "- " as bullets, "1. " as numbered items and
"# " as headings; `fig.text(source="intro.md", width=..., height=...)` takes the text from a Markdown file (keep a
poster's prose there), `height=` reports overflow and `fit="shrink"` fits it. `fig.qr("https://...", x=, y=, w=40)`
is a QR code. Export a poster with `--bleed 3 --crop-marks` when a print shop asks for them.

Other options: `label=None` (no letter) or `label="a'"` on a panel; `Labels(numbering="none")` for no letters at all; `fit=`/`crop=` on images; `group="g1"` to keep
elements together; `fig.guide(x=91.5)` for an editor guide line. Run `python -c "import fignimble; help(fignimble.Figure)"`
for every argument.

## Look at what you made

- `fignimble snapshot fig1.py` renders the figure to a PNG and prints its path. Open that picture to check your work.
- `fignimble info fig1.py` lists every element with its box (mm), the panel labels, and the preflight warnings (text
  below the journal's minimum size, low-resolution pictures, elements outside the figure).
- `fignimble export fig1.py fig1.pdf` writes the figure (PDF, EPS, SVG, TIFF, PNG, JPEG, PPTX; `--dark` for its dark
  version, `--journal cell` for that journal's resolutions; TIFF/JPEG are raised to them and every export is measured).
  Don't run other renderers. Each export records what it was made from; `fignimble provenance fig1.pdf` checks it
  against the project.
- `fignimble source-data fig1.py` writes the values plotted in each panel to `fig1_source_data.xlsx`, one sheet per
  panel, as journals ask. A panel whose plotted values can't be read back (shaded areas, violins, huge heatmaps)
  supplies them with a `source_data(...)` function next to `draw` (`<name>_source_data` for a panel function
  `<name>`), taking the same arguments and returning a DataFrame, a dict of them, or rows.

## Keep the figures editable by hand

Fignimble moves and resizes an element by rewriting the numbers in the call that made it, so:

- Give `x`, `y`, `w`, `h` (and line ends) as **literal numbers**. A computed position (`x=a.right + 2`) can't be
  dragged; use one only when asked to tie elements together.
- **Never draw panel letters yourself** (no `fig.text("a", ...)`, no `ax.set_title("a")`). Labels are automatic, in
  reading order, styled by `Labels` in `style.py`; to change their look, change that one `Labels(...)`.
- In panel code, **don't set font sizes, font families or line widths, and don't call `plt.figure`, `figsize`,
  `savefig`, `tight_layout` or `plt.show`**. The style sets fonts and sizes so all panels match, and Fignimble lays out and
  saves each panel at its size. (Use `fig` rather than `plt` where you can.)
- Don't reformat or reorder code you weren't asked to change: the order of the calls is the stacking order, and the
  user's own edits are in those lines.

## Working with the user

- Make one change at a time and say which figure and panel it touched.
- If a panel fails, Fignimble shows it red with the error, and `fignimble info` prints it. Fix the code.
- The user's Undo in Fignimble undoes only Fignimble's own edits, not yours.
