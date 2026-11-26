from fignimble import *
from style import STYLE

fig = Figure(width=85, height=120, style=STYLE)
fig.panel("panels/panel_msd.py", x=0.5, y=0, w=84.5, h=65.4)
fig.panel("panels/panel_pdf.py", x=0.25, y=68.4, w=44.2, h=50)
fig.panel("panels/panel_ngp.py", x=43.95, y=68.4, w=40.8, h=50)