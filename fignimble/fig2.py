from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)
fig.panel("panels/panel_msd.py", x=0.5, y=0, w=84.5, h=65.4)
fig.panel("panels/panel_pdf.py", x=91.5, y=2.8, w=44.2, h=50)
fig.panel("panels/panel_ngp.py", x=137.9, y=2.8, w=40.8, h=50)
fig.panel("panels/panel_global_order.py", x=2.75, y=59)
fig.panel("panels/panel_bg_angular_correlation_cr.py", x=103, y=57.7)
fig.panel("panels/panel_xi_vs_radius.py", x=139.6, y=59, label=None, w=43.4, h=34.9)