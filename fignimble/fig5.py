from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)

fig.panel("panels/panel_kl_divergence.py", x=7.5, y=0)
fig.panel("panels/panel_dpi.py", x=100.4, y=0)
fig.panel("panels/panel_eta.py", x=7.5, y=60)