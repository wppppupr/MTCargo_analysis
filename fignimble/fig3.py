from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)
fig.panel("panels/panel_global_order.py")
fig.panel("panels/panel_bg_angular_correlation_cr.py", x=103)
fig.panel("panels/panel_xi_vs_radius.py", x=139.6, y=1.3, label=None, w=43.4, h=34.9)
fig.panel("panels/panel_local_polar.py", x=0, y=60)
fig.panel("panels/panel_local_polar_variance.py", x=103, y=60)

