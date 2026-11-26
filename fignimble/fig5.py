from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)

fig.panel("panels/panel_potential_gvm_06um.py", x=0,   y=0,  w=59, h=59)
fig.panel("panels/panel_potential_gvm_1um.py",  x=62,  y=0,  w=59, h=59)
fig.panel("panels/panel_potential_gvm_3um.py",  x=124, y=0,  w=59, h=59)
fig.panel("panels/panel_potential_gvm_5um.py",  x=0,   y=61, w=59, h=59)
fig.panel("panels/panel_potential_gvm_7um.py",  x=62,  y=61, w=59, h=59)
fig.panel("panels/panel_potential_gvm_20um.py", x=124, y=61, w=59, h=59)
