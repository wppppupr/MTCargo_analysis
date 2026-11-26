from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)
fig.panel("panels/panel_pc_histogram.py", x=0, y=0, w=91.5, h=60)
fig.panel("panels/panel_c_moments.py", x=35.6, y=2.2, w=49.6, h=35.1, label=None)
fig.panel("panels/panel_effective_potential.py", x=93, y=0, w=90, h=60)

fig.panel("panels/panel_potential_gvm_06um.py", x=1.7,   y=60,  w=36.6, h=30, texts={"ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_1um.py", x=41.2,   y=60,  w=30, h=30, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_3um.py", x=71.2,   y=60,  w=30, h=30, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_5um.py", x=1.7,   y=89.6,  w=36.6, h=30)
fig.panel("panels/panel_potential_gvm_7um.py", x=41.2,   y=89.6,  w=30, h=30, texts={"ax0.ylabel": ""})
fig.panel("panels/panel_potential_gvm_20um.py", x=71.2,   y=89.6,  w=30, h=30, texts={"ax0.ylabel": ""})
fig.panel("panels/panel_gvm_rmse.py", x=104.1, y=60, w=78.9, h=60)
fig.panel("panels/panel_gvm_k1_k2.py", x=128.1, y=60, w=53.4, h=29.6, label=None)
