from fignimble import *
from style import STYLE

fig = Figure(width=183, height=120, style=STYLE)
fig.panel("panels/panel_lagrangian_flow_autocorr.py", x=0.5, y=0, w=84.5, h=65.4)
fig.panel("panels/panel_cargo_vacf.py", x=94.6, y=0)
fig.panel("panels/panel_relaxation_times.py", x=125.7, y=0, w=48.9, h=30)
fig.panel("panels/panel_msd_kl_06um.py", x=0.5, y=60, h=30, w=30, texts={"ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_1um.py", x=33.6, y=60, h=30, w=30, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_3um.py", x=63.6, y=60, h=30, w=30, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_5um.py", x=0.5, y=90, h=30, w=30)
fig.panel("panels/panel_msd_kl_7um.py", x=33.6, y=90, h=30, w=30, texts={"ax0.ylabel": ""})
fig.panel("panels/panel_msd_kl_20um.py", x=63.6, y=90, h=30, w=30, texts={"ax0.ylabel": ""})