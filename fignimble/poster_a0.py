from fignimble import *
from style import ACTIVE_PALETTE

POSTER_STYLE = Style(
    font="Arial",
    size=18,
    linewidth=1.0,
    labels=Labels(format="({})", case="lower", size=12, weight="bold"),
    palette=ACTIVE_PALETTE,
    despine=True,
)

# A0 Portrait: 841 mm x 1189 mm
# Top y = 0..300 mm intentionally left blank for PowerPoint abstract / intro
fig = Figure(width=841, height=1189, style=POSTER_STYLE)

fig.image("images/poster_MolRobEX.pdf", x=0, y=0, w=841)

# ------------------------------------------------------------------
# Card 1: Macroscopic Cargo Transport
# ------------------------------------------------------------------
fig.rect(15, 599.1, 399, 244, stroke="#cbd5e1", fill="#f8fafc", width=0.6, radius=3)
fig.rect(15, 599.1, 399, 13, stroke=None, fill="#0072B2", radius=3)
fig.text("1. Macroscpic Cargo Dynamics", x=22, y=602.3, size=28, weight="bold", color="#ffffff")

fig.panel("panels/panel_msd.py", x=21,  y=616.1, w=208, h=131.3, label=None)
fig.panel("panels/panel_pdf.py", x=21.5, y=750.9, w=207, h=86.2, label=None)
fig.panel("panels/panel_ngp.py", x=231.5, y=616.1, w=175, h=108, label=None)

# ------------------------------------------------------------------
# Card 2: Collective MT Dynamics & Spatial Correlation
# ------------------------------------------------------------------
fig.rect(424, 599.1, 402, 244, stroke="#cbd5e1", fill="#f8fafc", width=0.6, radius=3)
fig.rect(424, 599.1, 402, 13, stroke=None, fill="#0072B2", radius=3)
fig.text("2. Collective MT Dynamics & Spatial Correlation", x=431, y=602.3, size=32, weight="bold", color="#ffffff")

fig.panel("panels/panel_bg_angular_correlation_cr.py", x=433, y=616.1, w=192, h=123, label=None)
fig.panel("panels/panel_xi_vs_radius.py",              x=534.5, y=620.4, w=86,  h=85.7, label=None)
fig.panel("panels/panel_local_polar.py",               x=632, y=620.4, w=192, h=127, label=None)
fig.panel("panels/panel_local_polar_variance.py",      x=433, y=739.1, w=192, h=104, label=None)
fig.panel("panels/panel_global_order.py",              x=728, y=616.1, w=95, h=61.2, label=None, texts={"ax0.ylabel": "Polar Order $\\Phi_{\\mathrm{gl}}$", "ax1.ylabel": "Nematic Order $S_{\\mathrm{gl}}$"})

# ------------------------------------------------------------------
# Card 3: Quasi-Potential & GvM Alignment Model
# ------------------------------------------------------------------
fig.rect(15, 853.1, 310, 276, stroke="#cbd5e1", fill="#f8fafc", width=0.6, radius=53)
fig.rect(15, 853.1, 310, 13, stroke=None, fill="#0072B2", radius=3)
fig.text("3. Quasi-Potential & GvM Alignment Model", x=22, y=856.3, size=32, weight="bold", color="#ffffff")

fig.rect(21, 869.1, 298, 17, stroke="#93c5fd", fill="#eff6ff", width=0.5, radius=2)
fig.text(r"$\tilde{U}_{\mathrm{eff}}(c) = -\ln\frac{P(c)}{P_{\mathrm{iso}}(c)} \approx -\kappa_1 c - \kappa_2(2c^2 - 1) + C$",
         x=170, y=873.6, size=10.5, align="center", color="#0f172a")

fig.panel("panels/panel_pc_histogram.py",        x=21,  y=889.1, w=146, h=104, label=None)
fig.panel("panels/panel_c_moments.py",           x=211.41,  y=886.1, w=77,  h=50, label=None)
fig.panel("panels/panel_effective_potential.py", x=172, y=889.1, w=147, h=104, label=None)

fig.panel("panels/panel_potential_gvm_06um.py", x=21,  y=997.1, w=62, h=62, texts={"ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_1um.py",  x=83,  y=997.1, w=56, h=62, label=None, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_3um.py",  x=139, y=997.1, w=56, h=62, label=None, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_potential_gvm_5um.py",  x=21,  y=1059.1, w=62, h=64, label=None)
fig.panel("panels/panel_potential_gvm_7um.py",  x=83,  y=1059.1, w=56, h=64, label=None, texts={"ax0.ylabel": ""})
fig.panel("panels/panel_potential_gvm_20um.py", x=139, y=1059.1, w=56, h=64, label=None, texts={"ax0.ylabel": ""})

fig.panel("panels/panel_gvm_rmse.py",  x=199, y=997.1, w=120, h=126)
fig.panel("panels/panel_gvm_k1_k2.py", x=244, y=1003.1, w=71,  h=52, label=None)

# ------------------------------------------------------------------
# Card 4: Information Bottleneck (DPI)
# ------------------------------------------------------------------
fig.rect(335, 853.1, 180, 276, stroke="#cbd5e1", fill="#f8fafc", width=0.6, radius=3)
fig.rect(335, 853.1, 180, 13, stroke=None, fill="#0072B2", radius=3)
fig.text("4. Information Bottleneck (DPI)", x=342, y=856.3, size=32, weight="bold", color="#ffffff")

fig.rect(341, 869.1, 168, 17, stroke="#93c5fd", fill="#eff6ff", width=0.5, radius=2)
fig.text(r"$D_{\mathrm{KL}} \leq D_{\mathrm{KL}}^{\mathrm{conv}} \leq D_{\mathrm{MT}},\ \eta = \eta_{\mathrm{geom}}\eta_{\mathrm{mech}}$",
         x=425, y=874.1, size=9.5, align="center", color="#0f172a")

fig.panel("panels/panel_kl_divergence.py", x=341, y=889.1,  w=168, h=76)
fig.panel("panels/panel_dpi.py",           x=341, y=968.1, w=168, h=76)
fig.panel("panels/panel_eta.py",           x=341, y=1047.1, w=168, h=76)

# ------------------------------------------------------------------
# Card 5: Lagrangian Memory & Zero-Param MSD Model
# ------------------------------------------------------------------
fig.rect(525, 853.1, 301, 276, stroke="#cbd5e1", fill="#f8fafc", width=0.6, radius=3)
fig.rect(525, 853.1, 301, 13, stroke=None, fill="#0072B2", radius=3)
fig.text("5. Lagrangian Memory & Zero-Param MSD Model", x=532, y=856.3, size=32, weight="bold", color="#ffffff")

fig.rect(531, 869.1, 289, 17, stroke="#93c5fd", fill="#eff6ff", width=0.5, radius=2)
fig.text(r"$\mathrm{MSD}(M\Delta t) = \sigma_1^2 [ M + 2\sqrt{D_{\mathrm{KL}}}\sum_{k=1}^{M-1}(M-k)e^{-k\Delta t/\tau_{\mathbf{v}}^{\mathrm{slow}}} ]$",
         x=675.5, y=873.6, size=10, align="center", color="#0f172a")

fig.panel("panels/panel_lagrangian_flow_autocorr.py", x=531, y=889.1, w=142, h=104)
fig.panel("panels/panel_cargo_vacf.py",               x=678, y=889.1, w=142, h=104)
fig.panel("panels/panel_relaxation_times.py",         x=742, y=894.1, w=74,  h=50, label=None)

fig.panel("panels/panel_msd_kl_06um.py", x=531, y=997.1, w=101, h=62, texts={"ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_1um.py",  x=632, y=997.1, w=94,  h=62, label=None, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_3um.py",  x=726, y=997.1, w=94,  h=62, label=None, texts={"ax0.ylabel": "", "ax0.xlabel": ""})
fig.panel("panels/panel_msd_kl_5um.py",  x=531, y=1059.1, w=101, h=64, label=None)
fig.panel("panels/panel_msd_kl_7um.py",  x=632, y=1059.1, w=94,  h=64, label=None, texts={"ax0.ylabel": ""})
fig.panel("panels/panel_msd_kl_20um.py", x=726, y=1059.1, w=94,  h=64, label=None, texts={"ax0.ylabel": ""})

