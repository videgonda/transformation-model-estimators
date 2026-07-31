"""Combined Lambda_hat(Y) vs Z-hat diagnostic for the three estimators.

Reads lambda_estimates.csv and scatter_data.csv (produced by
air_quality_application.py) and lays the per-estimator diagnostics out as a
single 1x3 panel with a shared slope-one diagonal and the correlation
annotated in each panel title.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

lam = np.genfromtxt(os.path.join(RESULTS, "lambda_estimates.csv"),
                    delimiter=",", names=True)
sca = np.genfromtxt(os.path.join(RESULTS, "scatter_data.csv"),
                    delimiter=",", names=True)

y_vals = sca["Y"]
z_vals = sca["Z"]

METHODS = [
    ("ye_duan",    "Ye–Duan", "tab:blue"),
    ("chen",       "Chen",       "tab:red"),
    ("engression", "Engression", "tab:green"),
]

# common square window so the slope-one diagonal renders at 45 degrees,
# cropping the sparse low-Z outliers to avoid excess whitespace
lim_lo, lim_hi = -2100, 1500

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, (key, label, color) in zip(axes, METHODS):
    lam_y = np.interp(y_vals, lam["y"], lam[key], left=np.nan, right=np.nan)
    valid = np.isfinite(lam_y) & np.isfinite(z_vals)
    zc, lc = z_vals[valid], lam_y[valid]
    corr = np.corrcoef(lc, zc)[0, 1]

    ax.scatter(zc, lc, s=8, alpha=0.4, color=color)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1)
    ax.set_title(f"{label}  (corr $= {corr:.3f}$)", fontsize=12)
    ax.set_xlabel(r"$\widehat{Z}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r"$\widehat{\Lambda}_0(Y)$")

for ax in axes:
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
out = os.path.join(RESULTS, "lambda_vs_z_combined.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
