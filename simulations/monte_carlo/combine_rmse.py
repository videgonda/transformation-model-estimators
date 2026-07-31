"""Combined RMSE and Bias comparison figures: Ye-Duan vs Chen vs Engression.

Reads the per-method, per-design results CSVs and generates two figures:
1. RMSE on a common x-axis Lambda_0(y) in [-3, 3]
2. Bias on a common x-axis Lambda_0(y) in [-3, 3]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

DESIGNS = {
    1: r"Identity: $\Lambda_0(y)=y$",
    2: r"Logarithmic: $\Lambda_0(y)=\log(y)$",
    3: r"Hyperbolic sine: $\Lambda_0(y)=\frac{1}{13}\sinh(2y)$",
}

# method -> (csv prefix, label, colour, linestyle, marker)
METHODS = [
    ("ye_duan",    "Ye–Duan",            "tab:blue",   "-",  "o"),
    ("chen",       "Chen",               "tab:red",    "-",  "s"),
    ("engression", "Engression (tuned)", "tab:green",  "-",  "^"),
]

def load(prefix, design):
    path = os.path.join(HERE, f"{prefix}_design_{design}_results.csv")
    return np.genfromtxt(path, delimiter=",", names=True)

# ==========================================
# FIGURE 1: RMSE COMPARISON
# ==========================================
fig_rmse, axes_rmse = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (d, title) in zip(axes_rmse, DESIGNS.items()):
    for prefix, label, colour, ls, marker in METHODS:
        a = load(prefix, d)
        ax.plot(a["lambda_true"], a["rmse"], color=colour, linestyle=ls,
                marker=marker, markersize=3, markevery=5, linewidth=1.5,
                label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\Lambda_0(y)$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(bottom=0)  # Safe for RMSE
    ax.grid(True, alpha=0.3)

axes_rmse[0].set_ylabel("RMSE")
axes_rmse[0].legend(loc="upper center", framealpha=0.9)

fig_rmse.tight_layout()
outpath_rmse = os.path.join(HERE, "rmse_comparison.png")
fig_rmse.savefig(outpath_rmse, dpi=150, bbox_inches="tight")
print(f"saved {outpath_rmse}")

# ==========================================
# FIGURE 2: BIAS COMPARISON
# ==========================================
fig_bias, axes_bias = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (d, title) in zip(axes_bias, DESIGNS.items()):    
    for prefix, label, colour, ls, marker in METHODS:
        a = load(prefix, d)
        ax.plot(a["lambda_true"], a["bias"], color=colour, linestyle=ls,
                marker=marker, markersize=3, markevery=5, linewidth=1.5,
                label=label, zorder=2)
                
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\Lambda_0(y)$")
    ax.set_xlim(-3, 3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, zorder=1, alpha=0.5)
    # 2. DO NOT set bottom=0 here, bias can be negative!
    ax.grid(True, alpha=0.3)

axes_bias[0].set_ylabel("Bias")
# You might need to change 'upper center' to 'best' if the lines overlap the legend
axes_bias[0].legend(loc="best", framealpha=0.9)

fig_bias.tight_layout()
outpath_bias = os.path.join(HERE, "bias_comparison.png")
fig_bias.savefig(outpath_bias, dpi=150, bbox_inches="tight")
print(f"saved {outpath_bias}")