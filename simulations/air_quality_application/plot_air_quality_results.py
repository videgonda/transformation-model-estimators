"""
Plotter for air quality transformation-model results.

Reads the CSV files created by air_quality_application.py and saves plots:
- lambda_estimates.png
- F_hat_ye_duan.png
- scatter_data.png
- lambda_vs_z_chen.png
- lambda_vs_z_ye_duan.png

Usage:
    python plot_air_quality_results.py
    python plot_air_quality_results.py --results-dir ../results --output-dir ../results --show
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_result_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float)


def plot_lambda_estimates(lambda_path: str, output_path: str):
    data = load_result_csv(lambda_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["y"], data["chen"],    label="Chen estimator",      linewidth=2, color="tab:red")
    ax.plot(data["y"], data["ye_duan"], label="Ye & Duan estimator", linewidth=2)
    ax.plot(data["y"], data["engression"], label="Engression",
            linewidth=2, color="tab:green")
    ax.set_xlabel("$y$  (NOx concentration in ppb)")
    ax.set_ylabel(r"$\hat{\Lambda}(y)$")
    ax.set_title("Estimated transformation function")
    ax.legend()
    ax.grid(True, alpha=0.4)
    # Variable info box
    info = ("Y:   PT08.S3 (NOx sensor)\n"
            "X_1: PT08.S2 (NMHC sensor)\n"
            "X_2: PT08.S5 (O3 sensor)")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_f_hat(f_path: str, output_path: str):
    data = load_result_csv(f_path)
    plt.figure(figsize=(8, 6))
    plt.plot(data["x"], data["F_hat"], label=r"$\,\hat{F}(x)$", color="tab:blue")
    plt.xlabel("x")
    plt.ylabel(r"$\hat{F}(x)$")
    plt.title("Estimated error CDF from Ye & Duan for Air quality data")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scatter(scatter_path: str, output_path: str):
    data = load_result_csv(scatter_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(data["Y"], data["Z"], s=10, alpha=0.5, color="gray")
    plt.xlabel("Y")
    plt.ylabel(r"$\widehat{Z}$")
    plt.title("Scatter sample of Y vs $\widehat{Z}$")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lambda_vs_z(
    lambda_path: str,
    scatter_path: str,
    output_path: str,
    estimator: str = "chen",
    label: str = "Chen (2002)",
    color: str = "tab:purple",
):
    lambda_data = load_result_csv(lambda_path)
    scatter_data = load_result_csv(scatter_path)

    y_vals = scatter_data["Y"]
    z_vals = scatter_data["Z"]
    lambda_y = np.interp(y_vals, lambda_data["y"], lambda_data[estimator],
                         left=np.nan, right=np.nan)
    valid = np.isfinite(lambda_y) & np.isfinite(z_vals)

    if not np.any(valid):
        raise ValueError("No valid points for Lambda(Y) vs Z diagnostic plot.")

    corr = np.corrcoef(lambda_y[valid], z_vals[valid])[0, 1]
    residuals = lambda_y[valid] - z_vals[valid]
    rmse = np.sqrt(np.mean(residuals ** 2))

    x_min = np.nanmin(z_vals[valid])
    x_max = np.nanmax(z_vals[valid])
    pad = 0.15 * (x_max - x_min)
    x_line = [x_min - pad, x_max + pad]
    plt.figure(figsize=(8, 6))
    plt.scatter(z_vals[valid], lambda_y[valid], s=10, alpha=0.5, color=color)
    plt.plot(x_line, x_line,
             color="black", linestyle="--", linewidth=1)
    plt.xlabel(r"$\widehat{Z}$")
    plt.ylabel(r"$\widehat{\Lambda}_0(Y)$")
    plt.title(fr"{label} $\widehat{{Z}}$ versus $\widehat{{\Lambda}}_0(Y)$")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return corr, rmse, np.sum(valid)


def main():
    parser = argparse.ArgumentParser(description="Plot air quality model result CSVs")
    parser.add_argument(
        "--results-dir",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results")),
        help="Directory containing the result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plot files (defaults to results dir)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    lambda_path = os.path.join(args.results_dir, "lambda_estimates.csv")
    f_path = os.path.join(args.results_dir, "F_hat_ye_duan.csv")
    scatter_path = os.path.join(args.results_dir, "scatter_data.csv")

    print(f"Reading: {lambda_path}")
    print(f"Reading: {f_path}")
    print(f"Reading: {scatter_path}")

    lambda_png       = os.path.join(output_dir, "lambda_estimates.png")
    f_png            = os.path.join(output_dir, "F_hat_ye_duan.png")
    scatter_png      = os.path.join(output_dir, "scatter_data.png")
    diag_chen_png    = os.path.join(output_dir, "lambda_vs_z_chen.png")
    diag_yd_png      = os.path.join(output_dir, "lambda_vs_z_ye_duan.png")
    diag_eng_png     = os.path.join(output_dir, "lambda_vs_z_engression.png")

    plot_lambda_estimates(lambda_path, lambda_png)
    print(f"Saved: {lambda_png}")

    plot_f_hat(f_path, f_png)
    print(f"Saved: {f_png}")

    plot_scatter(scatter_path, scatter_png)
    print(f"Saved: {scatter_png}")

    corr, rmse, n_points = plot_lambda_vs_z(
        lambda_path, scatter_path, diag_chen_png,
        estimator="chen", label="Chen", color="tab:red",
    )
    print(f"Saved: {diag_chen_png}")
    print(f"  Chen:    n={n_points}, corr={corr:.3f}, rmse={rmse:.3f}")

    corr, rmse, n_points = plot_lambda_vs_z(
        lambda_path, scatter_path, diag_yd_png,
        estimator="ye_duan", label="Ye & Duan ", color="tab:blue",
    )
    print(f"Saved: {diag_yd_png}")
    print(f"  Ye & Duan: n={n_points}, corr={corr:.3f}, rmse={rmse:.3f}")

    corr, rmse, n_points = plot_lambda_vs_z(
        lambda_path, scatter_path, diag_eng_png,
        estimator="engression", label="Engression", color="tab:green",
    )
    print(f"Saved: {diag_eng_png}")
    print(f"  Engression: n={n_points}, corr={corr:.3f}, rmse={rmse:.3f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
