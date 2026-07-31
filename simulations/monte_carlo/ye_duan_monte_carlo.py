"""
Monte Carlo study for the Ye & Duan (1997) Shifted Quantile Estimator.

Uses the same three designs as Chen (2002) for comparability:
    Design I:   Lambda(y) = y
    Design II:  Lambda(y) = ln(y)
    Design III: Lambda(y) = (1/13)*sinh(2y)

In all designs: X = (X1, X2) iid N(0,1), beta = (1,1), eps ~ N(0,1).

Usage:
    python ye_duan_monte_carlo.py --design 1 --n_samples 1000 --n_reps 50
"""

import argparse
import sys
import os
import time
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from transformation_models.ye_duan_estimator import YeDuanEstimator

# Import shared design definitions from chen_monte_carlo
from chen_monte_carlo import DESIGNS, generate_data


def run_single_replication(
    design_id: int,
    n: int,
    eval_points: np.ndarray,
    rng: np.random.Generator,
    use_true_beta: bool = True,
    n_delta_pts: int = 60,
    delta_n: Optional[int] = None,
) -> tuple:
    spec = DESIGNS[design_id]
    Y, X, Z, beta_true = generate_data(design_id, n, rng)

    if not use_true_beta:
        from transformation_models.beta_estimators import han_mrc
        beta = han_mrc(Y, X, sign_b1=1.0, rng=rng)
    else:
        beta = beta_true

    estimator = YeDuanEstimator(
        y0=spec["y0"],
        estimation_interval=spec["interval"],
        n_delta_pts=n_delta_pts,
        delta_n=delta_n,
    )
    estimator.fit(Y=Y, X=X, beta=beta, eval_points=eval_points)

    return estimator.lambda_hat_, estimator.F_grid_, estimator.F_hat_


def run_monte_carlo(
    design_id: int,
    n_samples: int = 1000,
    n_reps: int = 50,
    n_eval: int = 50,
    n_eval_F: int = 200,
    n_delta_pts: int = 60,
    delta_n: Optional[int] = None,
    seed: int = 42,
    use_true_beta: bool = True,
):
    spec = DESIGNS[design_id]
    y_low, y_high = spec["interval"]
    eval_points = np.linspace(y_low, y_high, n_eval)
    lambda_true = spec["lambda_func"](eval_points)

    rng = np.random.default_rng(seed)
    results = np.zeros((n_reps, n_eval))
    common_grid = np.linspace(-3, 3, n_eval_F)
    F_hat_matrix = np.zeros((n_reps, len(common_grid)))

    print(f"\n{'='*60}")
    print(f"Ye & Duan (1997) Monte Carlo: {spec['name']}")
    print(f"n = {n_samples}, replications = {n_reps}")
    print(f"Evaluation interval: [{y_low}, {y_high}]")
    print(f"y0 = {spec['y0']}, beta_true = {spec['beta']}")
    print(f"Using true beta: {use_true_beta}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for rep in range(n_reps):
        if (rep + 1) % 10 == 0 or rep == 0:
            elapsed = time.time() - t0
            print(f"  Replication {rep+1}/{n_reps}  (elapsed: {elapsed:.1f}s)")
        lambda_hat, F_grid, F_hat = run_single_replication(
            design_id, n_samples, eval_points, rng, use_true_beta, n_delta_pts, delta_n
        )
        results[rep, :] = lambda_hat
        F_hat_matrix[rep, :] = np.interp(common_grid, F_grid, F_hat)
    total_time = time.time() - t0

    # Handle NaN (replace with column mean for statistics)
    for col in range(n_eval):
        mask = np.isnan(results[:, col])
        if mask.any() and not mask.all():
            results[mask, col] = np.nanmean(results[:, col])

    mean_hat = np.nanmean(results, axis=0)
    bias = mean_hat - lambda_true
    sd = np.nanstd(results, axis=0, ddof=1)
    rmse = np.sqrt(bias**2 + sd**2)

    print(f"\nCompleted in {total_time:.1f}s ({total_time/n_reps:.2f}s/rep)")
    print(f"\n{'y':>10s} {'Lambda(y)':>10s} {'Mean':>10s} {'Bias':>10s} {'SD':>10s} {'RMSE':>10s}")
    print("-" * 65)
    for i in range(n_eval):
        print(f"{eval_points[i]:10.3f} {lambda_true[i]:10.3f} {mean_hat[i]:10.3f} "
              f"{bias[i]:10.3f} {sd[i]:10.3f} {rmse[i]:10.3f}")

    # Save results to CSV
    csv_path = os.path.join(os.path.dirname(__file__), f"ye_duan_design_{design_id}_results.csv")
    header = "y,lambda_true,mean_hat,bias,sd,rmse"
    csv_data = np.column_stack([eval_points, lambda_true, mean_hat, bias, sd, rmse])
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"Results saved as ye_duan_design_{design_id}_results.csv")

    # Generate one sample for the rug plot (Y observations spread across the interval)
    Y_rug, _, _, _ = generate_data(design_id, n_samples, np.random.default_rng(seed))
    Y_rug = Y_rug[(Y_rug >= y_low) & (Y_rug <= y_high)]

    from scipy.stats import norm
    F_mean = np.mean(F_hat_matrix, axis=0)
    F_sd   = np.std(F_hat_matrix, axis=0, ddof=1)
    F_true = norm.cdf(common_grid)

    # Plot F separately
    fig_f, ax_f = plt.subplots(figsize=(7, 5))
    ax_f.plot(common_grid, F_true, "k-", linewidth=2, label=r"$F_\varepsilon = \Phi$")
    ax_f.plot(common_grid, F_mean, "b--", linewidth=2, label=r"$\hat{F}_\varepsilon$ (mean)")
    ax_f.fill_between(common_grid, F_mean - 2*F_sd, F_mean + 2*F_sd,
                      alpha=0.2, color="blue", label=r"$\pm 2$ SD")
    ax_f.set_xlabel(r"$x$")
    ax_f.set_ylabel(r"$F_\varepsilon(x)$")
    ax_f.set_title("Estimated error CDF")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3)
    fig_f.tight_layout()
    outpath_f = os.path.join(os.path.dirname(__file__), f"ye_duan_F_design_{design_id}.png")
    fig_f.savefig(outpath_f, dpi=150)
    print(f"Plot saved as ye_duan_F_design_{design_id}.png")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    true_eval = np.linspace(y_low, y_high, 100)
    true_lambda_dense = spec["lambda_func"](true_eval)

    ax = axes[0]
    ax.plot(true_eval, true_lambda_dense, "k-", linewidth=2, label=r"$\Lambda_0(y)$")
    ax.plot(eval_points, mean_hat, "b--", linewidth=2, label=r"$\hat{\Lambda}_0(y)$ (mean)")
    ax.fill_between(eval_points, mean_hat - 2*sd, mean_hat + 2*sd, alpha=0.2, color="blue")
    # Rug plot: Y observations along the x-axis showing data coverage
    from matplotlib.transforms import blended_transform_factory
    rug_transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(Y_rug, np.zeros(len(Y_rug)), "|", transform=rug_transform,
            color="gray", alpha=0.15, markersize=8, clip_on=False, label="observed $Y$")
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$\Lambda_0(y)$")
    ax.set_title(spec["plot_title"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(eval_points, bias, "b-o", label="Bias", markersize=3)
    ax.plot(eval_points, rmse, "r-s", label="RMSE", markersize=3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("$y$")
    ax.set_title("Bias and RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), f"ye_duan_design_{design_id}.png")
    plt.savefig(outpath, dpi=150)
    plt.show()
    print(f"\nPlot saved as ye_duan_design_{design_id}.png")

    return eval_points, lambda_true, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ye & Duan (1997) Monte Carlo study")
    parser.add_argument("--design", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--n_eval_F", type=int, default=200)
    parser.add_argument("--n_delta_pts", type=int, default=60)
    parser.add_argument("--delta_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate_beta", action="store_true")
    args = parser.parse_args()

    run_monte_carlo(
        design_id=args.design,
        n_samples=args.n_samples,
        n_reps=args.n_reps,
        n_eval=args.n_eval,
        n_eval_F=args.n_eval_F,
        n_delta_pts=args.n_delta_pts,
        delta_n=args.delta_n,
        seed=args.seed,
        use_true_beta=not args.estimate_beta,
    )
