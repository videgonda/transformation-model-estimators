"""
Monte Carlo study replicating the simulation designs in Chen (2002), Section 4.

 (Table I & II):
    Design I:   Lambda_0(y) = y,          Z ~ N(0,1), eps ~ N(0,1)
    Design II:  Lambda_0(y) = ln(y),      Z ~ N(0,1), eps ~ N(0,1)
    Design III: Lambda_0(y) = (1/13)*sinh(2y), Z ~ N(0,1), eps ~ N(0,1)

In all designs, X = (X1, X2) with X1, X2 iid N(0, 1/2), beta = (1, 1),
so Z = X1 + X2 ~ N(0, 1), matching Chen (2002), who generates Z ~ N(0,1).

Location normalisation: Lambda_0(y0) = 0, with y0 = 0 (Design I, III), y0 = 1 (Design II).

Estimation intervals:
    Design I:   [-3, 3]
    Design II:  [0.050, 20.086]
    Design III: [-2.178, 2.178]

Sample size: n = 1000, replications: 1000.

Usage:
    python chen_monte_carlo.py --design 1 --n_samples 1000 --n_reps 100
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from transformation_models.chen_estimator import ChenRankEstimator


# ---------------------------------------------------------------------------
# Design specifications
# ---------------------------------------------------------------------------

DESIGNS = {
    1: {
        "name": "Design I: Lambda(y) = y",
        "lambda_title": r"$\Lambda(y) = y$",
        "plot_title": r"Identity: $\Lambda_0(y)=y$",
        "lambda_func": lambda y: y,
        "y0": 0.0,
        "interval": (-3.0, 3.0),
        "beta": np.array([1.0, 1.0]),
    },
    2: {
        "name": "Design II: Lambda(y) = log(y)",
        "lambda_title": r"$\Lambda(y) = \log(y)$",
        "plot_title": r"Logarithmic: $\Lambda_0(y)=\log(y)$",
        "lambda_func": lambda y: np.log(y),
        "y0": 1.0,
        "interval": (0.050, 20.086),
        "beta": np.array([1.0, 1.0]),
    },
    3: {
        "name": "Design III: Lambda(y) = (1/13)*sinh(2y)",
        "lambda_title": r"$\Lambda(y) = \frac{1}{13}\sinh(2y)$",
        "plot_title": r"Hyperbolic sine: $\Lambda_0(y)=\frac{1}{13}\sinh(2y)$",
        "lambda_func": lambda y: (1.0 / 13.0) * np.sinh(2.0 * y),
        "y0": 0.0,
        "interval": (-2.178, 2.178),
        "beta": np.array([1.0, 1.0]),
    },
}


def inverse_lambda(design_id: int, z: np.ndarray) -> np.ndarray:
    """Compute Lambda^{-1}(z) for generating Y from Z + eps."""
    if design_id == 1:
        return z  # Lambda(y) = y => y = z
    elif design_id == 2:
        return np.exp(z)  # Lambda(y) = ln(y) => y = exp(z)
    elif design_id == 3:
        # Lambda(y) = (1/13)*sinh(2y) => y = (1/2)*arcsinh(13*z)
        return 0.5 * np.arcsinh(13.0 * z)
    else:
        raise ValueError(f"Unknown design {design_id}")


def generate_data(
    design_id: int, n: int, rng: np.random.Generator,
    beta_override: np.ndarray = None,
) -> tuple:
    """
    Generate a sample from the specified design.

    Returns (Y, X, Z, beta_true)
    """
    spec = DESIGNS[design_id]
    beta = beta_override if beta_override is not None else spec["beta"]
    q = len(beta)

    # Scale X so that the single index Z = X @ beta has unit variance, matching
    # Chen (2002), who generates Z ~ N(0,1). With X_k iid N(0, sigma^2) we have
    # Var(Z) = sigma^2 ||beta||^2, so sigma = 1/||beta|| gives Var(Z) = 1.
    # (The scale normalisation |beta_1| = 1 is unaffected; only X's scale changes.)
    X = rng.standard_normal((n, q)) / np.linalg.norm(beta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Z = X @ beta
    eps = rng.standard_normal(n)

    # Lambda_0(Y) = Z + eps => Y = Lambda_0^{-1}(Z + eps)
    Y = inverse_lambda(design_id, Z + eps)

    return Y, X, Z, beta


def run_single_replication(
    design_id: int,
    n: int,
    eval_points: np.ndarray,
    rng: np.random.Generator,
    use_true_beta: bool = True,
    beta_override: np.ndarray = None,
) -> np.ndarray:
    """
    Run a single Monte Carlo replication.

    Returns lambda_hat at the evaluation points.
    """
    spec = DESIGNS[design_id]
    Y, X, Z, beta_true = generate_data(design_id, n, rng, beta_override)

    if beta_override is not None:
        beta = beta_override
    elif use_true_beta:
        beta = beta_true
    else:
        from transformation_models.beta_estimators import han_mrc
        beta = han_mrc(Y, X, sign_b1=1.0, rng=rng)

    estimator = ChenRankEstimator(
        y0=spec["y0"],
        estimation_interval=spec["interval"],
        alpha_factor=0.75,
    )

    estimator.fit(
        Y=Y, X=X, beta=beta,
        eval_points=eval_points,
        use_segmentation=True,
    )

    return estimator.lambda_hat_


def run_monte_carlo(
    design_id: int,
    n_samples: int = 1000,
    n_reps: int = 100,
    n_eval: int = 50,
    seed: int = 42,
    use_true_beta: bool = True,
    beta_override: np.ndarray = None,
):
    """
    Run the full Monte Carlo study for a given design.
    """
    spec = DESIGNS[design_id]
    if beta_override is not None:
        spec = dict(spec)
        spec["beta"] = beta_override
    y_low, y_high = spec["interval"]
    eval_points = np.linspace(y_low, y_high, n_eval)
    lambda_true = spec["lambda_func"](eval_points)

    rng = np.random.default_rng(seed)

    results = np.zeros((n_reps, n_eval))

    print(f"\n{'='*60}")
    print(f"Chen (2002) Monte Carlo: {spec['name']}")
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
        results[rep, :] = run_single_replication(
            design_id, n_samples, eval_points, rng, use_true_beta, beta_override,
        )
    total_time = time.time() - t0

    # Compute statistics
    mean_hat = np.nanmean(results, axis=0)
    bias = mean_hat - lambda_true
    sd = np.nanstd(results, axis=0, ddof=1)
    rmse = np.sqrt(bias**2 + sd**2)

    # Print table (matching Chen's Table I/II format)
    print(f"\nCompleted in {total_time:.1f}s ({total_time/n_reps:.2f}s/rep)")
    print(f"\n{'y':>10s} {'Lambda(y)':>10s} {'Mean':>10s} {'Bias':>10s} {'SD':>10s} {'RMSE':>10s}")
    print("-" * 65)
    for i in range(n_eval):
        print(f"{eval_points[i]:10.3f} {lambda_true[i]:10.3f} {mean_hat[i]:10.3f} "
              f"{bias[i]:10.3f} {sd[i]:10.3f} {rmse[i]:10.3f}")

    # Save results to CSV
    csv_path = os.path.join(os.path.dirname(__file__), f"chen_design_{design_id}_results.csv")
    header = "y,lambda_true,mean_hat,bias,sd,rmse"
    csv_data = np.column_stack([eval_points, lambda_true, mean_hat, bias, sd, rmse])
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"Results saved as chen_design_{design_id}_results.csv")

    # Generate one sample for the rug plot (Y observations spread across the interval)
    Y_rug, _, _, _ = generate_data(design_id, n_samples, np.random.default_rng(seed), beta_override)
    Y_rug = Y_rug[(Y_rug >= y_low) & (Y_rug <= y_high)]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dense evaluation for true lambda
    true_eval = np.linspace(y_low, y_high, 100)
    true_lambda_dense = spec["lambda_func"](true_eval)

    ax = axes[0]
    ax.plot(true_eval, true_lambda_dense, "k-", linewidth=2, label=r"$\Lambda_0(y)$")
    ax.plot(eval_points, mean_hat, "r--", linewidth=2, label=r"$\hat{\Lambda}_0(y)$ (mean)")
    ax.fill_between(eval_points, mean_hat - 2*sd, mean_hat + 2*sd, alpha=0.2, color="red")
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
    plt.savefig(os.path.join(os.path.dirname(__file__), f"chen_design_{design_id}.png"), dpi=150)
    plt.show()
    print(f"\nPlot saved as chen_design_{design_id}.png")

    return eval_points, lambda_true, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chen (2002) Monte Carlo study")
    parser.add_argument("--design", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate_beta", action="store_true",
                        help="Estimate beta via Han's MRC instead of using true beta")
    parser.add_argument("--beta", type=float, nargs="+", default=None,
                        help="Override true beta, e.g. --beta 1.0 2.0")
    args = parser.parse_args()

    run_monte_carlo(
        design_id=args.design,
        n_samples=args.n_samples,
        n_reps=args.n_reps,
        n_eval=args.n_eval,
        seed=args.seed,
        use_true_beta=not args.estimate_beta,
        beta_override=np.array(args.beta) if args.beta is not None else None,
    )
