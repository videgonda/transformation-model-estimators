"""
Monte Carlo study using the Engression estimator to estimate the
transformation function Lambda_0, following the same designs as
Chen (2002) and Ye & Duan (1997).

Estimation approach (multi-anchor, via the Ye-Duan identity)
-------------------------------------------------------------
Under the model Lambda_0(Y) = Z + eps with eps ~ N(0,1) (F = Phi), the
quantile identity holds for EVERY anchor value z:

    Lambda_0(y) = z + Phi^{-1}( G(y | z) ).

Engression provides a sampler for Y | X = x, hence an estimate of
G(y | z) at any anchor.  So we:
    1. Fit engression(X, Y).
    2. Choose a grid of anchors z_1, ..., z_m spanning the central range
       of Z = beta^T X, with anchor covariates x_j = z_j * beta / |beta|^2.
    3. Estimate G_hat(y | z_j) as the fraction of engression samples <= y.
    4. Average Lambda_hat(y) = mean_j [ z_j + Phi^{-1}(G_hat(y | z_j)) ]
       over the anchors with G_hat in [w_lo, w_hi] (Ye-Duan-style weight
       function: Phi^{-1} amplifies errors near 0 and 1).
    5. Subtract Lambda_hat(y0) to enforce Lambda_hat(y0) = 0.

Each Lambda_0(y) is thus estimated from central probabilities at
suitable anchors rather than from tail quantiles at a single point,
which covers the full evaluation range and reduces the variance.

Usage:
    python3 engression_monte_carlo.py --design 1 --n_samples 1000 --n_reps 100
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

# Path to src and shared Monte Carlo helpers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chen_monte_carlo import DESIGNS, generate_data
import engression as eng


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def run_single_replication(
    design_id: int,
    n: int,
    eval_points: np.ndarray,
    rng: np.random.Generator,
    num_epoches: int = 300,
    hidden_dim: int = 100,
    noise_dim: int = 100,
    num_layer: int = 2,
    add_bn: bool = True,
    lr: float = 0.001,
    resblock: bool = False,
    n_anchors: int = 40,
    sample_size: int = 2000,
    w_lo: float = 0.05,
    w_hi: float = 0.95,
) -> np.ndarray:
    """
    Fit one engression model and return Lambda_hat at eval_points.

    Parameters
    ----------
    n_anchors : int
        Number of anchor values z_j spanning the central range of Z.
    sample_size : int
        Samples drawn from engression at each anchor to estimate G(y | z_j).
    w_lo, w_hi : float
        Ye-Duan-style weight window: anchors only contribute to
        Lambda_hat(y) when G_hat(y | z_j) lies in [w_lo, w_hi].
    """
    spec = DESIGNS[design_id]
    Y, X, _, _ = generate_data(design_id, n, rng)
    y0 = spec["y0"]
    beta = spec["beta"]

    # --- Fit engression -------------------------------------------------------
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)

    engressor = eng.engression(
        X_t, Y_t,
        num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim,
        resblock=resblock,
        add_bn=add_bn, lr=lr, standardize=True,
        num_epoches=num_epoches,
        batch_size=min(256, n),
        verbose=False,
        print_every_nepoch=num_epoches + 1,  # suppress epoch-level prints
        device="cpu",
    )

    # --- Multi-anchor extraction: Lambda_0(y) = z + Phi^{-1}(G(y | z)) -------
    Z = X @ beta
    z_lo, z_hi = np.quantile(Z, [0.025, 0.975])
    z_grid = np.linspace(z_lo, z_hi, n_anchors)
    # anchor covariates x_j with beta^T x_j = z_j
    X_anchor = np.outer(z_grid, beta / (beta @ beta))
    X_anchor_t = torch.tensor(X_anchor, dtype=torch.float32)

    with torch.no_grad():
        samples = engressor.sample(X_anchor_t, sample_size=sample_size)
    samples = samples.squeeze(1).cpu().numpy()  # (n_anchors, sample_size)

    def lambda_hat_at(y: float) -> float:
        G_hat = (samples <= y).mean(axis=1)
        valid = (G_hat >= w_lo) & (G_hat <= w_hi)
        if not valid.any():
            return np.nan
        return float(np.mean(z_grid[valid] + norm.ppf(G_hat[valid])))

    lambda_hat = np.array([lambda_hat_at(y) for y in eval_points])

    # --- Location normalisation: Lambda_hat(y0) = 0 -------------------------
    lambda_at_y0 = lambda_hat_at(y0)
    if np.isfinite(lambda_at_y0):
        lambda_hat -= lambda_at_y0

    return lambda_hat


# ---------------------------------------------------------------------------
# Full Monte Carlo run
# ---------------------------------------------------------------------------

def run_monte_carlo(
    design_id: int = 1,
    n_samples: int = 1000,
    n_reps: int = 100,
    n_eval: int = 50,
    seed: int = 42,
    num_epoches: int = 300,
    hidden_dim: int = 100,
    noise_dim: int = 100,
    num_layer: int = 2,
    add_bn: bool = True,
    lr: float = 0.001,
):
    spec = DESIGNS[design_id]
    y_low, y_high = spec["interval"]
    eval_points = np.linspace(y_low, y_high, n_eval)
    lambda_true = spec["lambda_func"](eval_points)

    rng = np.random.default_rng(seed)
    results = np.zeros((n_reps, n_eval))

    print(f"\n{'='*60}")
    print(f"Engression Monte Carlo: {spec['name']}")
    print(f"n = {n_samples}, replications = {n_reps}")
    print(f"Evaluation interval: [{y_low}, {y_high}]")
    print(f"y0 = {spec['y0']}, beta_true = {spec['beta']}")
    print(f"num_epoches = {num_epoches}, hidden_dim = {hidden_dim}, noise_dim = {noise_dim}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for rep in range(n_reps):
        if (rep + 1) % 5 == 0 or rep == 0:
            elapsed = time.time() - t0
            print(f"  Replication {rep+1}/{n_reps}  (elapsed: {elapsed:.1f}s)")
        results[rep, :] = run_single_replication(
            design_id, n_samples, eval_points, rng,
            num_epoches=num_epoches, hidden_dim=hidden_dim, noise_dim=noise_dim,
            num_layer=num_layer, add_bn=add_bn, lr=lr,
        )
    total_time = time.time() - t0

    # --- Summary statistics --------------------------------------------------
    mean_hat = np.nanmean(results, axis=0)
    bias     = mean_hat - lambda_true
    sd       = np.nanstd(results, axis=0, ddof=1)
    rmse     = np.sqrt(bias**2 + sd**2)

    print(f"\nCompleted in {total_time:.1f}s ({total_time/n_reps:.2f}s/rep)")
    print(f"\n{'y':>10s} {'Lambda(y)':>10s} {'Mean':>10s} "
          f"{'Bias':>10s} {'SD':>10s} {'RMSE':>10s}")
    print("-" * 65)
    for i in range(n_eval):
        print(f"{eval_points[i]:10.3f} {lambda_true[i]:10.3f} {mean_hat[i]:10.3f} "
              f"{bias[i]:10.3f} {sd[i]:10.3f} {rmse[i]:10.3f}")

    # --- Save results CSV --------------
    csv_path = os.path.join(os.path.dirname(__file__),
                            f"engression_design_{design_id}_results.csv")
    np.savetxt(csv_path,
               np.column_stack([eval_points, lambda_true, mean_hat, bias, sd, rmse]),
               delimiter=",", fmt="%.6f", comments="",
               header="y,lambda_true,mean_hat,bias,sd,rmse")
    print(f"\nResults saved as {os.path.basename(csv_path)}")

    # --- Plot ----------------------------------------------------------------
    Y_rug, _, _, _ = generate_data(design_id, n_samples, np.random.default_rng(seed))
    Y_rug = Y_rug[(Y_rug >= y_low) & (Y_rug <= y_high)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    true_eval = np.linspace(y_low, y_high, 100)
    true_lambda_dense = spec["lambda_func"](true_eval)

    ax = axes[0]
    ax.plot(true_eval, true_lambda_dense, "k-", linewidth=2, label=r"$\Lambda_0(y)$")
    ax.plot(eval_points, mean_hat, "g--", linewidth=2,
            label=r"$\hat{\Lambda}_0(y)$ (mean)")
    ax.fill_between(eval_points, mean_hat - 2*sd, mean_hat + 2*sd,
                    alpha=0.2, color="green")
    from matplotlib.transforms import blended_transform_factory
    rug_transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(Y_rug, np.zeros(len(Y_rug)), "|", transform=rug_transform,
            color="gray", alpha=0.15, markersize=8, clip_on=False, label="observed $Y$")
    ax.set_xlabel("y")
    ax.set_ylabel(r"$\Lambda(y)$")
    ax.set_title(f"Engression — {spec['lambda_title']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(eval_points, bias, "b-o", label="Bias", markersize=3)
    ax.plot(eval_points, rmse, "r-s", label="RMSE", markersize=3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("y")
    ax.set_title("Bias and RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__),
                           f"engression_design_{design_id}.png")
    plt.savefig(outpath, dpi=150)
    plt.show()
    print(f"\nPlot saved as engression_design_{design_id}.png")

    return eval_points, lambda_true, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engression Monte Carlo study")
    parser.add_argument("--design",      type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--n_samples",   type=int, default=1000)
    parser.add_argument("--n_reps",      type=int, default=100)
    parser.add_argument("--n_eval",      type=int, default=50)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_epoches", type=int, default=300,
                        help="Training epochs per replication")
    parser.add_argument("--hidden_dim",  type=int, default=100)
    parser.add_argument("--noise_dim",   type=int, default=100)
    parser.add_argument("--num_layer",   type=int, default=2)
    parser.add_argument("--no_bn",       action="store_true",
                        help="Disable batch normalisation")
    parser.add_argument("--lr",          type=float, default=0.001)
    args = parser.parse_args()

    run_monte_carlo(
        design_id=args.design,
        n_samples=args.n_samples,
        n_reps=args.n_reps,
        n_eval=args.n_eval,
        seed=args.seed,
        num_epoches=args.num_epoches,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        num_layer=args.num_layer,
        add_bn=not args.no_bn,
        lr=args.lr,
    )
