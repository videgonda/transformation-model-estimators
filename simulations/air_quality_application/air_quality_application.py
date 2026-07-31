"""
Application of the Chen (2002) and Ye & Duan (1997) estimators to
air quality data.

Model:  Lambda(Y) = X beta + epsilon

    Y  = PT08.S3(NOx)   (NOx metal oxide sensor response)
    X  = (PT08.S2(NMHC), PT08.S5(O3))   (NMHC and O3 sensor responses)

The script:
1. Loads and cleans the CSV.
2. Estimates beta via Han's MRC.
3. Fits both estimators (Chen rank, Ye-Duan SQE).
4. Exports CSV files for pgfplots (thesis figures).

Usage:
    python simulations/air_quality_application.py --csv path/to/air_quality.csv

Output (in results/):
    lambda_estimates.csv   — Lambda_hat(y) for both estimators + reference curves
    F_hat_ye_duan.csv      — Estimated error CDF from Ye-Duan
    scatter_data.csv       — (Y, Z) subsample for scatter plot
"""

import argparse
import sys
import os
import time
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from transformation_models.chen_estimator import ChenRankEstimator
from transformation_models.ye_duan_estimator import YeDuanEstimator
from transformation_models.beta_estimators import han_mrc



def load_and_clean(csv_path: str):
    """
    Load the air quality CSV and return cleaned (Y, X) arrays.

    Handles both comma- and semicolon-delimited files.
    Drops rows with missing values (coded as -200).
    """
    import csv

    # Detect delimiter
    with open(csv_path, "r") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    sep = dialect.delimiter

    # Read header
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader)

    # Strip whitespace and BOM from header
    header = [h.strip().strip("\ufeff") for h in header]

    # Find columns
    # Common names in the UCI Air Quality dataset
    y_candidates    = ["PT08.S3(NOx)",  "PT08.S3.NOx.", "PT08_S3_NOx"]
    nmhc_candidates = ["PT08.S2(NMHC)", "PT08.S2.NMHC.", "PT08_S2_NMHC"]
    o3_candidates   = ["PT08.S5(O3)",   "PT08.S5.O3.",   "PT08_S5_O3"]

    def normalize(col_name: str) -> str:
        return "".join(ch.lower() for ch in col_name if ch.isalnum())

    def find_col(candidates):
        # Prefer exact normalized matches first, then fallback to substring matches.
        for c in candidates:
            target = normalize(c)
            for i, h in enumerate(header):
                if normalize(h) == target:
                    return i, h
        for c in candidates:
            target = normalize(c)
            for i, h in enumerate(header):
                if target in normalize(h):
                    return i, h
        return None, None

    y_idx,    y_name    = find_col(y_candidates)
    nmhc_idx, nmhc_name = find_col(nmhc_candidates)
    o3_idx,   o3_name   = find_col(o3_candidates)

    if y_idx is None:
        raise ValueError(f"Could not find Y column. Header: {header}")
    if nmhc_idx is None:
        raise ValueError(f"Could not find NMHC column. Header: {header}")
    if o3_idx is None:
        raise ValueError(f"Could not find O3 column. Header: {header}")

    print(f"  Y  column: {y_name} (index {y_idx})")
    print(f"  X1 column: {nmhc_name} (index {nmhc_idx})")
    print(f"  X2 column: {o3_name} (index {o3_idx})")

    # Read data
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=sep)
        next(reader)  # skip header
        for row in reader:
            if len(row) <= max(y_idx, nmhc_idx, o3_idx):
                continue
            try:
                # Handle comma as decimal separator
                def parse(s):
                    return float(s.strip().replace(",", "."))

                y_val    = parse(row[y_idx])
                nmhc_val = parse(row[nmhc_idx])
                o3_val   = parse(row[o3_idx])
                rows.append([y_val, nmhc_val, o3_val])
            except (ValueError, IndexError):
                continue

    data = np.array(rows)
    print(f"  Raw rows loaded: {len(data)}")

    # Remove missing value indicators (-200)
    mask = np.all(data > -200, axis=1)
    data = data[mask]
    print(f"  After removing missing (-200): {len(data)}")

    # Remove non-finite
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    print(f"  After removing non-finite: {len(data)}")

    Y = data[:, 0]
    X = data[:, 1:]

    return Y, X

def engression_lambda(Z, Y, eval_points, y0,
                      n_anchors=40, sample_size=2000,
                      num_epoches=2000, num_layer=4, add_bn=False, lr=5e-4,
                      seed=0):
    """Recover Lambda_hat from engression via inversion of the conditional
    median curve. Uses only the median normalisation F_eps^{-1}(0.5)=0, so
    no knowledge of F_eps is required (unlike the Monte Carlo extraction)."""
    import torch
    import engression as eng

    torch.manual_seed(seed)
    Zt = torch.tensor(Z.reshape(-1, 1), dtype=torch.float32)
    Yt = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)

    model = eng.engression(
        Zt, Yt,
        num_layer=num_layer, hidden_dim=100, noise_dim=100,
        add_bn=add_bn, lr=lr, num_epoches=num_epoches,
        batch_size=256, standardize=True, device="cpu", verbose=False,
        print_every_nepoch=num_epoches + 1,
    )

    # anchors over the central range of the observed index
    z_lo, z_hi = np.quantile(Z, [0.025, 0.975])
    z_grid = np.linspace(z_lo, z_hi, n_anchors).astype(np.float32)

    with torch.no_grad():
        samples = model.sample(torch.tensor(z_grid.reshape(-1, 1)),
                               sample_size=sample_size)
    samples = samples.squeeze(1).cpu().numpy()       # (n_anchors, sample_size)

    m = np.median(samples, axis=1)                   # conditional median m(z_j)

    # Lambda is the inverse of the median curve: Lambda(m(z)) = z
    order = np.argsort(m)
    lam  = np.interp(eval_points, m[order], z_grid[order], left=np.nan, right=np.nan)
    lam0 = np.interp(y0,          m[order], z_grid[order])
    return lam - lam0                                 # normalise Lambda(y0)=0

def main():
    parser = argparse.ArgumentParser(
        description="Air quality application of transformation model estimators"
    )
    parser.add_argument(
    "--csv", type=str,
    default=os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "AirQualityUCI.csv")),
    help="Path to the air quality CSV file"
)
    parser.add_argument(
        "--n_eval", type=int, default=80,
        help="Number of evaluation points for Lambda_hat"
    )
    parser.add_argument(
        "--subsample", type=int, default=2000,
        help="Number of points for scatter plot CSV"
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading data ===")
    Y, X = load_and_clean(args.csv)
    n = len(Y)
    print(f"  Final sample size: n = {n}")
    print(f"  Y  (NOx sensor)  range: [{Y.min():.1f}, {Y.max():.1f}]")
    print(f"  X1 (NMHC sensor) range: [{X[:,0].min():.1f}, {X[:,0].max():.1f}]")
    print(f"  X2 (O3 sensor)   range: [{X[:,1].min():.1f}, {X[:,1].max():.1f}]")

    # ------------------------------------------------------------------
    # 2. Estimate beta via Han's MRC
    # ------------------------------------------------------------------
    print("\n=== Estimating beta (Han's MRC) ===")
    # Subsample for beta estimation (MRC is O(n^2))
    if n > 3000:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n, 3000, replace=False)
        Y_sub, X_sub = Y[idx_sub], X[idx_sub]
        print(f"  Using subsample of {len(Y_sub)} for beta estimation")
    else:
        Y_sub, X_sub = Y, X

    t0 = time.time()
    beta_hat = han_mrc(Y_sub, X_sub, sign_b1=-1.0, n_restarts=10)
    t_beta = time.time() - t0
    print(f"  beta_hat = {beta_hat}")
    print(f"  Time: {t_beta:.1f}s")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Z = X @ beta_hat

    # ------------------------------------------------------------------
    # 3. Set up estimation
    # ------------------------------------------------------------------
    y_median = np.median(Y)
    y_low = np.percentile(Y, 5)
    y_high = np.percentile(Y, 95)
    eval_points = np.linspace(y_low, y_high, args.n_eval)

    print(f"\n  y0 (median): {y_median:.1f}")
    print(f"  Estimation interval: [{y_low:.1f}, {y_high:.1f}]")

    # ------------------------------------------------------------------
    # 4. Chen rank estimator
    # ------------------------------------------------------------------
    print("\n=== Fitting Chen (2002) rank estimator ===")
    t0 = time.time()
    chen = ChenRankEstimator(
        y0=y_median,
        estimation_interval=(y_low, y_high),
        alpha_factor=0.75,
        max_candidates=800,
    )
    chen.fit(Y=Y, X=X, beta=beta_hat, eval_points=eval_points)
    t_chen = time.time() - t0
    print(f"  Time: {t_chen:.1f}s")

    # ------------------------------------------------------------------
    # 5. Ye & Duan SQE
    # ------------------------------------------------------------------
    print("\n=== Fitting Ye & Duan (1997) SQE ===")
    # The Ye-Duan estimator is scale-equivariant, and it is defined on the same
    # index scale Z = X @ beta as Chen's estimator, so both are directly
    # comparable with no rescaling of the index.
    t0 = time.time()
    yd = YeDuanEstimator(
        y0=y_median,
        estimation_interval=(y_low, y_high),
        delta_n=60,
    )
    yd.fit(Y=Y, X=X, beta=beta_hat, eval_points=eval_points)
    t_yd = time.time() - t0
    print(f"  Time: {t_yd:.1f}s")

    # ------------------------------------------------------------------
    # 6. Reference curves (shifted and scaled to match at y0)
    # ------------------------------------------------------------------
    
    chen_range = np.nanmax(chen.lambda_hat_) - np.nanmin(chen.lambda_hat_)

    def scale_ref(ref):
        r = np.nanmax(ref) - np.nanmin(ref)
        if r < 1e-10:
            return ref
        return ref * (chen_range / r)

    # log(y) - log(y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ref = np.log(eval_points) - np.log(y_median)
    log_ref[~np.isfinite(log_ref)] = np.nan
    log_ref = scale_ref(log_ref)


    # sqrt(y) - sqrt(y0), scaled
    sqrt_ref = scale_ref(np.sqrt(eval_points) - np.sqrt(y_median))

    # identity: y - y0
    id_ref = scale_ref(eval_points - y_median)

    # ------------------------------------------------------------------
    # 6b. Engression Lambda (median-curve inversion)
    # ------------------------------------------------------------------
    print("\n=== Fitting engression ===")
    t0 = time.time()
    lambda_eng = engression_lambda(Z, Y, eval_points, y_median)
    t_eng = time.time() - t0
    print(f"  Time: {t_eng:.1f}s")

    # ------------------------------------------------------------------
    # 7. Export CSVs
    # ------------------------------------------------------------------
    print("\n=== Exporting results ===")

    # Lambda estimates
    lambda_path = os.path.join(results_dir, "lambda_estimates.csv")
    hdr = "y,chen,ye_duan,engression,log,sqrt,identity"
    data_out = np.column_stack([
        eval_points,
        chen.lambda_hat_,
        yd.lambda_hat_,
        lambda_eng,
        log_ref,
        sqrt_ref,
        id_ref,
    ])
    np.savetxt(lambda_path, data_out, delimiter=",", header=hdr, comments="", fmt="%.6f")
    print(f"  Saved: {lambda_path}")

    # F_hat from Ye-Duan
    f_path = os.path.join(results_dir, "F_hat_ye_duan.csv")
    hdr_f = "x,F_hat"
    f_data = np.column_stack([yd.F_grid_, yd.F_hat_])
    np.savetxt(f_path, f_data, delimiter=",", header=hdr_f, comments="", fmt="%.6f")
    print(f"  Saved: {f_path}")

    # Scatter plot data (subsample)
    # Centre Z to match the Lambda normalisation Lambda(y0)=0:
    # subtract c = median(Z - Lambda(Y)) so that Z≈0 when Y≈y0,
    # making the reference line pass through the origin.
    lambda_at_Y = np.interp(Y, eval_points, chen.lambda_hat_, left=np.nan, right=np.nan)
    valid_mask  = np.isfinite(lambda_at_Y)
    c_shift     = np.nanmedian(Z[valid_mask] - lambda_at_Y[valid_mask])
    Z_centred   = Z - c_shift

    scatter_path = os.path.join(results_dir, "scatter_data.csv")
    rng = np.random.default_rng(123)
    n_scatter = min(args.subsample, n)
    idx_scatter = rng.choice(n, n_scatter, replace=False)
    scatter_data = np.column_stack([Y[idx_scatter], Z_centred[idx_scatter]])
    np.savetxt(scatter_path, scatter_data, delimiter=",", header="Y,Z", comments="", fmt="%.6f")
    print(f"  Saved: {scatter_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"  beta_hat:        {beta_hat}")
    print(f"  Chen Lambda range: [{chen.lambda_hat_.min():.2f}, {chen.lambda_hat_.max():.2f}]")
    print(f"  YD Lambda range:   [{yd.lambda_hat_.min():.2f}, {yd.lambda_hat_.max():.2f}]")
    print(f"  Eng Lambda range:  [{np.nanmin(lambda_eng):.2f}, {np.nanmax(lambda_eng):.2f}]")
    print(f"  F_hat range:       [{yd.F_hat_.min():.3f}, {yd.F_hat_.max():.3f}]")
    print(f"  Timing — beta: {t_beta:.1f}s, Chen: {t_chen:.1f}s, YD: {t_yd:.1f}s, Eng: {t_eng:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
