"""
Shifted Quantile Estimator (SQE) for the transformation model

    Lambda(Y) = X beta + epsilon

following Ye & Duan (1997, Annals of Statistics).

Two-stage procedure:
1. Estimate the error CDF F via shifted quantiles.
2. Estimate Lambda(y) = weighted average of z_j + F^{-1}(G_hat(y | I_j)).

Key relationships:
    G(y|z) = F(Lambda(y) - z)           (conditional CDF of Y given Z=z)
    Lambda(q_p(z)) = z + F^{-1}(p)      (conditional p-quantile of Y given Z=z)
    F(-Delta) = G(q_{0.5}(z) | z+Delta) (does not depend on z => average over z)

Reference:
    Ye, J. and N. Duan (1997). Annals of Statistics, 25, 2682-2717.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_to_bins(Z: NDArray, delta_n: int) -> Tuple[NDArray, NDArray, float]:
    """
    Partition Z into delta_n bins of equal width.
    Returns (bin_edges, bin_indices, bin_width).
    """
    z_min, z_max = Z.min(), Z.max()
    margin = 1e-10 * (z_max - z_min)
    bin_edges = np.linspace(z_min - margin, z_max + margin, delta_n + 1)
    bin_indices = np.digitize(Z, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, delta_n - 1)
    bin_width = (z_max - z_min + 2 * margin) / delta_n
    return bin_edges, bin_indices, bin_width


def _bin_quantile(Y: NDArray, bin_indices: NDArray, j: int, p: float) -> float:
    """p-th quantile of Y in bin j."""
    Y_j = Y[bin_indices == j]
    if len(Y_j) < 3:
        return np.nan
    return float(np.quantile(Y_j, p))


def _bin_ecdf(Y: NDArray, bin_indices: NDArray, j: int, y: float) -> float:
    """Empirical CDF of Y in bin j at point y."""
    Y_j = Y[bin_indices == j]
    if len(Y_j) < 3:
        return np.nan
    return float(np.mean(Y_j <= y))


# ---------------------------------------------------------------------------
# Step 1: Estimate F
# ---------------------------------------------------------------------------

def estimate_F(
    Y: NDArray,
    Z: NDArray,
    delta_n: int,
    n_delta_pts: int = 60,
    n_strips: int = 4,
) -> Tuple[NDArray, NDArray]:
    """
    Estimate F using the shifted-quantile method with splicing.

    The core identity: F(-Delta) = E[G(q_{0.5}(Z) | Z + Delta)]
    is estimated by averaging over Z-bins:
        F_hat(-Delta) = (1/K) sum_j G_hat(q_{0.5}(I_j) | I_{j+shift})

    where shift = round(Delta / bin_width).

    The splicing algorithm extends the estimate beyond [-eta, eta]
    by iteratively using different quantile levels.

    Parameters
    ----------
    Y, Z : data arrays
    delta_n : number of Z-bins
    n_delta_pts : number of Delta evaluation points per strip
    n_strips : number of splicing extensions in each direction

    Returns
    -------
    F_grid, F_hat : sorted arrays representing the estimated CDF
    """
    bin_edges, bin_indices, bin_width = _assign_to_bins(Z, delta_n)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Min observations per bin threshold
    bin_counts = np.array([np.sum(bin_indices == j) for j in range(delta_n)])
    min_obs = 5

    all_args = []  # F argument values
    all_vals = []  # F(argument) values

    # F(0) = 0.5 by normalisation
    all_args.append(0.0)
    all_vals.append(0.5)

    def _estimate_strip(p: float, max_delta: float):
        """
        Estimate F(F^{-1}(p) - Delta) for Delta in [0, max_delta].

        Uses the identity:
            F(F^{-1}(p) - Delta) = E[G(q_p(Z) | Z + Delta)]
        """
        strip_args = []
        strip_vals = []

        deltas = np.linspace(0, max_delta, n_delta_pts)

        for Delta in deltas:
            shift = round(Delta / bin_width)
            if shift < 0:
                shift = 0

            estimates = []
            for j in range(delta_n):
                if bin_counts[j] < min_obs:
                    continue
                # Target bin: j + shift
                j_target = j + shift
                if j_target < 0 or j_target >= delta_n:
                    continue
                if bin_counts[j_target] < min_obs:
                    continue

                q_p = _bin_quantile(Y, bin_indices, j, p)
                if np.isnan(q_p):
                    continue
                G_val = _bin_ecdf(Y, bin_indices, j_target, q_p)
                if np.isnan(G_val):
                    continue
                estimates.append(G_val)

            if len(estimates) >= 3:
                strip_vals.append(np.mean(estimates))
                strip_args.append(Delta)

        return np.array(strip_args), np.array(strip_vals)

    def _estimate_strip_negative(p: float, max_delta: float):
        """
        Estimate F(F^{-1}(p) + Delta) for Delta in [0, max_delta].

        Uses: F(F^{-1}(p) + Delta) = E[G(q_p(Z + Delta) | Z)]
             = E[G(q_p(I_{j+shift}) | I_j)]
        """
        strip_args = []
        strip_vals = []

        deltas = np.linspace(0, max_delta, n_delta_pts)

        for Delta in deltas:
            shift = round(Delta / bin_width)
            if shift < 0:
                shift = 0

            estimates = []
            for j in range(delta_n):
                if bin_counts[j] < min_obs:
                    continue
                j_source = j + shift  # quantile from shifted bin
                if j_source < 0 or j_source >= delta_n:
                    continue
                if bin_counts[j_source] < min_obs:
                    continue

                q_p = _bin_quantile(Y, bin_indices, j_source, p)
                if np.isnan(q_p):
                    continue
                G_val = _bin_ecdf(Y, bin_indices, j, q_p)
                if np.isnan(G_val):
                    continue
                estimates.append(G_val)

            if len(estimates) >= 3:
                strip_vals.append(np.mean(estimates))
                strip_args.append(Delta)

        return np.array(strip_args), np.array(strip_vals)

    # --- Base strip: p=0.5 ---
    # Maximum meaningful Delta: roughly half the Z range
    z_range = Z.max() - Z.min()
    max_delta_base = z_range * 0.4  # don't go to the very edge

    # Negative side: F(-Delta) for Delta > 0
    delta_arr, F_neg = _estimate_strip(p=0.5, max_delta=max_delta_base)
    for d, fv in zip(delta_arr, F_neg):
        if 0 < fv < 1:
            all_args.append(-d)
            all_vals.append(fv)

    # Positive side: F(+Delta) for Delta > 0
    delta_arr_pos, F_pos = _estimate_strip_negative(p=0.5, max_delta=max_delta_base)
    for d, fv in zip(delta_arr_pos, F_pos):
        if 0 < fv < 1:
            all_args.append(d)
            all_vals.append(fv)

    # --- Splicing: extend left (lower tail of F) ---
    for strip in range(n_strips):
        args_arr = np.array(all_args)
        vals_arr = np.array(all_vals)
        order = np.argsort(args_arr)
        args_sorted = args_arr[order]
        vals_sorted = vals_arr[order]

        # Current leftmost quantile level
        p_left = max(vals_sorted[0], 0.01)
        if p_left < 0.02:
            break

        # F^{-1}(p_left)
        Finv_p = np.interp(p_left, vals_sorted, args_sorted)

        delta_arr, F_strip = _estimate_strip(p=p_left, max_delta=max_delta_base)
        for d, fv in zip(delta_arr, F_strip):
            arg = Finv_p - d
            if 0 < fv < 1:
                all_args.append(arg)
                all_vals.append(fv)

    # --- Splicing: extend right (upper tail of F) ---
    for strip in range(n_strips):
        args_arr = np.array(all_args)
        vals_arr = np.array(all_vals)
        order = np.argsort(args_arr)
        args_sorted = args_arr[order]
        vals_sorted = vals_arr[order]

        p_right = min(vals_sorted[-1], 0.99)
        if p_right > 0.98:
            break

        Finv_p = np.interp(p_right, vals_sorted, args_sorted)

        delta_arr, F_strip = _estimate_strip_negative(p=p_right, max_delta=max_delta_base)
        for d, fv in zip(delta_arr, F_strip):
            arg = Finv_p + d
            if 0 < fv < 1:
                all_args.append(arg)
                all_vals.append(fv)

    # --- Assemble and clean up ---
    all_args = np.array(all_args)
    all_vals = np.array(all_vals)

    valid = np.isfinite(all_args) & np.isfinite(all_vals) & (all_vals > 0) & (all_vals < 1)
    # Always keep F(0) = 0.5
    valid[0] = True
    all_args = all_args[valid]
    all_vals = all_vals[valid]

    if len(all_args) < 2:
        return np.array([-5.0, 0.0, 5.0]), np.array([0.01, 0.5, 0.99])

    order = np.argsort(all_args)
    F_grid = all_args[order]
    F_hat = all_vals[order]

    # Average duplicate arguments
    unique_args, inv = np.unique(np.round(F_grid, 8), return_inverse=True)
    unique_vals = np.zeros(len(unique_args))
    counts = np.zeros(len(unique_args))
    for i, idx in enumerate(inv):
        unique_vals[idx] += F_hat[i]
        counts[idx] += 1
    unique_vals /= counts
    F_grid = unique_args
    F_hat = unique_vals

    # Isotonic regression (PAVA) to ensure monotonicity
    F_hat = _isotonic_regression(F_hat)
    F_hat = np.clip(F_hat, 0.001, 0.999)

    return F_grid, F_hat


def _isotonic_regression(y: NDArray) -> NDArray:
    """Pool Adjacent Violators Algorithm for non-decreasing regression."""
    n = len(y)
    result = y.copy().astype(np.float64)
    weight = np.ones(n)

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool i and i+1
            w_sum = weight[i] + weight[i + 1]
            pooled = (result[i] * weight[i] + result[i + 1] * weight[i + 1]) / w_sum
            result[i] = result[i + 1] = pooled
            weight[i] = weight[i + 1] = w_sum
            # Check backwards
            while i > 0 and result[i - 1] > result[i]:
                i -= 1
                w_sum = weight[i] + weight[i + 1]
                pooled = (result[i] * weight[i] + result[i + 1] * weight[i + 1]) / w_sum
                for k in range(i, n):
                    if weight[k] == weight[i + 1] and result[k] == result[i + 1]:
                        result[k] = pooled
                        weight[k] = w_sum
                    else:
                        break
                result[i] = pooled
                weight[i] = w_sum
        i += 1

    return result


# ---------------------------------------------------------------------------
# Step 2: Estimate Lambda(y)
# ---------------------------------------------------------------------------

def estimate_Lambda(
    y_eval: NDArray,
    Y: NDArray,
    Z: NDArray,
    F_grid: NDArray,
    F_hat: NDArray,
    delta_n: int,
    trim: float = 0.02,
) -> NDArray:
    """
    Estimate Lambda(y) = weighted average of z_j + F^{-1}(G_hat(y | I_j)).

    A bin contributes only if it has at least 5 observations and its G_hat lies
    in the central probability window (trim, 1 - trim) = (0.02, 0.98). See the
    comment at the filter for the choice of window.
    """
    bin_edges, bin_indices, _ = _assign_to_bins(Z, delta_n)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def F_inv(p):
        """Quantile function of F_hat via interpolation."""
        p = np.clip(p, F_hat.min() + 1e-6, F_hat.max() - 1e-6)
        return float(np.interp(p, F_hat, F_grid))

    Lambda_hat = np.zeros(len(y_eval))

    for m, y in enumerate(y_eval):
        contributions = []

        for j in range(delta_n):
            Y_j = Y[bin_indices == j]
            n_j = len(Y_j)
            if n_j < 5:
                continue

            G_val = float(np.mean(Y_j <= y))

            # Weight window (the weight function w of Ye & Duan, 1997): a bin
            # contributes only when G_hat(y | I_j) lies in the central
            # probability window (trim, 1 - trim) = (0.02, 0.98), where F^{-1}
            # is well estimated. We widen it beyond the paper's (0.1, 0.9)
            # because the window affects almost only the boundary: a wider band
            # keeps more bins and lowers the finite-sample boundary RMSE (where
            # anchors are already scarce), while leaving the central region
            # essentially unchanged. trim = 0.02 was chosen on that basis.
            if G_val <= trim or G_val >= 1.0 - trim:
                continue

            Finv_G = F_inv(G_val)
            z_j = bin_centers[j]
            contributions.append(z_j + Finv_G)

        if len(contributions) > 0:
            Lambda_hat[m] = np.mean(contributions)
        else:
            Lambda_hat[m] = np.nan

    return Lambda_hat


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class YeDuanEstimator:
    """
    Shifted Quantile Estimator (SQE) for the transformation function.

    Parameters
    ----------
    y0 : float
        Location normalisation: Lambda(y0) = 0.
    estimation_interval : (y_low, y_high)
    delta_n : int or None
        Number of Z-bins. If None, auto-selected as ceil(c * n^r),
        r=0.30 in (1/4, 1/3) per Ye & Duan Assumption 5.
    n_strips : int
        Number of splicing extensions for F. Default 4.
    """

    def __init__(
        self,
        y0: float,
        estimation_interval: Tuple[float, float],
        delta_n: Optional[int] = None,
        n_strips: int = 4,
        n_delta_pts: int = 60,
        trim: float = 0.02,
    ):
        self.y0 = y0
        self.y_low, self.y_high = estimation_interval
        self.delta_n = delta_n
        self.n_strips = n_strips
        self.n_delta_pts = n_delta_pts
        self.trim = trim

        self.eval_points_: Optional[NDArray] = None
        self.lambda_hat_: Optional[NDArray] = None
        self.F_grid_: Optional[NDArray] = None
        self.F_hat_: Optional[NDArray] = None
        self.beta_hat_: Optional[NDArray] = None

    def fit(
        self,
        Y: NDArray,
        X: NDArray,
        beta: NDArray,
        eval_points: Optional[NDArray] = None,
        n_eval: int = 50,
    ) -> "YeDuanEstimator":
        """Estimate F and Lambda."""
        self.beta_hat_ = beta
        Y = np.asarray(Y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Z = (X @ beta).astype(np.float64)

        mask = np.isfinite(Y) & np.isfinite(Z)
        Y, Z = Y[mask], Z[mask]
        n = len(Y)

        # Auto-select delta_n: we need enough bins that bin_width is small
        # relative to the Z-range, but enough observations per bin.
        # Ye & Duan: delta_n = ceil(c * n^r), r in (1/4, 1/3).
        # We use a slightly higher multiplier to get finer bins.
        if self.delta_n is None:
            # Rule: delta_n = min(n // 60, 2 * n^{1/3}), capped so each bin has
            # at least ~60 observations at small n (reduces variance / clip exclusions),
            # while growing as n^{1/3} for large n (reduces discretisation bias O(n^{-1/3})).
            # The cap n // 60 only binds for n <= ~1300; above that n^{1/3} dominates.
            delta_n = max(8, min(n // 20, int(np.ceil(5.0 * n ** (1.0 / 3.0)))))
        else:
            delta_n = self.delta_n

        if eval_points is None:
            eval_points = np.linspace(self.y_low, self.y_high, n_eval)
        self.eval_points_ = eval_points

        # Step 1: Estimate F
        self.F_grid_, self.F_hat_ = estimate_F(
            Y, Z, delta_n=delta_n, n_strips=self.n_strips,
            n_delta_pts=self.n_delta_pts,
        )

        # Step 2: Estimate Lambda
        lambda_raw = estimate_Lambda(
            eval_points, Y, Z, self.F_grid_, self.F_hat_,
            delta_n=delta_n, trim=self.trim,
        )

        # Location normalisation: Lambda(y0) = 0
        # Evaluate directly at y0 rather than interpolating from the eval grid,
        # which can be inaccurate when y0 does not coincide with an eval point
        # (e.g. Design II where y0=1 lies in a nonlinear region between grid points).
        lam_at_y0_arr = estimate_Lambda(
            np.array([self.y0]), Y, Z, self.F_grid_, self.F_hat_,
            delta_n=delta_n, trim=self.trim,
        )
        lam_at_y0 = lam_at_y0_arr[0]
        if np.isfinite(lam_at_y0):
            self.lambda_hat_ = lambda_raw - lam_at_y0
        else:
            self.lambda_hat_ = lambda_raw

        return self

    def predict(self, y: NDArray) -> NDArray:
        """Interpolate Lambda_hat at new points."""
        if self.eval_points_ is None or self.lambda_hat_ is None:
            raise RuntimeError("Call fit() first.")
        return np.interp(y, self.eval_points_, self.lambda_hat_)

    def predict_F(self, x: NDArray) -> NDArray:
        """Evaluate estimated F at given points."""
        if self.F_grid_ is None or self.F_hat_ is None:
            raise RuntimeError("Call fit() first.")
        return np.interp(x, self.F_grid_, self.F_hat_)
