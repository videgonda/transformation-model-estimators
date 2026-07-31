"""
Rank estimator for the transformation function Lambda_0 in the model

    Lambda_0(Y) = X beta + epsilon

following Chen (2002, Econometrica).

Efficient O(n log n) implementation per evaluation point using ranks
and searchsorted. The segmentation scheme from Section 4 divides the
estimation interval into subintervals to improve finite-sample performance.

Reference:
    Chen, S. (2002). "Rank Estimation of Transformation Models."
    Econometrica, 70(4), 1683-1697.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core: efficient single-point estimator
# ---------------------------------------------------------------------------

def _eval_objective_at_candidates(
    d_iy: NDArray,
    d_jy0: NDArray,
    Z: NDArray,
    Z_sorted: NDArray,
    candidates: NDArray,
) -> NDArray:
    """
    Vectorised evaluation of Gamma_n(y, Lambda, b) at multiple Lambda candidates.

    Gamma_n = 1/(n(n-1)) sum_{i!=j} (d_{iy} - d_{jy0}) 1{Z_i - Z_j >= Lambda}

    Rewritten as:
        T1(Lambda) = sum_i d_{iy} * #{j: Z_j <= Z_i - Lambda}
        T2(Lambda) = sum_j d_{jy0} * #{i: Z_i >= Z_j + Lambda}
        Gamma_n = (T1 - T2) / (n*(n-1))

    Parameters
    ----------
    d_iy, d_jy0 : (n,) float arrays — indicator weights.
    Z : (n,) float — index values.
    Z_sorted : (n,) float — pre-sorted Z.
    candidates : (K,) float — Lambda values to evaluate.

    Returns
    -------
    obj : (K,) float — objective values (unnormalised).
    """
    n = len(Z)

    # Fully vectorised: searchsorted preserves the (K, n) shape of the threshold
    # matrix, so no ravel/reshape is needed.
    # thresh1[k, i] = Z[i] - candidates[k]
    # count1[k, i] = #{j : Z_sorted[j] <= thresh1[k,i]}
    thresh1 = Z[np.newaxis, :] - candidates[:, np.newaxis]
    count1 = np.searchsorted(Z_sorted, thresh1, side="right").astype(np.float64)

    # Self-correction: j=i included iff Z_i <= Z_i - lam iff lam <= 0
    neg_mask = candidates <= 0  # (K,)
    count1[neg_mask, :] -= 1.0

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        t1 = count1 @ d_iy  # (K,)

    thresh2 = Z[np.newaxis, :] + candidates[:, np.newaxis]
    count2 = (n - np.searchsorted(Z_sorted, thresh2, side="left")).astype(np.float64)
    count2[neg_mask, :] -= 1.0

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        t2 = count2 @ d_jy0  # (K,)

    return t1 - t2


def _generate_candidates(
    Z: NDArray,
    lambda_min: float,
    lambda_max: float,
    max_candidates: int = 800,
) -> NDArray:
    """
    Generate candidate Lambda values from pairwise Z differences.

    For small n, use all pairwise differences Z_i - Z_j within range.
    For large n, use a stratified subsample + grid.
    """
    n = len(Z)

    if n <= 200:
        # All pairwise differences
        diffs = (Z[:, None] - Z[None, :]).ravel()
        candidates = diffs[(diffs >= lambda_min) & (diffs <= lambda_max)]
    else:
        # Subsample: take a random subset of Z differences + a uniform grid.
        # The jump points of Gamma_n are at Z_i - Z_j, so we sample from those.
        Z_s = np.sort(Z)
        n_grid = max_candidates // 2
        # Stratified sample: differences between evenly-spaced Z quantiles
        idx = np.linspace(0, n - 1, min(n, int(np.sqrt(max_candidates * 2)))).astype(int)
        Z_sub = Z_s[idx]
        diffs = (Z_sub[:, None] - Z_sub[None, :]).ravel()
        diffs = diffs[(diffs >= lambda_min) & (diffs <= lambda_max)]

        # Add a uniform grid to fill gaps
        grid = np.linspace(lambda_min, lambda_max, n_grid)
        candidates = np.concatenate([diffs, grid])

    candidates = np.unique(candidates)
    # Cap the number
    if len(candidates) > max_candidates:
        idx = np.linspace(0, len(candidates) - 1, max_candidates).astype(int)
        candidates = candidates[idx]

    return candidates

def _estimate_lambda_rank(
    y: float,
    y0: float,
    Y: NDArray,
    Z: NDArray,
    Z_sorted: NDArray,
    lambda_min: float,
    lambda_max: float,
    max_candidates: int = 800,
) -> float:
    """
    Estimate Lambda_0(y) by maximising Gamma_n over [lambda_min, lambda_max].

    Uses pre-sorted Z for efficiency.
    """
    d_iy = (Y >= y).astype(np.float64)
    d_jy0 = (Y >= y0).astype(np.float64)

    if d_iy.sum() == 0 or d_jy0.sum() == 0:
        return 0.0

    candidates = _generate_candidates(Z, lambda_min, lambda_max, max_candidates)
    if len(candidates) == 0:
        return 0.0

    obj = _eval_objective_at_candidates(d_iy, d_jy0, Z, Z_sorted, candidates)

    # Handle NaN (can occur with extreme values)
    valid = np.isfinite(obj)
    if not valid.any():
        return 0.0
    obj = np.where(valid, obj, -np.inf)

    # Convention: rightmost maximiser for monotonicity
    max_val = obj.max()
    best_idx = np.where(obj == max_val)[0][-1]
    return candidates[best_idx]


# ---------------------------------------------------------------------------
# Segmentation scheme
# ---------------------------------------------------------------------------

def _build_segmentation(
    y0: float,
    y_low: float,
    y_high: float,
    Y: NDArray,
    Z: NDArray,
    Z_sorted: NDArray,
    alpha: float,
    estimate_fn,
) -> Tuple[NDArray, dict]:
    """
    Build adaptive segmentation points following Chen (2002), Section 4.

    Reference points t_j are chosen so that Lambda_0(t_j) - Lambda_0(t_{j-1})
    is approximately alpha = alpha_factor * z_d, where z_d is the SD of Z_i - Z_j.
    """
    Y_in_range = Y[(Y >= y_low) & (Y <= y_high)]
    if len(Y_in_range) < 10:
        return np.array([y_low, y_high]), {}

    # Use ~100 quantile-spaced grid points for speed
    n_grid = 100

    # Forward segmentation (y0 -> y_high)
    seg_above = [y0]
    if np.any(Y_in_range > y0):
        grid_above = np.percentile(Y_in_range[Y_in_range > y0], np.linspace(0, 100, n_grid))
        current_ref = y0
        for t in grid_above:
            if t <= current_ref:
                continue
            lam_est = estimate_fn(t, current_ref, -alpha, 2 * alpha)
            if abs(lam_est) >= alpha:
                seg_above.append(t)
                current_ref = t
    if seg_above[-1] < y_high:
        seg_above.append(y_high)

    # Backward segmentation (y0 -> y_low)
    seg_below = [y0]
    if np.any(Y_in_range < y0):
        grid_below = np.percentile(Y_in_range[Y_in_range < y0], np.linspace(100, 0, n_grid))
        current_ref = y0
        for t in grid_below:
            if t >= current_ref:
                continue
            lam_est = estimate_fn(t, current_ref, -2 * alpha, alpha)
            if abs(lam_est) >= alpha:
                seg_below.append(t)
                current_ref = t
    if seg_below[-1] > y_low:
        seg_below.append(y_low)

    all_seg = np.array(sorted(set(seg_above + seg_below)))

    # Estimate Lambda differences between consecutive segmentation points
    seg_diffs = {}
    for j in range(1, len(all_seg)):
        t_prev = all_seg[j - 1]
        t_curr = all_seg[j]
        diff = estimate_fn(t_curr, t_prev, -2 * alpha, 2 * alpha)
        seg_diffs[(j - 1, j)] = diff

    return all_seg, seg_diffs


def _evaluate_segmented(
    eval_points: NDArray,
    all_seg: NDArray,
    seg_diffs: dict,
    y0: float,
    estimate_fn,
    alpha: float,
) -> NDArray:
    """
    Evaluate Lambda_hat at arbitrary points using the segmented estimates.

    Lambda_hat(y) = sum of segment diffs from y0 to the segment + local estimate.
    """
    y0_idx = np.searchsorted(all_seg, y0, side="left")
    if y0_idx >= len(all_seg):
        y0_idx = len(all_seg) - 1

    lambda_hat = np.zeros(len(eval_points))

    for k, y in enumerate(eval_points):
        seg_idx = np.searchsorted(all_seg, y, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(all_seg) - 2)

        t_prev = all_seg[seg_idx]

        # Local estimate within the subinterval
        local_lam = estimate_fn(y, t_prev, -2 * alpha, 2 * alpha)

        # Accumulate diffs from y0's position
        cumulative = 0.0
        if seg_idx >= y0_idx:
            for j in range(y0_idx, seg_idx):
                cumulative += seg_diffs.get((j, j + 1), 0.0)
        else:
            for j in range(y0_idx - 1, seg_idx - 1, -1):
                cumulative -= seg_diffs.get((j, j + 1), 0.0)

        lambda_hat[k] = cumulative + local_lam

    return lambda_hat


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class ChenRankEstimator:
    """
    Rank estimator for the transformation function Lambda_0.

    Parameters
    ----------
    y0 : float
        Location normalisation point: Lambda_0(y0) = 0.
    estimation_interval : tuple (y_low, y_high)
        Interval [y2, y1] over which to estimate Lambda_0.
    alpha_factor : float
        Multiplier for the segmentation threshold. Default 0.75.
    max_candidates : int
        Maximum number of Lambda candidate values per evaluation. Default 800.
    """

    def __init__(
        self,
        y0: float,
        estimation_interval: Tuple[float, float],
        alpha_factor: float = 0.75,
        max_candidates: int = 800,
    ):
        self.y0 = y0
        self.y_low, self.y_high = estimation_interval
        self.alpha_factor = alpha_factor
        self.max_candidates = max_candidates

        self.eval_points_: Optional[NDArray] = None
        self.lambda_hat_: Optional[NDArray] = None
        self.beta_hat_: Optional[NDArray] = None

    def fit(
        self,
        Y: NDArray,
        X: NDArray,
        beta: NDArray,
        eval_points: Optional[NDArray] = None,
        n_eval: int = 50,
        use_segmentation: bool = True,
    ) -> "ChenRankEstimator":
        """
        Estimate the transformation function.

        Parameters
        ----------
        Y : (n,) array
            Observed response.
        X : (n, q) array
            Covariates.
        beta : (q,) array
            First-step consistent estimator of beta.
        eval_points : array, optional
            Points at which to evaluate Lambda_hat.
        n_eval : int
            Number of evaluation points if eval_points is None.
        use_segmentation : bool
            Whether to use the segmentation scheme from Section 4.
        """
        self.beta_hat_ = beta

        # Ensure finite values
        Y = np.asarray(Y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Z = X @ beta  # spurious BLAS warning suppressed (Apple Accelerate)

        # Check for non-finite values
        finite_mask = np.isfinite(Y) & np.isfinite(Z)
        if not finite_mask.all():
            Y = Y[finite_mask]
            Z = Z[finite_mask]

        Z_sorted = np.sort(Z)

        if eval_points is None:
            eval_points = np.linspace(self.y_low, self.y_high, n_eval)
        self.eval_points_ = eval_points

        z_d = np.sqrt(2.0) * np.std(Z, ddof=1)
        alpha = self.alpha_factor * z_d

        mc = self.max_candidates

        self._fit(Y, Z, Z_sorted, eval_points, use_segmentation, alpha, mc)

        return self

    def _fit(self, Y, Z, Z_sorted, eval_points, use_segmentation, alpha, mc):
        y0 = self.y0

        def estimate_fn(y, ref, lmin, lmax):
            return _estimate_lambda_rank(y, ref, Y, Z, Z_sorted, lmin, lmax, mc)

        if not use_segmentation:
            lrange = self.y_high - self.y_low
            self.lambda_hat_ = np.array([
                estimate_fn(y, y0, -lrange, lrange) for y in eval_points
            ])
        else:
            all_seg, seg_diffs = _build_segmentation(
                y0, self.y_low, self.y_high, Y, Z, Z_sorted, alpha, estimate_fn,
            )
            self.lambda_hat_ = _evaluate_segmented(
                eval_points, all_seg, seg_diffs, y0, estimate_fn, alpha,
            )

    def predict(self, y: NDArray) -> NDArray:
        """Interpolate Lambda_hat at new points."""
        if self.eval_points_ is None or self.lambda_hat_ is None:
            raise RuntimeError("Call fit() first.")
        return np.interp(y, self.eval_points_, self.lambda_hat_)
