"""
First-step estimators for the regression coefficient beta.

Both Chen (2002) and Ye & Duan (1997) require a sqrt(n)-consistent
estimator of beta as input. We implement Han's (1987) Maximum Rank
Correlation (MRC) estimator, which maximises

    S_n(b) = (1 / n(n-1)) sum_{i != j} 1{Y_i > Y_j} * 1{X_i b > X_j b}

over b on the unit sphere with |b_1| = 1 (scale normalisation).

Reference:
    Han, A. K. (1987). "Non-parametric analysis of a generalized regression
    model: The maximum rank correlation estimator." Journal of Econometrics,
    35, 303-316.
"""

import numpy as np
from scipy.optimize import minimize
from numpy.typing import NDArray
from typing import Optional


def _mrc_objective(b_free: NDArray, Y: NDArray, X: NDArray, sign_b1: float) -> float:
    """
    Negative of the rank correlation objective (we minimise).

    Parameters
    ----------
    b_free : (q-1,) array
        Free parameters (components 2..q of beta).
    Y : (n,) array
        Dependent variable.
    X : (n, q) array
        Covariates.
    sign_b1 : float
        Sign of beta_1 (+1 or -1). |beta_1| = 1 by normalisation.
    """
    q = X.shape[1]
    b = np.empty(q)
    b[0] = sign_b1
    b[1:] = b_free

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Z = X @ b  # (n,)
    n = len(Y)

    # Vectorised pairwise comparisons
    # Using broadcasting: (n, 1) vs (1, n)
    Y_diff = Y[:, None] - Y[None, :]  # Y_i - Y_j
    Z_diff = Z[:, None] - Z[None, :]  # Z_i - Z_j

    # 1{Y_i > Y_j} * 1{Z_i > Z_j}, exclude diagonal
    concordant = np.sum((Y_diff > 0) & (Z_diff > 0))
    total_pairs = n * (n - 1)

    return -concordant / total_pairs


def han_mrc(
    Y: NDArray,
    X: NDArray,
    sign_b1: float = 1.0,
    b0: Optional[NDArray] = None,
    n_restarts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """
    Han's Maximum Rank Correlation estimator for beta.

    Parameters
    ----------
    Y : (n,) array
        Response variable.
    X : (n, q) array
        Covariate matrix. The first column corresponds to beta_1,
        which is normalised to |beta_1| = 1.
    sign_b1 : float
        Sign of beta_1. Default +1.
    b0 : (q-1,) array, optional
        Initial guess for the free parameters. If None, uses zeros.
    n_restarts : int
        Number of random restarts for the optimiser.
    rng : numpy Generator, optional
        Random number generator for restarts.

    Returns
    -------
    beta_hat : (q,) array
        Estimated coefficient vector with |beta_1| = 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n, q = X.shape

    if q == 1:
        # Only one covariate: beta = (sign_b1,)
        return np.array([sign_b1])

    best_val = np.inf
    best_b = np.zeros(q - 1)

    starts = []
    if b0 is not None:
        starts.append(b0)
    starts.append(np.zeros(q - 1))
    for _ in range(n_restarts - len(starts)):
        starts.append(rng.standard_normal(q - 1))

    for start in starts:
        res = minimize(
            _mrc_objective,
            start,
            args=(Y, X, sign_b1),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_b = res.x

    beta_hat = np.empty(q)
    beta_hat[0] = sign_b1
    beta_hat[1:] = best_b
    return beta_hat
