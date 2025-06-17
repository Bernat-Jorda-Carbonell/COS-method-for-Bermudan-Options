# cos_utils.py

import numpy as np

def build_cosine_matrix(N: int) -> np.ndarray:
    """
    Build the NxN cosine matrix for COS expansions:
        cos_mat[j, k] = cos(pi * k * (j + 0.5) / N)
    Parameters
    ----------
    N : int
        Number of COS terms / grid points.
    Returns
    -------
    cos_mat : (N, N) np.ndarray
        Cosine matrix.
    """
    j = np.arange(N)
    k = np.arange(N)
    cos_mat = np.cos(np.pi * np.outer(j + 0.5, k) / N)
    return cos_mat


def fft_convolution(V: np.ndarray, phi: np.ndarray, N: int) -> np.ndarray:
    """
    Compute continuation values by FFT-based convolution.
    Given COS coefficients V (length N) and characteristic factors phi (length N),
    returns cont_vals[j] = sum_{k=0..N-1} V[k] * phi[k] * cos(pi * k * (j+0.5) / N)
    for j = 0..N-1, using O(N log N) operations.

    Parameters
    ----------
    V : (N,) np.ndarray
        COS coefficients at time t_{m+1}.
    phi : (N,) np.ndarray
        Characteristic function values at frequencies k*pi/(b-a).
    N : int
        Number of COS terms.

    Returns
    -------
    cont_vals : (N,) np.ndarray
        Continuation values on the COS grid.
    """
    # Form w = V * phi
    w = V * phi

    # Build m_full array of length 2N: m_full[0] = 1j*pi, m_full[j] = ((-1)^j - 1) / j for j>=1
    m_full = np.empty(2 * N, dtype=complex)
    m_full[0] = 1j * np.pi
    jidx = np.arange(1, 2 * N)
    m_full[1:] = ((-1) ** jidx - 1) / jidx

    # Build ms (length 2N) for Toeplitz part
    ms = np.zeros(2 * N, dtype=complex)
    ms[0] = m_full[0]
    for k in range(1, N):
        ms[k] = m_full[2 * N - k]
    ms[N] = 0
    for k in range(1, N):
        ms[N + k] = m_full[N + k]

    # Build mc (length 2N) for Hankel part
    mc = m_full[::-1]

    # FFT of ms and mc
    Ms_hat = np.fft.fft(ms)
    Mc_hat = np.fft.fft(mc)

    # Sign vector for Hankel embedding
    sgn = np.ones(2 * N)
    sgn[1::2] = -1

    # Zero-pad w to length 2N
    w_pad = np.concatenate([w, np.zeros(N, dtype=w.dtype)])
    W_hat = np.fft.fft(w_pad)

    # Toeplitz part: T w = IFFT(Ms_hat * W_hat)[0:N]
    Tw = np.fft.ifft(Ms_hat * W_hat)[:N].real

    # Hankel part: H w = reverse(IFFT(Mc_hat * (sgn * W_hat))[0:N])
    Hw = np.fft.ifft(Mc_hat * (sgn * W_hat))[:N].real[::-1]

    # Combine and divide by pi
    cont_vals = (Tw + Hw) / np.pi
    return cont_vals


def newton_solve_boundary(x_grid: np.ndarray,
                          cont_vals: np.ndarray,
                          payoff_vals: np.ndarray,
                          cont_deriv: np.ndarray,
                          payoff_deriv: np.ndarray,
                          x0_guess: float,
                          tol: float = 1e-8,
                          maxiter: int = 30) -> float:
    """
    Find the early-exercise boundary x* such that cont(x*) = payoff(x*) using Newton's method
    with a fallback to bisection if needed.

    Parameters
    ----------
    x_grid : (N,) np.ndarray
        COS grid points (log-price values).
    cont_vals : (N,) np.ndarray
        Continuation values at each x_grid.
    payoff_vals : (N,) np.ndarray
        Payoff values at each x_grid.
    cont_deriv : (N,) np.ndarray
        Derivative of continuation values w.r.t. x at each x_grid.
    payoff_deriv : (N,) np.ndarray
        Derivative of payoff w.r.t. x at each x_grid.
    x0_guess : float
        Initial guess for x*.
    tol : float, optional
        Tolerance for Newton's method (default 1e-8).
    maxiter : int, optional
        Maximum Newton iterations (default 30).

    Returns
    -------
    x_star : float
        The computed boundary in [x_grid[0], x_grid[-1]].
    """
    # Linear interpolators for values and derivatives
    cont_interp = lambda x: np.interp(x, x_grid, cont_vals)
    payoff_interp = lambda x: np.interp(x, x_grid, payoff_vals)
    contd_interp = lambda x: np.interp(x, x_grid, cont_deriv)
    payd_interp = lambda x: np.interp(x, x_grid, payoff_deriv)

    a, b = x_grid[0], x_grid[-1]
    x = x0_guess

    for _ in range(maxiter):
        Cx = cont_interp(x)
        Px = payoff_interp(x)
        f = Cx - Px
        if abs(f) < tol:
            return x
        Cpx = contd_interp(x)
        Ppx = payd_interp(x)
        df = Cpx - Ppx
        if df == 0:
            break
        x_new = x - f / df
        if x_new < a or x_new > b:
            break
        if abs(x_new - x) < tol:
            return x_new
        x = x_new

    # Bisection fallback
    fa = cont_interp(a) - payoff_interp(a)
    fb = cont_interp(b) - payoff_interp(b)
    if fa * fb > 0:
        # No sign change: pick endpoint where cont >= payoff, else the other
        return b if cont_interp(b) >= payoff_interp(b) else a

    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = cont_interp(mid) - payoff_interp(mid)
        if abs(fm) < tol:
            return mid
        if fa * fm <= 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return 0.5 * (lo + hi)
