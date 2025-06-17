# cos_bermudan.py

import numpy as np
from cos_bs import bs_cf, cumulant_range
from cos_utils import build_cosine_matrix, fft_convolution, newton_solve_boundary

def price_bermudan_cos(
    S0: float,
    K: float,
    r: float,
    T: float,
    M: int,
    N: int,
    a: float,
    b: float,
    charfn,                # generic characteristic function: charfn(u, dt, r) -> array of length N
    option_type: str = 'put'
) -> (float, list):
    """
    Price a Bermudan option by the COS method under any model
    whose characteristic function is given by `charfn`. Returns (price, x_boundaries).

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    M : int
        Number of exercise dates.
    N : int
        Number of COS expansion terms.
    a : float
        Lower bound of the log-price truncation interval.
    b : float
        Upper bound of the log-price truncation interval.
    charfn : callable
        Model-specific characteristic function of log-returns. Should have signature
            phi_vals = charfn(u_array, dt, r)
        where `u_array` is a length-N vector of frequencies, `dt` is the time step, and `r`
        is the risk-free rate (if needed).
    option_type : str, default 'put'
        'put' or 'call'.

    Returns
    -------
    price : float
        The Bermudan option price at S0.
    x_boundaries : list of float
        Early-exercise boundaries for dates t_1, ..., t_{M-1}. The list has length M+1;
        entries for t_0 and t_M are None.
    """
    # 1) Build x_grid and cosine matrix
    x0 = np.log(S0 / K)
    j = np.arange(N)
    x_grid = a + (j + 0.5) * (b - a) / N
    cos_mat = build_cosine_matrix(N)

    # 2) Compute COS coefficients at maturity (t_M)
    if option_type == 'call':
        payoff_T = np.maximum(np.exp(x_grid) - 1, 0) * K
        payoff_deriv_T = np.where(x_grid >= 0, np.exp(x_grid) * K, 0.0)
    else:  # put
        payoff_T = np.maximum(1 - np.exp(x_grid), 0) * K
        payoff_deriv_T = np.where(x_grid <= 0, -np.exp(x_grid) * K, 0.0)

    V = (2.0 / N) * (cos_mat.T @ payoff_T)
    previous_boundary = 0.0

    # Prepare list of boundaries; index m corresponds to time t_m
    x_boundaries = [None] * (M + 1)
    x_boundaries[M] = None  # no boundary at maturity (t_M)

    # 3) Backward induction over exercise dates
    times = np.linspace(0, T, M + 1)
    delta_times = np.diff(times)[::-1]  # from t_M to t_{M-1}, etc.

    for m_index, delta_t in enumerate(delta_times, start=1):
        # m_index = 1 => stepping from t_M to t_{M-1}

        # 3a) Compute u_k and characteristic function values phi_vals
        k_idx = np.arange(N)
        u_k = k_idx * np.pi / (b - a)
        phi_vals = charfn(u_k, delta_t, r)

        # 3b) Compute continuation values on x_grid via FFT convolution
        raw_cont = fft_convolution(V, phi_vals, N)
        cont_vals = np.exp(-r * delta_t) * raw_cont

        # 3c) Recompute payoff and its derivative on x_grid
        if option_type == 'call':
            payoff_vals = np.maximum(np.exp(x_grid) - 1, 0) * K
            payoff_deriv = np.where(x_grid >= 0, np.exp(x_grid) * K, 0.0)
        else:
            payoff_vals = np.maximum(1 - np.exp(x_grid), 0) * K
            payoff_deriv = np.where(x_grid <= 0, -np.exp(x_grid) * K, 0.0)

        # Compute derivative of continuation values cont_deriv
        cont_deriv = np.zeros(N, dtype=complex)
        for kk in range(N):
            cont_deriv += V[kk] * phi_vals[kk] * u_k[kk] * np.sin(u_k[kk] * (x_grid - a))
        cont_deriv = (-np.exp(-r * delta_t) * cont_deriv).real

        # 3d) Solve for early-exercise boundary x* in [a, b]
        x_star = newton_solve_boundary(
            x_grid,
            cont_vals,
            payoff_vals,
            cont_deriv,
            payoff_deriv,
            x0_guess=previous_boundary,
            tol=1e-8,
            maxiter=30
        )
        # Store boundary at t_{M - m_index}
        t_index = M - m_index
        x_boundaries[t_index] = x_star
        previous_boundary = x_star

        # 3e) Update COS coefficients V at time t_{m_index - 1}
        exercise_vals = np.maximum(payoff_vals, cont_vals)
        V = (2.0 / N) * (cos_mat.T @ exercise_vals)

    # 4) Reconstruct final price at x0 = ln(S0/K)
    cos_x0 = np.cos(k_idx * np.pi * (x0 - a) / (b - a))
    price = np.real(np.sum(V * cos_x0))

    return price, x_boundaries



def price_barrier_cos(
    S0: float,
    K: float,
    r: float,
    T: float,
    M: int,
    N: int,
    a: float,
    b: float,
    charfn,                  # generic CF: charfn(u, dt, r) -> array of length N
    option_type: str = "put",  # "put" or "call"
    barrier_type: str = "up-and-out",  # "up-and-out" or "down-and-out"
    barrier_level: float = 100.0       # absolute barrier H (not log)
) -> float:
    """
    Price a discretely‐monitored barrier option via the COS method.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    M : int
        Number of monitoring dates (including maturity).
    N : int
        Number of COS expansion terms.
    a : float
        Lower bound of the log-price truncation interval.
    b : float
        Upper bound of the log-price truncation interval.
    charfn : callable
        Model-specific characteristic function of log-returns: charfn(u, dt, r) -> length-N array.
    option_type : str, default "put"
        "put" or "call".
    barrier_type : str, default "up-and-out"
        "up-and-out" or "down-and-out".
    barrier_level : float, default 100.0
        Barrier level H (in spot-space). Internally converted to x_H = ln(H/K).

    Returns
    -------
    price : float
        The discretely‐monitored barrier option price.
    """
    # 1) Build x_grid, cos matrix, and log-barrier
    x0 = np.log(S0 / K)
    j = np.arange(N)
    x_grid = a + (j + 0.5) * (b - a) / N
    cos_mat = build_cosine_matrix(N)

    # Convert barrier to log-space
    x_H = np.log(barrier_level / K)

    # 2) Terminal payoff at maturity t_M (assume alive unless barrier is already in-the-money)
    if option_type == "call":
        payoff_T_unfiltered = np.maximum(np.exp(x_grid) - 1, 0) * K
    else:  # "put"
        payoff_T_unfiltered = np.maximum(1 - np.exp(x_grid), 0) * K

    # Determine which grid points are alive at maturity
    if barrier_type == "up-and-out":
        alive_mask = (x_grid < x_H)   # alive only if x_j < x_H
    else:  # "down-and-out"
        alive_mask = (x_grid > x_H)   # alive only if x_j > x_H

    # Zero out payoffs at knocked-out points
    payoff_T = np.where(alive_mask, payoff_T_unfiltered, 0.0)
    V = (2.0 / N) * (cos_mat.T @ payoff_T)

    # 3) Backward induction over monitoring dates m = M-1, M-2, ..., 0
    times = np.linspace(0, T, M + 1)
    dts = np.diff(times)[::-1]  # dt from t_M→t_{M-1}, etc.

    for delta_t in dts:
        # 3a) Compute u_k and CF values
        k_idx = np.arange(N)
        u_k = k_idx * np.pi / (b - a)
        phi_vals = charfn(u_k, delta_t, r)

        # 3b) Continuation values via FFT convolution
        raw_cont = fft_convolution(V, phi_vals, N)
        cont_vals = np.exp(-r * delta_t) * raw_cont

        # 3c) Compute “unfiltered” payoff at this date
        if option_type == "call":
            payoff_unfiltered = np.maximum(np.exp(x_grid) - 1, 0) * K
        else:  # "put"
            payoff_unfiltered = np.maximum(1 - np.exp(x_grid), 0) * K

        # Recompute alive_mask at this date (barrier status does not change with time step)
        if barrier_type == "up-and-out":
            alive_mask = (x_grid < x_H)
        else:  # "down-and-out"
            alive_mask = (x_grid > x_H)

        # 3d) Barrier-filtered exercise payoff: zero if already knocked out
        exercise_vals = np.where(alive_mask, payoff_unfiltered, 0.0)

        # 3e) Recombine: V_j = max(exercise, continuation), but force to zero if knocked out
        Vj = np.maximum(exercise_vals, cont_vals)
        Vj = np.where(alive_mask, Vj, 0.0)

        # 3f) Update COS coefficients for the next step
        V = (2.0 / N) * (cos_mat.T @ Vj)

    # 4) Reconstruct final price at x0 = ln(S0/K)
    cos_x0 = np.cos(k_idx * np.pi * (x0 - a) / (b - a))
    price = np.real(np.sum(V * cos_x0))

    return price

if __name__ == "__main__":
    # Example usage with Black-Scholes characteristic function
    import time
    from cos_bs import bs_cf, cumulant_range


    # 1) Define a BS characteristic-function wrapper that matches the signature
    def bs_charfn(u, dt, r):
        # here sigma must be captured from the outer scope
        return bs_cf(u, dt, r, sigma)

    # 2) Example parameters
    S0, K, r, sigma, T, M, N = 100, 100, 0.05, 0.2, 1.0, 10, 256
    L = 8  # truncation width parameter

    # 3) Compute the truncation range [a, b] using cumulants
    a, b = cumulant_range(r, sigma, S0, K, T, L=L)

    # 4) Price a Bermudan put under BS
    start = time.time()
    price_put, boundaries_put = price_bermudan_cos(
        S0, K, r, T, M, N, a, b, bs_charfn, option_type='put'
    )
    end = time.time()
    print(f"BS Bermudan put price: {price_put:.6f}  (computed in {end - start:.4f} s)")

    # 5) Define a CGMY characteristic-function wrapper as another example
    #    (C, G, M, Y) must be set here
    C, G, Mpar, Y = 1.0, 5.0, 5.0, 1.5
    def cgmy_charfn(u, dt, r):
        # import or define cgmy_cf(u, dt, r, C, G, M, Y) elsewhere
        return cgmy_cf(u, dt, r, C, G, Mpar, Y)

    # 6) Price a Bermudan put under CGMY
    #    (Assumes cgmy_cf and corresponding cumulant_range are available)
    # a_cgmy, b_cgmy = cgmy_cumulant_range(r, C, G, Mpar, Y, S0, K, T, L=L)
    # price_cgmy_put, boundaries_cgmy_put = price_bermudan_cos(
    #     S0, K, r, T, M, N, a_cgmy, b_cgmy, cgmy_charfn, option_type='put'
    # )
    # print(f"CGMY Bermudan put price: {price_cgmy_put:.6f}")
