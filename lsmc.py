# lsmc.py

import numpy as np
from scipy.special import gamma as _gamma, gammaincc
from scipy.stats import poisson, expon
from tqdm import tqdm

def price_american_lsmc_bs(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    M_steps: int,
    M_sims: int,
    basis_degree: int = 3,
    seed: int = None
) -> float:
    """
    Price an American put under Black–Scholes by Longstaff–Schwartz (LSMC).
    
    Parameters
    ----------
    S0           : float
        Initial spot price.
    K            : float
        Strike price.
    r            : float
        Risk‐free rate.
    sigma        : float
        Volatility.
    T            : float
        Time to maturity.
    M_steps      : int
        Number of exercise dates (including t=0 and t=T).  So dt = T/(M_steps-1).
    M_sims       : int
        Number of Monte Carlo paths to simulate.
    basis_degree : int
        Degree of polynomial basis to use in regression (e.g. 3 for up-to-cubic).
    seed         : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    estimated_price : float
        LSMC estimate of the American put price.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Time grid
    dt = T / (M_steps - 1)
    times = np.linspace(0, T, M_steps)

    # 2) Simulate M_sims paths of S under BS, on the M_steps time grid
    #    Use exact GBM discretization: S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    lnS = np.zeros((M_sims, M_steps))
    lnS[:, 0] = np.log(S0)
    drift = (r - 0.5 * sigma**2) * dt
    vol_sqrt = sigma * np.sqrt(dt)
    for t in range(1, M_steps):
        Z = np.random.randn(M_sims)
        lnS[:, t] = lnS[:, t-1] + drift + vol_sqrt * Z

    S = np.exp(lnS)  # (M_sims x M_steps) array of simulated spot prices

    # 3) Payoff matrix: immediate exercise value at each time step
    #    For an American put: payoff = max(K - S, 0)
    payoff = np.maximum(K - S, 0)

    # 4) Discount factor per step
    discount_factor = np.exp(-r * dt)

    # 5) Initialize the cash flows at maturity: at t=T, the payout is payoff[:, -1].
    #    We will work backwards from t = M_steps-2 down to t=1 (t=0 is trivial).
    CF = payoff[:, -1].copy()  # shape (M_sims,)

    # 6) Backward induction
    #    At each time index t=M_steps-2, M_steps-3, ..., 1:
    #       (a) Identify which paths are in-the-money (payoff > 0)
    #       (b) Regress the discounted CF (from the next step) onto basis functions of S[:, t]
    #       (c) Compare immediate payoff[:, t] with the regression estimate; decide to exercise or continue.
    #    At time t=0 we simply exercise if that is optimal, but in practice for a put it's rarely optimal to exercise at t=0
    #    when using these parameters.

    for t in reversed(range(1, M_steps - 1)):
        # (a) All paths that are in‐the‐money at time t
        itm_mask = payoff[:, t] > 0  # boolean mask, shape (M_sims,)
        if not np.any(itm_mask):
            # If no path is in‐the‐money, just carry CF backward by discounting
            CF = CF * discount_factor
            continue

        S_itm = S[itm_mask, t]    # (N_itm,) 
        Y_itm = CF[itm_mask] * discount_factor  # discounted continuation values from step t+1

        # (b) Build polynomial basis of degree <= basis_degree in S_itm
        #     We'll do a simple Vandermonde: [1, S, S^2, ..., S^{basis_degree}].
        #     Then solve least-squares for regression coefficients, so that
        #     E[Y_itm | S_itm] ≈ sum_{k=0}^d a_k * (S_itm)^k.
        X = np.vander(S_itm, N=basis_degree+1, increasing=True)  # shape (N_itm, basis_degree+1)
        # Solve: minimize || X @ a - Y_itm ||^2  → a = (X^T X)^{-1} X^T Y_itm
        # We can use numpy.linalg.lstsq for stability:
        a, *_ = np.linalg.lstsq(X, Y_itm, rcond=None)  # a has length (basis_degree+1,)

        # (c) Evaluate the regression estimate on all in‐the‐money paths:
        cont_est = X @ a  # shape (N_itm,)
        # Now for each in‐the‐money path i, compare payoff[i,t] vs cont_est[i_itm_index].
        exercise_mask = np.zeros_like(itm_mask)
        exercise_paths = payoff[itm_mask, t] >= cont_est   # boolean array of length N_itm
        exercise_mask[itm_mask] = exercise_paths

        # For paths that exercise at t: set CF = payoff[:, t]
        # Otherwise, path continues, so CF = CF * discount_factor
        # We need a new CF_next array:
        CF_next = np.zeros_like(CF)
        # i) exercised paths get immediate payoff:
        CF_next[exercise_mask] = payoff[exercise_mask, t]
        # ii) continuation paths get discounted carry‐on:
        cont_indices = ~exercise_mask
        CF_next[cont_indices] = CF[cont_indices] * discount_factor

        CF = CF_next

    # 7) At t=0, the option value is simply the average of CF discounted from t=1 to t=0 again.
    #    But since we always discount one step in the loop, CF already represents the value at t=0.
    estimated_price = CF.mean()  # This is already at t=0

    return estimated_price



def sample_large_jump_batch(beta, lam, eps, size, rng, Y):
    """
    Draw exactly `size` samples from the “density”
    by doing batched accept/reject. 
    """
    out = np.empty(size, dtype=float)
    c = (eps ** (-1.0 - Y)) / (lam / (beta ** Y))
    got = 0
    while got < size:
        batch = eps + rng.exponential(scale=1.0 / beta, size=(size - got) * 4)
        numer = (batch ** (-1.0 - Y)) * np.exp(-beta * batch)
        accept_prob = (numer / lam) / c
        u = rng.random(len(batch))
        accepted = batch[u < accept_prob]
        take = min(len(accepted), size - got)
        out[got : got + take] = accepted[:take]
        got += take
    return out

def simulate_cgmy_paths(
    S0: float,
    r: float,
    C: float,
    G: float,
    M_jump: float,
    Y: float,
    T: float,
    M_steps: int,
    M_sims: int,
    epsilon: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """
    Simulate CGMY paths under Q at M_steps equally spaced times in [0,T].
    Uses a small-jump / large-jump decomposition with threshold epsilon.
    Returns array of shape (M_steps, M_sims).
    """
    rng = np.random.default_rng(seed)
    dt = T / (M_steps - 1)

    # 1) small‐jump variance
    small_var = (epsilon ** (2.0 - Y)) / (2.0 - Y)
    var_small = C * small_var + C * small_var

    # 2) large‐jump intensity (both pos and neg symmetric)
    lam_pos = C * (epsilon ** (-Y)) / Y
    lam_neg = lam_pos
    lam_pos_dt = lam_pos * dt
    lam_neg_dt = lam_neg * dt

    # 3) drift correction: ideally include E[e^(large jumps)-1], but at minimum:
    drift_corr = r - 0.5 * var_small

   
    # 4) allocate output array as (time_index, path_index)
    S = np.empty((M_steps, M_sims), dtype=float)
    S[0, :] = S0
 # 5) loop over time steps
    for j in range(1, M_steps):
        # 5a) small-jump Gaussian increment
        Z = rng.standard_normal(M_sims)
        X_small = np.sqrt(var_small * dt) * Z

        # 5b) Poisson counts of large jumps
        Np = rng.poisson(lam_pos_dt, size=M_sims)
        Nn = rng.poisson(lam_neg_dt, size=M_sims)

        # 5c) vectorized large-jump contributions
        X_large = np.zeros(M_sims, dtype=float)

        # Positive jumps
        pos_idx = np.nonzero(Np > 0)[0]
        if pos_idx.size > 0:
            uniq_k, counts_k = np.unique(Np[pos_idx], return_counts=True)
            for k, num_paths in zip(uniq_k, counts_k):
                idx_k = pos_idx[Np[pos_idx] == k]
                ups = sample_large_jump_batch(G, lam_pos, epsilon,
                                              size=num_paths * k,
                                              rng=rng, Y=Y)
                ups = ups.reshape(num_paths, k)
                X_large[idx_k] += ups.sum(axis=1)

        # Negative jumps
        neg_idx = np.nonzero(Nn > 0)[0]
        if neg_idx.size > 0:
            uniq_k2, counts_k2 = np.unique(Nn[neg_idx], return_counts=True)
            for k2, num_paths2 in zip(uniq_k2, counts_k2):
                idx_k2 = neg_idx[Nn[neg_idx] == k2]
                downs = sample_large_jump_batch(M_jump, lam_neg, epsilon,
                                                size=num_paths2 * k2,
                                                rng=rng, Y=Y)
                downs = downs.reshape(num_paths2, k2)
                X_large[idx_k2] -= downs.sum(axis=1)

        # 5d) total log increment and update spot
        dX = drift_corr * dt + X_small + X_large
        S[j, :] = S[j - 1, :] * np.exp(dX)

    return S

def price_american_lsmc_cgmy(
    S0: float,
    K: float,
    r: float,
    C: float,
    G: float,
    M_jump: float,
    Y: float,
    T: float,
    M_steps: int,
    M_sims: int,
    basis_degree: int = 3,
    epsilon: float = 0.1,
    seed: int = None,
    debug: bool = False
) -> float:
    """
    Price an American put under CGMY by Longstaff‐Schwartz. Returns price at t=0.
    If debug=True, also returns regression_coefs and itm_counts.
    """
    # 1) simulate CGMY paths (shape: M_steps × M_sims)
    S = simulate_cgmy_paths(
        S0=S0, r=r, C=C, G=G, M_jump=M_jump, Y=Y,
        T=T, M_steps=M_steps, M_sims=M_sims, epsilon=epsilon, seed=seed
    )
    payoff = np.maximum(K - S, 0)   # shape (M_steps, M_sims)
    dt = T / (M_steps - 1)
    disc = np.exp(-r * dt)

    # 2) cash flows at maturity
    CF = payoff[-1, :].copy()       # CF[i] is cash‐flow for path i at time T

    # 3) prepare storage if debug=True
    if debug:
        regression_coefs = {}
        itm_counts = {}

    # 4) backward induction
    # prepare a "powers" array of shape (basis_degree+1, M_sims)
    powers = np.empty((basis_degree + 1, M_sims), dtype=float)

    for j in reversed(range(1, M_steps - 1)):
        itm_idx = np.nonzero(payoff[j, :] > 0)[0]
        if debug:
            itm_counts[j] = itm_idx.size

        if itm_idx.size == 0:
            CF = CF * disc
            continue


        # 4a) build design matrix for in‐the‐money paths
        # powers[0, :] = 1, powers[1, :] = S[j, :], powers[2, :] = S[j, :]^2, …
        powers[0, :] = 1.0
        powers[1, :] = S[j, :]
        for d in range(2, basis_degree + 1):
            powers[d, :] = powers[d - 1, :] * S[j, :]

        X_itm = powers[:, itm_idx].T          # shape = (num_itm, basis_degree+1)
        Y_itm = CF[itm_idx] * disc            # continuation values

         # 4b) regression
        a, *_ = np.linalg.lstsq(X_itm, Y_itm, rcond=None)
        cont_est = X_itm @ a

        if debug:
            regression_coefs[j] = a.copy()

        # 4c) compare exercise vs. continuation
        intrinsic = payoff[j, itm_idx]
        exer_mask = intrinsic >= cont_est   # boolean vector of length = len(itm_idx)

        # 4d) form new CF: if exercise, CF[t=j] = payoff; else CF[t=j+1] discounted
        CF_next = np.empty_like(CF)
        CF_next[:] = CF * disc              # first assume everyone continued
        # those who exercise:
        exercise_idx = itm_idx[exer_mask]
        CF_next[exercise_idx] = payoff[j, exercise_idx]
        CF = CF_next

    v0 = CF.mean()
    if debug:
        return v0, regression_coefs, itm_counts
    return v0