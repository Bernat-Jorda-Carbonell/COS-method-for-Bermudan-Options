# cos_bs.py
import numpy as np

def bs_cf(u: np.ndarray, t: float, r: float, sigma: float) -> np.ndarray:
    """
    Black-Scholes characteristic function for ln(S_t / S_0).

    phi(u; t) = E[exp(i * u * ln(S_t / S_0))]
              = exp(i * u * (r - 0.5 * sigma^2) * t - 0.5 * sigma^2 * u^2 * t)

    Parameters
    ----------
    u     : array_like
        Frequencies (can be a vector).
    t     : float
        Time horizon.
    r     : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    phi : np.ndarray
        Complex-valued characteristic function evaluated at each u.
    """
    return np.exp(1j * u * (r - 0.5 * sigma**2) * t
                  - 0.5 * sigma**2 * u**2 * t)


def cumulant_range(r: float, sigma: float,
                   S0: float, K: float, T: float,
                   L: float = 8.0) -> (float, float):
    """
    Compute [a, b] for the COS integration domain of x = ln(S / K):
      [a, b] = (c1 + x0) +/- L * sqrt(c2 + sqrt(c4))
    Under Black-Scholes: c1 = (r - 0.5 * sigma^2) * T, c2 = sigma^2 * T, c4 = 0.

    Parameters
    ----------
    r     : float
        Risk-free rate.
    sigma : float
        Volatility.
    S0    : float
        Initial spot price.
    K     : float
        Strike price.
    T     : float
        Time to maturity.
    L     : float, optional
        Width parameter (default 8).

    Returns
    -------
    a, b : floats
        Truncation bounds for x = ln(S / K) in the COS expansion.
    """
    x0 = np.log(S0 / K)
    c1 = (r - 0.5 * sigma**2) * T
    c2 = sigma**2 * T
    c4 = 0.0
    delta = L * np.sqrt(c2 + np.sqrt(c4))
    a = x0 + c1 - delta
    b = x0 + c1 + delta
    return a, b


# Example quick test
if __name__ == "__main__":
    u = np.array([0.1, 1.0, 2.0])
    print("phi(u):", bs_cf(u, t=1.0, r=0.05, sigma=0.2))
    print("Range [a, b]:", cumulant_range(0.05, 0.2, S0=100, K=100, T=1.0))
