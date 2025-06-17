
import numpy as np

def bs_charfn(u: np.ndarray, dt: float, r: float, sigma: float) -> np.ndarray:
    """
    Black–Scholes CF for ln(S_t/S_0) over time dt:
      phi(u; dt) = exp(i u (r - 0.5 sigma^2) dt - 0.5 sigma^2 u^2 dt)
    """
    return np.exp(1j * u * (r - 0.5 * sigma**2) * dt
                  - 0.5 * sigma**2 * u**2 * dt)

def cgmy_charfn(u: np.ndarray, dt: float, r: float,
                C: float, G: float, M: float, Y: float) -> np.ndarray:
    """
    CGMY CF for ln(S_t/S_0) over time dt (under risk-neutral measure):

    For CGMY with parameters (C, G, M, Y), zero diffusion and zero dividends (q=0),
    the risk‐neutral characteristic exponent is
      psi(u) = r*i*u
               + C * Gamma(-Y) * [ (M - i u)^Y - M^Y  +  (G + i u)^Y - G^Y ].
    Then phi(u; dt) = exp(dt * psi(u)).

    Note: Gamma(-Y) uses scipy.special.gamma if Y is not integer.
    """
    from scipy.special import gamma

    # (i) compute the pure‐jump exponent
    
    iu = 1j * u
    coeff = C * gamma(-Y)
    term1 = (M - iu)**Y - (M**Y)
    term2 = (G + iu)**Y - (G**Y)
    psi_u = 1j * u * r + coeff * (term1 + term2)

    # (ii) return the CF
    return np.exp(dt * psi_u)

import numpy as np

def nig_charfn(u: np.ndarray,
               dt: float,
               r: float,
               alpha: float,
               beta: float,
               delta: float) -> np.ndarray:
    """
    Risk‐neutral characteristic function of ln(S_t/S_0) under a NIG model
    with parameters (alpha, beta, delta) and zero Gaussian volatility.

    We choose the drift so that E[e^{X_t}] = e^{r t}.  Equivalently,
    define
        omega = delta * ( sqrt(alpha^2 - beta^2)
                       - sqrt(alpha^2 - (beta + 1)^2 ) )
    and the Lévy exponent
        psi(u) = i*u*(r + omega)
                 + delta * ( sqrt(alpha^2 - beta^2)
                             - sqrt(alpha^2 - (beta + i*u)^2 ) ).
    Then φ(u; dt) = exp( dt * psi(u) ).

    Parameters
    ----------
    u     : np.ndarray
        Frequencies (array of real values).
    dt    : float
        Time increment.
    r     : float
        Risk‐free rate.
    alpha : float
        NIG alpha parameter (alpha > 0).
    beta  : float
        NIG beta parameter (|beta| < alpha).
    delta : float
        NIG delta parameter (delta > 0).

    Returns
    -------
    phi : np.ndarray
        Complex‐valued characteristic function φ(u; dt) evaluated at each u.
    """
    # Precompute sqrt(alpha^2 - beta^2), which is real when |beta| < alpha.
    sqrt_ab = np.sqrt(alpha**2 - beta**2)

    # Compute omega = delta*(sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + 1)^2))
    sqrt_b1 = np.sqrt(alpha**2 - (beta + 1.0)**2)
    omega = delta * (sqrt_ab - sqrt_b1)

    # i*u
    iu = 1j * u

    # Compute the jump‐part exponent: δ * ( sqrt(alpha^2 - beta^2)
    #                                    - sqrt(alpha^2 - (beta + i u)^2 ) )
    sqrt_biu = np.sqrt(alpha**2 - (beta + iu)**2)
    psi_jump = delta * (sqrt_ab - sqrt_biu)

    # Total exponent: i u (r + omega) + psi_jump
    psi = (1j * u * (r + omega)) + psi_jump

    return np.exp(psi * dt)