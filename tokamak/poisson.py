"""
Poisson solver for electrostatic potential.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from tokamak.config import TokamakConfig
from tokamak.equilibrium import density_profile, temperature_profile


def solve_poisson_2d(density_2d, rho_arr, theta_arr, dr, dtheta, cfg):
    """
    Solve for electrostatic potential on (ρ, θ) grid.

    Uses quasi-neutrality approximation for the 1D radial profile:
        eΦ ≈ T_e ln(n/n₀)    (ambipolar potential)
    Then adds m=1 Shafranov-shift perturbation for 2D structure.
    """
    Nr, Ntheta = density_2d.shape

    # 1D radial ambipolar potential from quasi-neutrality
    rho_charge = np.mean(density_2d, axis=1)
    phi_1d = np.zeros(Nr)

    rho = np.linspace(0, 1, Nr)
    for i in range(Nr):
        n_loc = density_profile(rho[i], cfg.n0)
        Te_loc = temperature_profile(rho[i], cfg.T0_e)
        # eΦ = Te ln(n/n0)  →  Φ = (Te[keV] × 1e3) × ln(n/n0)
        ratio = max(n_loc / cfg.n0, 1e-6)
        phi_1d[i] = Te_loc * 1e3 * np.log(ratio)  # in Volts

    # Build 2D: constant on flux surfaces + m=1 perturbation
    phi_2d = np.zeros((Nr, Ntheta))
    for i in range(Nr):
        for j in range(Ntheta):
            # m=1 Shafranov-shift contribution
            theta_j = theta_arr[j] if j < len(theta_arr) else 0.0
            phi_2d[i, j] = phi_1d[i] * (1.0 + 0.05 * rho[i] * np.cos(theta_j))

    return phi_2d, phi_1d


def solve_poisson_1d_matrix(density_1d, dr, Nr, cfg):
    """
    Solve 1D radial Poisson equation with tridiagonal matrix.
    (1/r) d/dr(r dΦ/dr) = -ρ/ε₀
    """
    rho_arr = np.linspace(dr, 1.0, Nr)

    diag_main  = np.zeros(Nr)
    diag_upper = np.zeros(Nr - 1)
    diag_lower = np.zeros(Nr - 1)
    rhs        = np.zeros(Nr)

    for i in range(1, Nr - 1):
        r  = rho_arr[i]
        rp = 0.5 * (rho_arr[i] + rho_arr[i+1])
        rm = 0.5 * (rho_arr[i-1] + rho_arr[i])

        diag_lower[i-1] = rm / (r * dr**2)
        diag_main[i]    = -(rp + rm) / (r * dr**2)
        diag_upper[i]   = rp / (r * dr**2)
        rhs[i]          = -density_1d[i] / cfg.eps0

    # BCs
    diag_main[0]  = 1.0
    rhs[0]        = 0.0
    diag_main[-1] = 1.0
    rhs[-1]       = 0.0

    A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc')
    return spsolve(A, rhs)
