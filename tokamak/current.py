"""
Current density calculation — Ohmic + Bootstrap.
"""

import numpy as np
from tokamak.config import TokamakConfig
from tokamak.equilibrium import q_profile, B_poloidal


def compute_current_density(rho, n, Ti, Te, cfg: TokamakConfig):
    """
    Compute toroidal current density j_φ [A/m²].

    Components:
        j_ohmic:     from Ampère's law / equilibrium relation
        j_bootstrap: pressure-gradient-driven current (neoclassical)
    """
    Nr = len(rho)
    j_phi = np.zeros(Nr)
    j_bs  = np.zeros(Nr)
    dr = rho[1] - rho[0]

    for i in range(1, Nr - 1):
        q  = q_profile(rho[i], cfg.q0, cfg.qa)
        R  = cfg.R0 + cfg.a * rho[i]
        r  = rho[i] * cfg.a
        Bp = B_poloidal(rho[i], cfg.a, cfg.R0, cfg.B0,
                        cfg.Ip, cfg.mu0, cfg.q0, cfg.qa)

        # Ohmic: j = (1/μ₀) (1/R) d(R Bp)/dr (Ampère)
        # For parabolic current profile: j ∝ (1 - ρ²)
        j0 = 2.0 * cfg.Ip / (np.pi * cfg.a**2)  # Peak current density
        j_phi[i] = j0 * max(1.0 - rho[i]**2, 0.0)

        # Bootstrap current
        eps = cfg.epsilon * rho[i]
        if rho[i] > 0.01 and Bp > 1e-8:
            dp_dr = ((n[i+1] * Ti[i+1] - n[i-1] * Ti[i-1]) * cfg.keV_to_J
                     / (2 * dr * cfg.a))
            f_bs = -np.sqrt(max(eps, 0.001)) * 2.44
            j_bs[i] = f_bs * dp_dr / max(Bp, 1e-8)

    return j_phi, j_bs
