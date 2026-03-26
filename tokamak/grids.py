"""
Grid construction and coordinate transforms for toroidal geometry.
"""

import numpy as np
from tokamak.config import TokamakConfig


def build_grids(cfg: TokamakConfig):
    """Build (ρ, θ) grids for poloidal cross-sections."""
    rho = np.linspace(0, 1, cfg.Nr)
    theta = np.linspace(0, 2 * np.pi, cfg.Ntheta)
    dr = rho[1] - rho[0]
    dtheta = theta[1] - theta[0]
    RHO, THETA = np.meshgrid(rho, theta, indexing='ij')
    return rho, theta, dr, dtheta, RHO, THETA


def poloidal_to_RZ(rho_grid, theta_grid, R0, a, kappa, delta_tri,
                   shafranov_shift=None):
    """Convert (ρ, θ) → (R, Z) with D-shaped shaping + Shafranov shift."""
    if shafranov_shift is None:
        # Default Shafranov shift ~ β_p a/2 (1 - ρ²), with β_p ≈ 0.3
        shafranov_shift = 0.15 * a * (1.0 - rho_grid**2)
    R = (R0 + shafranov_shift
         + a * rho_grid * np.cos(theta_grid + delta_tri * np.sin(theta_grid)))
    Z = a * kappa * rho_grid * np.sin(theta_grid)
    return R, Z


def toroidal_to_XYZ(R, Z, phi):
    """Convert (R, Z, φ) → (X, Y, Z_cart) in lab Cartesian frame."""
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    Z_cart = Z
    return X, Y, Z_cart


def build_3d_mesh(cfg: TokamakConfig):
    """
    Build full 3D toroidal mesh (ρ, θ, φ) → (X, Y, Z).
    Returns Cartesian coordinates and the grids.
    """
    rho = np.linspace(0, 1, cfg.Nr)
    theta = np.linspace(0, 2 * np.pi, cfg.Ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, cfg.Nphi, endpoint=False)

    # 3D meshgrid: (Nr, Ntheta, Nphi)
    RHO, THETA, PHI = np.meshgrid(rho, theta, phi, indexing='ij')

    # Shafranov shift
    delta_S = 0.15 * cfg.a * (1.0 - RHO**2)

    R = (cfg.R0 + delta_S
         + cfg.a * RHO * np.cos(THETA + cfg.delta_tri * np.sin(THETA)))
    Z = cfg.a * cfg.kappa * RHO * np.sin(THETA)

    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    Z_cart = Z  # Z is independent of φ

    return {
        'rho': rho, 'theta': theta, 'phi': phi,
        'RHO': RHO, 'THETA': THETA, 'PHI': PHI,
        'R': R, 'Z': Z, 'X': X, 'Y': Y, 'Z_cart': Z_cart,
    }
