"""
Physical and numerical parameters — ITER-like tokamak configuration.
"""

import math


class TokamakConfig:
    """ITER-like tokamak parameters (SI units)."""

    # ── Geometry ──
    R0 = 6.2              # Major radius [m]
    a  = 2.0              # Minor radius [m]
    epsilon = a / R0      # Inverse aspect ratio
    kappa = 1.7           # Elongation
    delta_tri = 0.33      # Triangularity

    # ── Magnetic field ──
    B0 = 5.3              # Toroidal field on axis [T]
    Ip = 15e6             # Plasma current [A]
    q0 = 1.0              # Central safety factor
    qa = 4.5              # Edge safety factor
    N_TF = 18             # Number of TF coils (for ripple)

    # ── Plasma ──
    n0     = 1.0e20       # Central density [m⁻³]
    T0_i   = 15.0         # Central ion temperature [keV]
    T0_e   = 15.0         # Central electron temperature [keV]
    Z_eff  = 1.7          # Effective charge
    Z_imp  = 6            # Impurity charge (Carbon)
    f_DT   = 0.5          # D-T fuel mix fraction (50-50)

    # ── Physical constants ──
    e       = 1.602176634e-19   # Elementary charge [C]
    keV_to_J = 1.602176634e-16  # 1 keV in Joules
    mi      = 3.3437e-27        # DT average ion mass [kg]
    me      = 9.1094e-31        # Electron mass [kg]
    mp      = 1.6726e-27        # Proton mass [kg]
    eps0    = 8.8542e-12        # Vacuum permittivity [F/m]
    mu0     = 1.2566e-6         # Vacuum permeability [H/m]
    kB_SI   = 1.3807e-23        # Boltzmann constant [J/K]
    E_alpha = 3.5e3             # Alpha particle energy [keV]
    c_light = 2.998e8           # Speed of light [m/s]

    # ── Numerical ──
    Nr     = 128          # Radial grid points
    Ntheta = 128          # Poloidal grid points
    Nphi   = 64           # Toroidal grid points (for 3D export)
    Nphi_sections = 8     # Toroidal cross-section count (for plots)
    Nparticles = 100_000  # PIC particle count

    dt      = 1.0e-7      # PIC time step [s]
    Nt_pic  = 400         # PIC steps
    Nt_fvm  = 2000        # FVM diffusion steps
    dt_fvm  = 1.0e-4      # FVM time step [s]

    # ── Transport model parameters ──
    alpha_n   = 0.5       # Density peaking exponent
    alpha_T   = 1.5       # Temperature peaking exponent
    ln_Lambda = 17.0      # Coulomb logarithm
    chi_anomalous_mult = 0.5   # Fraction of Bohm for anomalous transport
    lambda_SOL = 0.02     # SOL decay length (in ρ units)

    # ── 3D export settings ──
    n_snapshots_3d = 20   # Number of 3D snapshots to export
    n_particle_export = 500  # Particles to include in 3D export
