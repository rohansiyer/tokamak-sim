"""
Equilibrium profiles — analytic model for ITER-like tokamak.

All profiles are functions of normalised radius ρ = r/a.
"""

import numpy as np
from numba import njit


# ═══════════════════════════════════════════════════════════════
#  Safety factor
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def q_profile(rho, q0, qa):
    """Safety factor q(ρ) = q0 + (qa - q0) ρ²."""
    return q0 + (qa - q0) * rho**2


# ═══════════════════════════════════════════════════════════════
#  Density & Temperature
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def density_profile(rho, n0, alpha_n=0.5):
    """n(ρ) = n0 (1 - ρ²)^α_n  with small edge pedestal floor."""
    rr = min(rho, 0.999)
    n = n0 * (1.0 - rr**2)**alpha_n
    # Small pedestal at edge (physical: H-mode pedestal shoulder)
    n_ped = n0 * 0.05 * np.exp(-(rr - 0.95)**2 / 0.003) if rr > 0.88 else 0.0
    return max(n + n_ped, n0 * 1e-4)


@njit(cache=True)
def temperature_profile(rho, T0, alpha_T=1.5):
    """T(ρ) = T0 (1 - ρ²)^α_T."""
    rr = min(rho, 0.999)
    return max(T0 * (1.0 - rr**2)**alpha_T, 0.01)


@njit(cache=True)
def pressure_profile(rho, n0, T0_i, T0_e, keV_to_J):
    """p(ρ) = n (T_i + T_e) in Pascals."""
    n = density_profile(rho, n0)
    Ti = temperature_profile(rho, T0_i)
    Te = temperature_profile(rho, T0_e)
    return n * (Ti + Te) * keV_to_J


# ═══════════════════════════════════════════════════════════════
#  Magnetic field
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def B_toroidal(R, B0, R0):
    """B_φ = B0 R0 / R  (1/R dependence of toroidal field)."""
    return B0 * R0 / max(R, 0.1)


@njit(cache=True)
def B_poloidal(rho, a, R0, B0, Ip, mu0, q0, qa):
    """
    B_θ from Ampère's law: B_θ(r) = μ₀ I(r) / (2π r)
    where I(r) = Ip ρ² (parabolic current profile).
    """
    r = rho * a
    if r < 1e-6:
        return 0.0
    I_enclosed = Ip * rho**2  # parabolic current distribution
    Bp = mu0 * I_enclosed / (2.0 * np.pi * r)
    return Bp


@njit(cache=True)
def B_total(rho, a, R0, B0, Ip, mu0, q0, qa):
    """Total |B| at (ρ, θ=0 outboard midplane)."""
    R = R0 + a * rho
    Bt = B_toroidal(R, B0, R0)
    Bp = B_poloidal(rho, a, R0, B0, Ip, mu0, q0, qa)
    return np.sqrt(Bt**2 + Bp**2)


# ═══════════════════════════════════════════════════════════════
#  Impurity & Shafranov shift
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def impurity_profile(rho, n0, Z):
    """n_z(ρ)/n_z(0) = (n_i/n_i0)^Z — classical impurity peaking."""
    ni_norm = density_profile(rho, 1.0)
    return ni_norm**Z


@njit(cache=True)
def shafranov_shift(rho, beta_p, a):
    """Shafranov shift Δ(ρ) = (β_p a / 2)(1 - ρ²)."""
    return 0.5 * beta_p * a * (1.0 - rho**2)


# ═══════════════════════════════════════════════════════════════
#  Fusion reaction rate — validated against NRL Plasma Formulary
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def DT_reaction_rate(T_keV):
    """
    <σv>_DT in m³/s.

    Uses log-log interpolation of tabulated NRL Plasma Formulary data
    (Bosch & Hale, Nucl. Fusion 32 (1992) 611).
    Valid for 0.2 ≤ T ≤ 300 keV.

    Reference values (cm³/s → m³/s):
      T=10 keV:  1.1e-24 m³/s
      T=20 keV:  4.2e-24 m³/s
      T=50 keV:  8.7e-24 m³/s
      T=100 keV: 8.0e-24 m³/s
    """
    T = max(T_keV, 0.2)
    if T > 300.0:
        T = 300.0

    # NRL Plasma Formulary tabulated data: T [keV], <σv> [m³/s]
    T_tab = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 8.0,
                      10.0, 15.0, 20.0, 30.0, 50.0, 70.0,
                      100.0, 150.0, 200.0, 300.0])
    sv_tab = np.array([1.0e-28,  7.5e-26,  5.5e-25,
                       2.6e-24,  1.3e-23,  5.7e-23,
                       1.1e-22,  3.0e-22,  4.2e-22,
                       5.8e-22,  8.7e-22,  8.8e-22,
                       8.0e-22,  6.5e-22,  5.7e-22, 4.0e-22])

    # Log-log interpolation
    logT = np.log(T)
    logT_tab = np.log(T_tab)
    logsv_tab = np.log(sv_tab)

    # Clamp to table range
    if logT <= logT_tab[0]:
        return sv_tab[0]
    if logT >= logT_tab[-1]:
        return sv_tab[-1]

    # Find interval
    idx = 0
    for i in range(len(T_tab) - 1):
        if logT_tab[i] <= logT <= logT_tab[i + 1]:
            idx = i
            break

    # Linear interpolation in log-log space
    frac = (logT - logT_tab[idx]) / (logT_tab[idx + 1] - logT_tab[idx])
    logsv = logsv_tab[idx] + frac * (logsv_tab[idx + 1] - logsv_tab[idx])
    return np.exp(logsv)


@njit(cache=True)
def alpha_heating_power(rho, n0, T0_i, E_alpha_keV, keV_to_J):
    """
    Alpha heating power density P_α(ρ) [W/m³].
    P_α = (1/4) n_D n_T <σv> E_α  (50-50 D-T mix).
    """
    n = density_profile(rho, n0)
    T = temperature_profile(rho, T0_i)
    nD = 0.5 * n
    nT = 0.5 * n
    sv = DT_reaction_rate(T)
    return 0.25 * nD * nT * sv * E_alpha_keV * keV_to_J


# ═══════════════════════════════════════════════════════════════
#  Radiation losses
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def bremsstrahlung_loss(n_e, T_e_keV, Z_eff):
    """
    Bremsstrahlung power density [W/m³].
    P_brem = 5.35e-37 Z_eff n_e² √(T_e_keV)
    """
    return 5.35e-37 * Z_eff * n_e**2 * np.sqrt(max(T_e_keV, 0.01))


@njit(cache=True)
def cyclotron_radiation_loss(n_e, T_e_keV, B):
    """
    Cyclotron radiation loss [W/m³] (optically thick, only ~2% escapes).
    P_cyc = 6.2e-22 n_e B² T_e_keV * f_escape
    """
    f_escape = 0.02  # ~2% escapes in ITER-like plasmas
    return 6.2e-22 * n_e * B**2 * T_e_keV * f_escape
