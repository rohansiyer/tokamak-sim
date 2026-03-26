"""
Finite Volume Method (FVM) — radial transport solver.

Solves 1D radial diffusion for density and temperature with:
  - Neoclassical + anomalous transport coefficients
  - Alpha heating source
  - Bremsstrahlung and cyclotron radiation sinks
  - SOL boundary model
"""

import numpy as np
from numba import njit

from tokamak.config import TokamakConfig
from tokamak.equilibrium import (
    density_profile, temperature_profile, q_profile,
    B_toroidal, B_poloidal, impurity_profile,
    DT_reaction_rate,
    alpha_heating_power, bremsstrahlung_loss, cyclotron_radiation_loss,
)


# ═══════════════════════════════════════════════════════════════
#  Transport coefficients
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def classical_diffusion(rho, T_keV, B, n, mi, Z_eff, ln_Lambda):
    """
    Classical diffusion D_cl = ρ_L² ν_⊥.
    ν_⊥ = n Z⁴ e⁴ lnΛ / (12π^(3/2) ε₀² m^(1/2) T^(3/2))
    """
    keV_to_J = 1.602176634e-16
    e    = 1.602176634e-19
    eps0 = 8.8542e-12

    T_J = T_keV * keV_to_J
    if T_J < 1e-20 or B < 0.01 or n < 1e10:
        return 1e-4

    rho_L = np.sqrt(mi * T_J) / (e * B)
    nu_perp = (n * Z_eff * e**4 * ln_Lambda /
               (12.0 * np.pi**1.5 * eps0**2 * mi**0.5 * T_J**1.5))
    D_cl = rho_L**2 * nu_perp
    return max(D_cl, 1e-6)


@njit(cache=True)
def neoclassical_diffusion(rho_norm, T_keV, B, n, mi, Z_eff,
                           q, epsilon, R0, ln_Lambda):
    """
    Neoclassical transport with proper collisionality regimes.

    ν* = qR₀ ν_ii / (ε^(3/2) v_th)
    Banana (ν* < 1):  D ~ ε^(-3/2) q² D_cl
    Plateau (1 < ν* < ε^(-3/2)):  D ~ q v_th ρ_L / R
    Pfirsch-Schlüter (ν* > ε^(-3/2)):  D ~ 2 q² D_cl
    """
    keV_to_J = 1.602176634e-16
    e    = 1.602176634e-19
    eps0 = 8.8542e-12

    D_cl = classical_diffusion(rho_norm, T_keV, B, n, mi, Z_eff, ln_Lambda)
    eps = max(epsilon * rho_norm, 0.01)

    T_J  = T_keV * keV_to_J
    v_th = np.sqrt(2.0 * max(T_J, 1e-20) / mi)
    rho_L = np.sqrt(mi * max(T_J, 1e-20)) / (e * B)

    # Ion-ion collision frequency
    nu_ii = (n * Z_eff * e**4 * ln_Lambda /
             (12.0 * np.pi**1.5 * eps0**2 * mi**0.5 * max(T_J, 1e-20)**1.5))

    # Collisionality parameter
    nu_star = q * R0 * nu_ii / (eps**1.5 * max(v_th, 1e-3))
    eps_inv32 = eps**(-1.5)

    if nu_star < 1.0:
        # Banana regime
        D_neo = eps_inv32 * q**2 * D_cl
    elif nu_star < eps_inv32:
        # Plateau regime
        D_neo = np.pi * q * v_th * rho_L / (2.0 * R0)
    else:
        # Pfirsch-Schlüter regime
        D_neo = 2.0 * q**2 * D_cl

    return max(D_neo, 1e-5)


# ═══════════════════════════════════════════════════════════════
#  FVM diffusion step
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def fvm_diffusion_step(n, T, chi_n, chi_T, rho, dr, dt,
                       source_n, source_T, sink_T):
    """
    1D radial FVM step for density and temperature.
    Conservative form: ∂f/∂t = (1/r) ∂/∂r(r D ∂f/∂r) + S - L
    Includes particle confinement loss n/τ_p.
    """
    Nr = len(n)
    n_new = n.copy()
    T_new = T.copy()
    tau_p = 0.5  # particle confinement time [s]
    tau_E = 1.5  # energy confinement time [s] (ITER H-mode scaling)

    for i in range(1, Nr - 1):
        r  = max(rho[i], 0.001)
        rm = 0.5 * (rho[i-1] + rho[i])
        rp = 0.5 * (rho[i]   + rho[i+1])

        # Density diffusion + source - confinement loss
        Dm = 0.5 * (chi_n[i-1] + chi_n[i])
        Dp = 0.5 * (chi_n[i]   + chi_n[i+1])
        flux_m = Dm * rm * (n[i] - n[i-1]) / dr
        flux_p = Dp * rp * (n[i+1] - n[i]) / dr
        n_new[i] = (n[i] + dt / (r * dr) * (flux_p - flux_m)
                    + dt * source_n[i] - dt * n[i] / tau_p)

        # Temperature diffusion + net heating/cooling - confinement loss
        Dm_T = 0.5 * (chi_T[i-1] + chi_T[i])
        Dp_T = 0.5 * (chi_T[i]   + chi_T[i+1])
        flux_m_T = Dm_T * rm * (T[i] - T[i-1]) / dr
        flux_p_T = Dp_T * rp * (T[i+1] - T[i]) / dr
        T_new[i] = (T[i] + dt / (r * dr) * (flux_p_T - flux_m_T)
                    + dt * source_T[i] - dt * sink_T[i]
                    - dt * T[i] / tau_E)

        # Clamp to physical range
        n_new[i] = min(max(n_new[i], 1e16), 1e22)
        T_new[i] = min(max(T_new[i], 0.01), 200.0)

    # Boundary conditions
    n_new[0]  = n_new[1]          # Neumann at core
    T_new[0]  = T_new[1]
    # SOL boundary
    n_new[-1] = min(max(n_new[-2] * 0.3, 1e16), 1e21)
    T_new[-1] = min(max(T_new[-2] * 0.2, 0.02), 5.0)

    return n_new, T_new


# ═══════════════════════════════════════════════════════════════
#  Full FVM transport driver
# ═══════════════════════════════════════════════════════════════

def run_fvm_transport(cfg: TokamakConfig):
    """Run full FVM radial transport simulation with realistic sources and sinks."""
    print("  [FVM] Initializing radial transport solver...")

    rho = np.linspace(0, 1, cfg.Nr)
    dr = rho[1] - rho[0]

    # Initial profiles
    n  = np.array([density_profile(r, cfg.n0, cfg.alpha_n) for r in rho])
    Ti = np.array([temperature_profile(r, cfg.T0_i, cfg.alpha_T) for r in rho])
    Te = np.array([temperature_profile(r, cfg.T0_e, cfg.alpha_T) for r in rho])

    # Transport coefficients
    chi_n = np.zeros(cfg.Nr)
    chi_T = np.zeros(cfg.Nr)
    D_cl_arr  = np.zeros(cfg.Nr)
    D_neo_arr = np.zeros(cfg.Nr)

    for i in range(cfg.Nr):
        R_loc = cfg.R0 + cfg.a * rho[i]
        B_loc = B_toroidal(R_loc, cfg.B0, cfg.R0)
        q_loc = q_profile(rho[i], cfg.q0, cfg.qa)

        D_cl = classical_diffusion(rho[i], Ti[i], B_loc, n[i],
                                   cfg.mi, cfg.Z_eff, cfg.ln_Lambda)
        D_neo = neoclassical_diffusion(rho[i], Ti[i], B_loc, n[i],
                                       cfg.mi, cfg.Z_eff, q_loc,
                                       cfg.epsilon, cfg.R0, cfg.ln_Lambda)
        D_cl_arr[i]  = D_cl
        D_neo_arr[i] = D_neo

        # Anomalous (gyro-Bohm scaling: D_gB = (ρ_i/a) × D_Bohm)
        D_bohm = Ti[i] * cfg.keV_to_J / (16.0 * cfg.e * max(B_loc, 0.01))
        rho_star = np.sqrt(cfg.mi * Ti[i] * cfg.keV_to_J) / (cfg.e * B_loc * cfg.a)
        D_anom = cfg.chi_anomalous_mult * rho_star * D_bohm

        chi_n[i] = D_neo + D_anom
        chi_T[i] = 3.0 * chi_n[i]

    # Particle fueling (NBI-like, peaked) — time-independent
    source_n = np.zeros(cfg.Nr)
    for i in range(cfg.Nr):
        source_n[i] = cfg.n0 * 1e-3 * np.exp(-rho[i]**2 / 0.15)

    # Target temperature profile for relaxation reference
    T_target = np.array([temperature_profile(r, cfg.T0_i, cfg.alpha_T) for r in rho])

    print(f"  [FVM] Running {cfg.Nt_fvm} steps (dt={cfg.dt_fvm:.1e} s)...")

    n_hist  = [n.copy()]
    Ti_hist = [Ti.copy()]
    save_every = max(1, cfg.Nt_fvm // 10)

    for step in range(cfg.Nt_fvm):
        # Recompute source_T and sink_T from CURRENT profiles
        source_T = np.zeros(cfg.Nr)
        sink_T   = np.zeros(cfg.Nr)

        for i in range(1, cfg.Nr - 1):
            R_loc = cfg.R0 + cfg.a * rho[i]
            B_loc = B_toroidal(R_loc, cfg.B0, cfg.R0)
            n_loc = min(max(n[i], 1e16), 1e22)  # clamp to physical range
            T_loc = min(max(Ti[i], 0.01), 200.0)  # clamp T

            # Alpha heating from CURRENT n, T
            nD = 0.5 * n_loc
            sv = DT_reaction_rate(T_loc)
            # Compute in safe order to avoid overflow
            P_alpha = nD * sv * cfg.E_alpha * cfg.keV_to_J * nD * 0.25
            heating = P_alpha / (n_loc * cfg.keV_to_J)

            # NBI heating (small, Gaussian)
            heating += cfg.T0_i * 0.02 * np.exp(-rho[i]**2 / 0.2)

            # Radiation losses from CURRENT T
            Te_loc = 0.9 * T_loc
            P_brem = bremsstrahlung_loss(n_loc, Te_loc, cfg.Z_eff)
            P_cyc  = cyclotron_radiation_loss(n_loc, Te_loc, B_loc)
            cooling = (P_brem + P_cyc) / (n_loc * cfg.keV_to_J)

            # Net source (heating - cooling)
            net = heating - cooling

            if not np.isfinite(net):
                net = 0.0

            # Cap: |ΔT| ≤ 5% of current T per step
            max_rate = 0.05 * T_loc / max(cfg.dt_fvm, 1e-10)
            if net > 0:
                source_T[i] = min(net, max_rate)
            else:
                sink_T[i] = min(abs(net), max_rate)

        # Cap diffusion coefficients for CFL stability
        chi_max_cfl = 0.4 * dr**2 / cfg.dt_fvm
        chi_n_step = np.minimum(chi_n, chi_max_cfl)
        chi_T_step = np.minimum(chi_T, chi_max_cfl)

        n, Ti = fvm_diffusion_step(n, Ti, chi_n_step, chi_T_step, rho, dr,
                                   cfg.dt_fvm, source_n, source_T, sink_T)

        # Replace NaN/inf, then enforce floors
        n  = np.nan_to_num(n,  nan=1e18, posinf=1e22, neginf=1e16)
        Ti = np.nan_to_num(Ti, nan=1.0,  posinf=100., neginf=0.01)
        for i in range(cfg.Nr):
            n[i]  = max(n[i],  1e16)
            Ti[i] = max(Ti[i], 0.01)

        if (step + 1) % save_every == 0:
            n_hist.append(n.copy())
            Ti_hist.append(Ti.copy())

    Te = 0.9 * Ti  # Te ≈ 0.9 Ti (electron-ion equilibration)

    # Impurity density
    n_z = np.array([impurity_profile(r, cfg.n0, cfg.Z_imp) * cfg.n0 * 0.02
                    for r in rho])

    # Compute fusion power for validation
    P_fus_total = 0.0
    for i in range(cfg.Nr):
        P_a = alpha_heating_power(rho[i], n[i], Ti[i],
                                  cfg.E_alpha, cfg.keV_to_J)
        vol_shell = 2 * np.pi * cfg.R0 * 2 * np.pi * cfg.a**2 * rho[i] * dr
        P_fus_total += P_a * 5.0 * vol_shell  # P_fus = 5 × P_alpha
    print(f"  [FVM] Estimated fusion power: {P_fus_total/1e6:.0f} MW")

    # Confinement time estimate
    W_total = 0.0
    for i in range(cfg.Nr):
        p_loc = n[i] * (Ti[i] + Te[i]) * cfg.keV_to_J
        vol_shell = 2 * np.pi * cfg.R0 * 2 * np.pi * cfg.a**2 * rho[i] * dr
        W_total += 1.5 * p_loc * vol_shell
    P_heat = P_fus_total / 5.0  # alpha power
    tau_E = W_total / max(P_heat, 1e3)
    print(f"  [FVM] Energy confinement time: {tau_E:.2f} s")
    print(f"  [FVM] Central Ti={Ti[0]:.1f} keV, n0={n[0]:.2e} m⁻³")
    print("  [FVM] Transport solver complete.")

    return {
        'rho': rho, 'n': n, 'Ti': Ti, 'Te': Te, 'n_z': n_z,
        'chi_n': chi_n, 'chi_T': chi_T,
        'D_cl': D_cl_arr, 'D_neo': D_neo_arr,
        'n_hist': n_hist, 'Ti_hist': Ti_hist,
        'source_T': source_T, 'sink_T': sink_T,
    }
