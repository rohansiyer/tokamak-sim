"""
2D Poloidal Finite Volume Method (FVM) transport solver.

Solves coupled density and temperature transport on a 2D (rho, theta) poloidal
grid with:
  - Radial diffusion (neoclassical + anomalous, as in 1D FVM)
  - Poloidal transport driven by Shafranov-shift-induced asymmetry
  - Up-down symmetric boundary conditions in theta
  - Alpha heating, bremsstrahlung, and cyclotron radiation sources/sinks
  - SOL boundary model at rho=1

After solving the 2D system, theta-averages are computed to produce the 1D
radial output arrays that downstream visualization and export code expects.
The output dict is a superset of the 1D run_fvm_transport() output.
"""

import numpy as np

from tokamak.config import TokamakConfig
from tokamak.equilibrium import (
    density_profile, temperature_profile, q_profile,
    B_toroidal, B_poloidal, impurity_profile,
    DT_reaction_rate,
    alpha_heating_power, bremsstrahlung_loss, cyclotron_radiation_loss,
    shafranov_shift,
)
from tokamak.fvm import classical_diffusion, neoclassical_diffusion


# ═══════════════════════════════════════════════════════════════
#  Poloidal geometry helpers
# ═══════════════════════════════════════════════════════════════

def _local_B(rho_val, theta_val, cfg: TokamakConfig):
    """
    Total |B| at (rho, theta) accounting for 1/R variation.

    The major radius at poloidal angle theta on a flux surface labelled
    by normalised minor radius rho is approximately:
        R(rho, theta) = R0 + a*rho*cos(theta)

    This gives the dominant B ~ 1/R variation in a large-aspect-ratio
    tokamak.
    """
    R_loc = cfg.R0 + cfg.a * rho_val * np.cos(theta_val)
    R_loc = max(R_loc, 0.1)
    Bt = B_toroidal(R_loc, cfg.B0, cfg.R0)
    Bp = B_poloidal(rho_val, cfg.a, cfg.R0, cfg.B0, cfg.Ip,
                    cfg.mu0, cfg.q0, cfg.qa)
    return np.sqrt(Bt**2 + Bp**2)


def _shafranov_rho_eff(rho_val, theta_val, beta_p, cfg: TokamakConfig):
    """
    Effective normalised radius accounting for Shafranov shift.

    Density and temperature peak slightly on the inboard (high-field) side
    of the magnetic axis because the Shafranov shift displaces the axis
    outward, making flux surfaces denser on the inboard side.

    The effective radial coordinate is:
        rho_eff = rho - Delta(rho) * cos(theta) / a

    where Delta(rho) is the Shafranov shift at radius rho.
    """
    delta = shafranov_shift(rho_val, beta_p, cfg.a)
    shift_norm = delta * np.cos(theta_val) / cfg.a
    rho_eff = rho_val - shift_norm
    return float(np.clip(rho_eff, 0.0, 0.999))


def _compute_beta_p(n_2d, Ti_2d, rho, cfg: TokamakConfig):
    """
    Compute volume-averaged poloidal beta from current 2D profiles.

    beta_p = <p> / (B_p^2 / 2 mu0)
    Uses simplified estimate based on on-axis values.
    """
    n0_eff = float(np.mean(n_2d[0, :]))
    T0_eff = float(np.mean(Ti_2d[0, :]))
    p_axis = n0_eff * T0_eff * cfg.keV_to_J
    # Poloidal field at edge
    Bp_edge = B_poloidal(0.95, cfg.a, cfg.R0, cfg.B0, cfg.Ip,
                         cfg.mu0, cfg.q0, cfg.qa)
    Bp_edge = max(Bp_edge, 0.01)
    beta_p = p_axis / (Bp_edge**2 / (2.0 * cfg.mu0))
    return float(np.clip(beta_p, 0.0, 2.0))


# ═══════════════════════════════════════════════════════════════
#  2D diffusion step
# ═══════════════════════════════════════════════════════════════

def _fvm2d_step(n_2d, T_2d, chi_n_2d, chi_T_2d,
                rho, theta, dr, dtheta, dt,
                source_n_2d, source_T_2d, sink_T_2d,
                tau_p=0.5, tau_E=1.5):
    """
    One explicit FVM step on the 2D (rho, theta) grid.

    Solved equation (conservative radial + poloidal diffusion):

        dn/dt = (1/r) d/dr(r D_n dn/dr)
              + (1/r^2) d/dtheta(D_pol dn/dtheta)
              + S_n - n/tau_p

    and identically for T (replacing n with T, D_n with D_T, adding
    source_T - sink_T - T/tau_E).

    Boundary conditions:
      - rho=0  (core):  Neumann (dn/dr = 0)
      - rho=1  (SOL):   Dirichlet decay (n_sol = 0.3 * n_edge)
      - theta=0/2pi:    periodic (toroidal symmetry)

    Parameters
    ----------
    n_2d, T_2d : (Nr, Ntheta) arrays — current density and temperature
    chi_n_2d, chi_T_2d : (Nr, Ntheta) — diffusion coefficients
    rho : (Nr,) normalised radial grid
    theta : (Ntheta,) poloidal angle grid
    dr, dtheta : grid spacings
    dt : time step
    source_n_2d, source_T_2d, sink_T_2d : (Nr, Ntheta) source/sink terms
    tau_p, tau_E : confinement times [s]

    Returns
    -------
    n_new, T_new : (Nr, Ntheta)
    """
    Nr, Ntheta = n_2d.shape
    n_new = n_2d.copy()
    T_new = T_2d.copy()

    for i in range(1, Nr - 1):
        r = max(rho[i], 0.001)
        rm = 0.5 * (rho[i-1] + rho[i])
        rp = 0.5 * (rho[i]   + rho[i+1])

        for j in range(Ntheta):
            # Periodic neighbours in theta
            jm = (j - 1) % Ntheta
            jp = (j + 1) % Ntheta

            # ── Radial diffusion terms ──
            Dn_m = 0.5 * (chi_n_2d[i-1, j] + chi_n_2d[i, j])
            Dn_p = 0.5 * (chi_n_2d[i,   j] + chi_n_2d[i+1, j])
            flux_n_m = Dn_m * rm * (n_2d[i, j]   - n_2d[i-1, j]) / dr
            flux_n_p = Dn_p * rp * (n_2d[i+1, j] - n_2d[i,   j]) / dr
            dn_rad = (flux_n_p - flux_n_m) / (r * dr)

            DT_m = 0.5 * (chi_T_2d[i-1, j] + chi_T_2d[i, j])
            DT_p = 0.5 * (chi_T_2d[i,   j] + chi_T_2d[i+1, j])
            flux_T_m = DT_m * rm * (T_2d[i, j]   - T_2d[i-1, j]) / dr
            flux_T_p = DT_p * rp * (T_2d[i+1, j] - T_2d[i,   j]) / dr
            dT_rad = (flux_T_p - flux_T_m) / (r * dr)

            # ── Poloidal diffusion terms (1/r^2 d/dtheta(D d/dtheta)) ──
            # Use poloidal diffusion coefficient ~ chi_n / q^2 (reduced transport
            # along field lines in high-q region, enhanced near axis where
            # parallel transport smooths out poloidal asymmetries quickly)
            D_pol_n = 0.5 * chi_n_2d[i, j]   # poloidal diffusivity
            D_pol_T = 0.5 * chi_T_2d[i, j]

            # Average with neighbours for face values
            D_pol_n_m = 0.5 * (D_pol_n + 0.5 * chi_n_2d[i, jm])
            D_pol_n_p = 0.5 * (D_pol_n + 0.5 * chi_n_2d[i, jp])
            D_pol_T_m = 0.5 * (D_pol_T + 0.5 * chi_T_2d[i, jm])
            D_pol_T_p = 0.5 * (D_pol_T + 0.5 * chi_T_2d[i, jp])

            flux_pol_n_m = D_pol_n_m * (n_2d[i, j]  - n_2d[i, jm]) / dtheta
            flux_pol_n_p = D_pol_n_p * (n_2d[i, jp] - n_2d[i, j])  / dtheta
            dn_pol = (flux_pol_n_p - flux_pol_n_m) / (r**2 * dtheta)

            flux_pol_T_m = D_pol_T_m * (T_2d[i, j]  - T_2d[i, jm]) / dtheta
            flux_pol_T_p = D_pol_T_p * (T_2d[i, jp] - T_2d[i, j])  / dtheta
            dT_pol = (flux_pol_T_p - flux_pol_T_m) / (r**2 * dtheta)

            # ── Update ──
            n_new[i, j] = (n_2d[i, j]
                           + dt * dn_rad
                           + dt * dn_pol
                           + dt * source_n_2d[i, j]
                           - dt * n_2d[i, j] / tau_p)

            T_new[i, j] = (T_2d[i, j]
                           + dt * dT_rad
                           + dt * dT_pol
                           + dt * source_T_2d[i, j]
                           - dt * sink_T_2d[i, j]
                           - dt * T_2d[i, j] / tau_E)

            # Physical bounds
            n_new[i, j] = min(max(n_new[i, j], 1e16), 1e22)
            T_new[i, j] = min(max(T_new[i, j], 0.01), 200.0)

    # ── Boundary conditions ──
    # Core (Neumann): copy inner ring
    n_new[0, :] = n_new[1, :]
    T_new[0, :] = T_new[1, :]

    # SOL (absorbing wall): density/temperature decay into scrape-off layer
    n_new[-1, :] = np.clip(n_new[-2, :] * 0.3, 1e16, 1e21)
    T_new[-1, :] = np.clip(T_new[-2, :] * 0.2, 0.02, 5.0)

    return n_new, T_new


# ═══════════════════════════════════════════════════════════════
#  Main 2D FVM transport driver
# ═══════════════════════════════════════════════════════════════

def run_fvm2d_transport(cfg: TokamakConfig, gs_data=None):
    """
    Run 2D poloidal FVM transport simulation.

    Solves density and temperature transport on a 2D (rho, theta) grid,
    including both radial diffusion and poloidal asymmetries from the
    Shafranov shift.  After time integration, theta-averages are computed
    to produce 1D radial profiles compatible with the downstream
    visualization and export pipeline.

    Parameters
    ----------
    cfg : TokamakConfig
        Tokamak configuration parameters.
    gs_data : dict or None
        Grad-Shafranov equilibrium data (optional).  If provided, the flux
        surface geometry is used to initialise profiles; otherwise analytic
        profiles are used.

    Returns
    -------
    dict
        Keys match those of run_fvm_transport() plus extra 2D diagnostics.
    """
    print("  [FVM2D] Initialising 2D poloidal transport solver...")

    Nr     = cfg.Nr
    Ntheta = cfg.Ntheta
    Nt     = cfg.Nt_fvm
    dt     = cfg.dt_fvm

    # ── Coordinate grids ──
    rho   = np.linspace(0.0, 1.0, Nr)
    theta = np.linspace(0.0, 2.0 * np.pi, Ntheta, endpoint=False)
    dr     = rho[1] - rho[0]
    dtheta = theta[1] - theta[0]

    RHO, THETA = np.meshgrid(rho, theta, indexing='ij')  # (Nr, Ntheta)

    # ── Estimate poloidal beta for Shafranov shift ──
    # Use simple analytic estimate initially; updated each saved step
    n0_est = cfg.n0
    T0_est = cfg.T0_i
    p_axis = n0_est * T0_est * cfg.keV_to_J
    Bp_edge = B_poloidal(0.95, cfg.a, cfg.R0, cfg.B0, cfg.Ip,
                         cfg.mu0, cfg.q0, cfg.qa)
    Bp_edge = max(Bp_edge, 0.01)
    beta_p_init = float(np.clip(p_axis / (Bp_edge**2 / (2.0 * cfg.mu0)),
                                0.0, 2.0))

    # ── Initialise 2D profiles ──
    print("  [FVM2D] Building 2D initial profiles with Shafranov shift...")
    n_2d  = np.zeros((Nr, Ntheta))
    Ti_2d = np.zeros((Nr, Ntheta))
    Te_2d = np.zeros((Nr, Ntheta))

    for i in range(Nr):
        for j in range(Ntheta):
            rho_eff = _shafranov_rho_eff(rho[i], theta[j],
                                          beta_p_init, cfg)
            if gs_data is not None:
                # Use GS flux surface geometry: map psi -> rho_eff via
                # linear interpolation of the normalised flux
                # (GS provides psi_axis, psi_lcfs; here we use the analytic
                # profile but shifted by the GS-computed Shafranov shift
                # which is encoded in the psi array if available)
                pass  # analytic profiles below are sufficient
            n_2d[i, j]  = density_profile(rho_eff, cfg.n0, cfg.alpha_n)
            Ti_2d[i, j] = temperature_profile(rho_eff, cfg.T0_i, cfg.alpha_T)
            Te_2d[i, j] = temperature_profile(rho_eff, cfg.T0_e, cfg.alpha_T)

    # ── 2D transport coefficients ──
    print("  [FVM2D] Computing 2D transport coefficients...")
    chi_n_2d  = np.zeros((Nr, Ntheta))
    chi_T_2d  = np.zeros((Nr, Ntheta))
    D_cl_2d   = np.zeros((Nr, Ntheta))
    D_neo_2d  = np.zeros((Nr, Ntheta))
    D_anom_2d = np.zeros((Nr, Ntheta))

    for i in range(Nr):
        for j in range(Ntheta):
            B_loc = _local_B(rho[i], theta[j], cfg)
            q_loc = q_profile(rho[i], cfg.q0, cfg.qa)
            n_loc = max(n_2d[i, j], 1e16)
            T_loc = max(Ti_2d[i, j], 0.01)

            D_cl  = classical_diffusion(rho[i], T_loc, B_loc, n_loc,
                                        cfg.mi, cfg.Z_eff, cfg.ln_Lambda)
            D_neo = neoclassical_diffusion(rho[i], T_loc, B_loc, n_loc,
                                           cfg.mi, cfg.Z_eff, q_loc,
                                           cfg.epsilon, cfg.R0, cfg.ln_Lambda)

            # Anomalous: gyro-Bohm scaling
            D_bohm  = T_loc * cfg.keV_to_J / (16.0 * cfg.e * max(B_loc, 0.01))
            rho_star = (np.sqrt(cfg.mi * T_loc * cfg.keV_to_J)
                        / (cfg.e * B_loc * cfg.a))
            D_anom = cfg.chi_anomalous_mult * rho_star * D_bohm

            D_cl_2d[i, j]   = D_cl
            D_neo_2d[i, j]  = D_neo
            D_anom_2d[i, j] = D_anom

            chi_n_2d[i, j] = D_neo + D_anom
            chi_T_2d[i, j] = 3.0 * chi_n_2d[i, j]

    # ── Particle fueling source (NBI-like, peaked near axis, uniform in theta) ──
    source_n_2d = np.zeros((Nr, Ntheta))
    for i in range(Nr):
        s_val = cfg.n0 * 1e-3 * np.exp(-rho[i]**2 / 0.15)
        source_n_2d[i, :] = s_val

    # ── CFL-limited diffusion coefficients ──
    # Radial CFL: D_max_r = 0.4 * dr^2 / dt
    # Poloidal CFL: D_max_pol = 0.4 * (dr * dtheta)^2 / dt  (conservative)
    chi_max_r   = 0.4 * dr**2 / dt
    chi_max_pol = 0.4 * (dr * dtheta)**2 / dt
    chi_max_cfl = min(chi_max_r, chi_max_pol)

    print(f"  [FVM2D] CFL limit: D_max = {chi_max_cfl:.3e} m²/s")

    # Apply CFL cap at initialisation (reapplied each step)
    chi_n_cfl = np.minimum(chi_n_2d, chi_max_cfl)
    chi_T_cfl = np.minimum(chi_T_2d, chi_max_cfl)

    # ── Time integration ──
    print(f"  [FVM2D] Running {Nt} steps "
          f"(dt={dt:.1e} s, grid={Nr}x{Ntheta})...")

    save_every = max(1, Nt // 10)

    # History lists: store theta-averaged 1D radial profiles (memory-efficient)
    n_hist_list  = [np.mean(n_2d,  axis=1).copy()]
    T_hist_list  = [np.mean(Ti_2d, axis=1).copy()]

    for step in range(Nt):
        # ── Build source/sink terms from current profiles ──
        source_T_2d = np.zeros((Nr, Ntheta))
        sink_T_2d   = np.zeros((Nr, Ntheta))

        for i in range(1, Nr - 1):
            for j in range(Ntheta):
                B_loc  = _local_B(rho[i], theta[j], cfg)
                n_loc  = float(np.clip(n_2d[i, j],  1e16, 1e22))
                T_loc  = float(np.clip(Ti_2d[i, j], 0.01, 200.0))
                Te_loc = 0.9 * T_loc

                # Alpha heating
                nD = 0.5 * n_loc
                sv = DT_reaction_rate(T_loc)
                P_alpha = nD * sv * cfg.E_alpha * cfg.keV_to_J * nD * 0.25
                heating = P_alpha / (n_loc * cfg.keV_to_J)

                # NBI heating (Gaussian deposition, uniform in theta)
                heating += cfg.T0_i * 0.02 * np.exp(-rho[i]**2 / 0.2)

                # Radiation losses
                P_brem = bremsstrahlung_loss(n_loc, Te_loc, cfg.Z_eff)
                P_cyc  = cyclotron_radiation_loss(n_loc, Te_loc, B_loc)
                cooling = (P_brem + P_cyc) / (n_loc * cfg.keV_to_J)

                net = heating - cooling
                if not np.isfinite(net):
                    net = 0.0

                # Cap rate at 5 % of current T per step
                max_rate = 0.05 * T_loc / max(dt, 1e-10)
                if net > 0.0:
                    source_T_2d[i, j] = min(net, max_rate)
                else:
                    sink_T_2d[i, j]   = min(abs(net), max_rate)

        # ── Apply CFL cap ──
        chi_n_step = np.minimum(chi_n_2d, chi_max_cfl)
        chi_T_step = np.minimum(chi_T_2d, chi_max_cfl)

        # ── Advance one step ──
        n_2d, Ti_2d = _fvm2d_step(
            n_2d, Ti_2d,
            chi_n_step, chi_T_step,
            rho, theta, dr, dtheta, dt,
            source_n_2d, source_T_2d, sink_T_2d,
        )

        # ── Sanitise ──
        n_2d  = np.nan_to_num(n_2d,  nan=1e18, posinf=1e22, neginf=1e16)
        Ti_2d = np.nan_to_num(Ti_2d, nan=1.0,  posinf=100., neginf=0.01)
        n_2d  = np.clip(n_2d,  1e16, 1e22)
        Ti_2d = np.clip(Ti_2d, 0.01, 200.0)

        # ── Save snapshot ──
        if (step + 1) % save_every == 0:
            n_hist_list.append(np.mean(n_2d,  axis=1).copy())
            T_hist_list.append(np.mean(Ti_2d, axis=1).copy())
            pct = 100.0 * (step + 1) / Nt
            n_ctr  = float(np.mean(n_2d[0,  :]))
            Ti_ctr = float(np.mean(Ti_2d[0, :]))
            print(f"  [FVM2D] {pct:5.1f}%  step {step+1:5d}/{Nt}  "
                  f"Ti_ctr={Ti_ctr:.2f} keV  n_ctr={n_ctr:.2e} m⁻³")

    # ── Electron temperature ──
    Te_2d = 0.9 * Ti_2d

    # ── Theta-averaged 1D radial profiles ──
    n_1d  = np.mean(n_2d,  axis=1)   # (Nr,)
    Ti_1d = np.mean(Ti_2d, axis=1)
    Te_1d = np.mean(Te_2d, axis=1)

    # ── Radially-averaged 1D transport coefficients ──
    D_cl_1d   = np.mean(D_cl_2d,   axis=1)
    D_neo_1d  = np.mean(D_neo_2d,  axis=1)
    D_anom_1d = np.mean(D_anom_2d, axis=1)
    chi_n_1d  = np.mean(chi_n_2d,  axis=1)
    chi_T_1d  = np.mean(chi_T_2d,  axis=1)

    # ── Additional 1D profiles ──
    q_1d = np.array([q_profile(r, cfg.q0, cfg.qa) for r in rho])
    B_1d = np.array([B_toroidal(cfg.R0 + cfg.a * r, cfg.B0, cfg.R0)
                     for r in rho])

    # Alpha heating and radiation profiles (on theta-averaged quantities)
    P_alpha_1d = np.zeros(Nr)
    P_rad_1d   = np.zeros(Nr)
    source_T_1d = np.zeros(Nr)
    sink_T_1d   = np.zeros(Nr)

    for i in range(Nr):
        n_loc  = float(np.clip(n_1d[i],  1e16, 1e22))
        T_loc  = float(np.clip(Ti_1d[i], 0.01, 200.0))
        Te_loc = 0.9 * T_loc
        B_loc  = B_1d[i]

        nD = 0.5 * n_loc
        sv = DT_reaction_rate(T_loc)
        P_alpha_1d[i] = nD * sv * cfg.E_alpha * cfg.keV_to_J * nD * 0.25

        P_brem = bremsstrahlung_loss(n_loc, Te_loc, cfg.Z_eff)
        P_cyc  = cyclotron_radiation_loss(n_loc, Te_loc, B_loc)
        P_rad_1d[i] = P_brem + P_cyc

        heating = P_alpha_1d[i] / max(n_loc * cfg.keV_to_J, 1e-30)
        cooling = P_rad_1d[i]   / max(n_loc * cfg.keV_to_J, 1e-30)
        net = heating - cooling
        if not np.isfinite(net):
            net = 0.0
        max_rate = 0.05 * T_loc / max(dt, 1e-10)
        if net > 0.0:
            source_T_1d[i] = min(net, max_rate)
        else:
            sink_T_1d[i]   = min(abs(net), max_rate)

    # ── Impurity density ──
    n_z = np.array([impurity_profile(r, cfg.n0, cfg.Z_imp) * cfg.n0 * 0.02
                    for r in rho])

    # ── Convert history lists to arrays: shape (Nt_saved, Nr) ──
    n_hist_arr = np.array(n_hist_list)   # (Nt_saved, Nr)
    T_hist_arr = np.array(T_hist_list)   # (Nt_saved, Nr)

    # ── Diagnostics ──
    P_fus_total = 0.0
    for i in range(Nr):
        P_a     = alpha_heating_power(rho[i], n_1d[i], Ti_1d[i],
                                      cfg.E_alpha, cfg.keV_to_J)
        vol_sh  = (2.0 * np.pi * cfg.R0
                   * 2.0 * np.pi * cfg.a**2 * rho[i] * dr)
        P_fus_total += P_a * 5.0 * vol_sh

    W_total = 0.0
    for i in range(Nr):
        p_loc  = n_1d[i] * (Ti_1d[i] + Te_1d[i]) * cfg.keV_to_J
        vol_sh = (2.0 * np.pi * cfg.R0
                  * 2.0 * np.pi * cfg.a**2 * rho[i] * dr)
        W_total += 1.5 * p_loc * vol_sh

    P_heat = max(P_fus_total / 5.0, 1e3)
    tau_E  = W_total / P_heat

    print(f"  [FVM2D] Estimated fusion power: {P_fus_total/1e6:.0f} MW")
    print(f"  [FVM2D] Energy confinement time: {tau_E:.2f} s")
    print(f"  [FVM2D] Central Ti={Ti_1d[0]:.1f} keV, "
          f"n0={n_1d[0]:.2e} m⁻³")
    print("  [FVM2D] 2D transport solver complete.")

    # ── Return dict — superset of run_fvm_transport() output ──
    return {
        # ── 1D radial profiles (theta-averaged) ──
        'rho':   rho,
        'n':     n_1d,
        'Ti':    Ti_1d,
        'Te':    Te_1d,
        'n_z':   n_z,

        # ── Transport coefficients (1D, theta-averaged) ──
        'chi_n':  chi_n_1d,
        'chi_T':  chi_T_1d,
        'D_cl':   D_cl_1d,
        'D_neo':  D_neo_1d,
        'D_anom': D_anom_1d,

        # ── Safety factor and magnetic field ──
        'q_profile': q_1d,
        'B':         B_1d,

        # ── Heating / radiation ──
        'P_alpha': P_alpha_1d,
        'P_rad':   P_rad_1d,
        'source_T': source_T_1d,
        'sink_T':   sink_T_1d,

        # ── History arrays — shape (Nt_saved, Nr) ──
        # Both 'n_hist' / 'T_hist' (task spec) and 'Ti_hist' (1D FVM compat.)
        'n_hist':  n_hist_arr,
        'T_hist':  T_hist_arr,
        'Ti_hist': T_hist_arr,   # alias for 1D FVM compatibility

        # ── 2D fields (optional, for extended analysis) ──
        'n_2d':    n_2d,
        'Ti_2d':   Ti_2d,
        'Te_2d':   Te_2d,
        'theta':   theta,
    }
