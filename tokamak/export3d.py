"""
3D animation data export — saves time-resolved toroidal mesh and particle data.

Output: tokamak_3d_data.npz (loadable by ParaView, PyVista, Blender, Three.js, etc.)
"""

import numpy as np
from tokamak.config import TokamakConfig
from tokamak.grids import build_3d_mesh
from tokamak.equilibrium import (
    density_profile, temperature_profile, pressure_profile,
    B_toroidal, B_poloidal, q_profile, impurity_profile,
    DT_reaction_rate,
)


def export_3d_animation_data(fvm_data, rho_hist, theta_hist, phi_hist,
                             j_phi, j_bs, cfg: TokamakConfig,
                             output_path='tokamak_3d_data.npz'):
    """
    Export full 3D toroidal data for external animation tools.

    Saved arrays:
        X, Y, Z_cart:  (Nr, Ntheta, Nphi) Cartesian mesh
        R, Z:          (Nr, Ntheta, Nphi) cylindrical coords
        density:       (Nr, Ntheta, Nphi) density field [m⁻³]
        temperature:   (Nr, Ntheta, Nphi) temperature [keV]
        pressure:      (Nr, Ntheta, Nphi) pressure [Pa]
        B_magnitude:   (Nr, Ntheta, Nphi) |B| [T]
        current:       (Nr, Ntheta, Nphi) current density [A/m²]
        q_factor:      (Nr, Ntheta, Nphi) safety factor
        fusion_rate:   (Nr, Ntheta, Nphi) DT reaction rate [m⁻³s⁻¹]

        particle_X, particle_Y, particle_Z: (Npart, Nsnap) orbit trajectories

        n_hist:   (Nsnap_fvm, Nr) density time evolution
        Ti_hist:  (Nsnap_fvm, Nr) temperature time evolution
        time_fvm: (Nsnap_fvm,) time array for FVM snapshots

        rho_1d, theta_1d, phi_1d: 1D coordinate arrays
        config_*: scalar parameters
    """
    print("  [EXPORT] Building 3D toroidal mesh...")

    mesh = build_3d_mesh(cfg)
    Nr, Ntheta, Nphi = cfg.Nr, cfg.Ntheta, cfg.Nphi

    # ── Scalar fields on 3D mesh ──
    density_3d   = np.zeros((Nr, Ntheta, Nphi))
    temp_3d      = np.zeros((Nr, Ntheta, Nphi))
    pressure_3d  = np.zeros((Nr, Ntheta, Nphi))
    B_mag_3d     = np.zeros((Nr, Ntheta, Nphi))
    current_3d   = np.zeros((Nr, Ntheta, Nphi))
    q_3d         = np.zeros((Nr, Ntheta, Nphi))
    fusion_3d    = np.zeros((Nr, Ntheta, Nphi))

    rho_1d   = mesh['rho']
    theta_1d = mesh['theta']
    phi_1d   = mesh['phi']

    for i in range(Nr):
        rr = rho_1d[i]
        for j in range(Ntheta):
            th = theta_1d[j]
            for k in range(Nphi):
                ph = phi_1d[k]
                R_loc = mesh['R'][i, j, k]

                # Base profiles with poloidal variation (Shafranov)
                rr_eff = rr + 0.05 * (1 - rr) * np.cos(th)
                rr_eff = max(0, min(0.999, rr_eff))

                n_loc = density_profile(rr_eff, cfg.n0)
                T_loc = temperature_profile(rr_eff, cfg.T0_i)

                # Toroidal mode structure (n=1 kink, ballooning)
                q_loc = q_profile(rr_eff, cfg.q0, cfg.qa)
                mode = (1.0 + 0.08 * rr**2 * np.cos(ph - th / max(q_loc, 0.5))
                        + 0.03 * rr**3 * np.cos(3*ph - 2*th))
                balloon = 1.0 + 0.12 * rr**2 * np.cos(th)
                # TF ripple
                ripple = 1.0 + 0.015 * np.cos(cfg.N_TF * ph)

                density_3d[i, j, k]  = n_loc * mode * balloon
                temp_3d[i, j, k]     = T_loc * mode * balloon
                pressure_3d[i, j, k] = (density_3d[i, j, k]
                                         * (temp_3d[i, j, k] + 0.9*temp_3d[i, j, k])
                                         * cfg.keV_to_J)

                Bt = B_toroidal(R_loc, cfg.B0, cfg.R0) * ripple
                Bp = B_poloidal(rr_eff, cfg.a, cfg.R0, cfg.B0,
                                cfg.Ip, cfg.mu0, cfg.q0, cfg.qa)
                B_mag_3d[i, j, k] = np.sqrt(Bt**2 + Bp**2)

                q_3d[i, j, k] = q_loc

                if i < len(j_phi):
                    current_3d[i, j, k] = (j_phi[i] + j_bs[i]) * mode

                # Fusion reaction rate (for visualization)
                nD = 0.5 * density_3d[i, j, k]
                nT = 0.5 * density_3d[i, j, k]
                sv = DT_reaction_rate(temp_3d[i, j, k])
                fusion_3d[i, j, k] = nD * nT * sv

    # ── Particle orbits in Cartesian ──
    print("  [EXPORT] Converting particle orbits to Cartesian...")
    Npart = min(rho_hist.shape[0], cfg.n_particle_export)
    Nsnap = rho_hist.shape[1]

    part_X = np.zeros((Npart, Nsnap))
    part_Y = np.zeros((Npart, Nsnap))
    part_Z = np.zeros((Npart, Nsnap))

    for p in range(Npart):
        for s in range(Nsnap):
            r_p = rho_hist[p, s] * cfg.a
            th_p = theta_hist[p, s]
            ph_p = phi_hist[p, s]
            R_p = cfg.R0 + r_p * np.cos(th_p)
            Z_p = cfg.kappa * r_p * np.sin(th_p)
            part_X[p, s] = R_p * np.cos(ph_p)
            part_Y[p, s] = R_p * np.sin(ph_p)
            part_Z[p, s] = Z_p

    # ── FVM time history ──
    n_hist_arr  = np.array(fvm_data['n_hist'])
    Ti_hist_arr = np.array(fvm_data['Ti_hist'])
    time_fvm = np.linspace(0, cfg.Nt_fvm * cfg.dt_fvm,
                           len(fvm_data['n_hist']))

    # ── Save ──
    print(f"  [EXPORT] Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        # 3D mesh
        X=mesh['X'], Y=mesh['Y'], Z_cart=mesh['Z_cart'],
        R=mesh['R'], Z=mesh['Z'],
        # Scalar fields
        density=density_3d,
        temperature=temp_3d,
        pressure=pressure_3d,
        B_magnitude=B_mag_3d,
        current=current_3d,
        q_factor=q_3d,
        fusion_rate=fusion_3d,
        # Particle orbits
        particle_X=part_X, particle_Y=part_Y, particle_Z=part_Z,
        # Time histories
        n_hist=n_hist_arr, Ti_hist=Ti_hist_arr, time_fvm=time_fvm,
        # 1D grids
        rho_1d=rho_1d, theta_1d=theta_1d, phi_1d=phi_1d,
        fvm_rho=fvm_data['rho'], fvm_n=fvm_data['n'],
        fvm_Ti=fvm_data['Ti'], fvm_Te=fvm_data['Te'],
        # Config scalars
        config_R0=cfg.R0, config_a=cfg.a,
        config_B0=cfg.B0, config_kappa=cfg.kappa,
        config_Nr=cfg.Nr, config_Ntheta=cfg.Ntheta, config_Nphi=cfg.Nphi,
    )

    size_MB = sum(arr.nbytes for arr in [
        mesh['X'], density_3d, temp_3d, pressure_3d, B_mag_3d,
        current_3d, part_X, part_Y, part_Z
    ]) / 1e6
    print(f"  [EXPORT] Done — ~{size_MB:.0f} MB uncompressed data")
