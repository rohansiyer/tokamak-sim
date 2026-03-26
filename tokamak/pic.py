"""
Particle-In-Cell (PIC) module — guiding-center orbit integration.

Supports both Numba CPU and CUDA backends.
"""

import numpy as np
from numba import njit, prange

from tokamak.config import TokamakConfig
from tokamak.backend import is_cuda, to_host, to_device


# ═══════════════════════════════════════════════════════════════
#  Particle initialisation
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def init_particles(Np, R0, a, n0, T0_i, mi, q0, qa, B0, Ip, mu0):
    """
    Initialise guiding-center particles with Maxwellian distribution.
    Positions sampled uniformly in volume (√ρ for area weighting).
    """
    rho   = np.sqrt(np.random.random(Np)) * 0.95
    theta = np.random.random(Np) * 2.0 * np.pi
    phi   = np.random.random(Np) * 2.0 * np.pi

    v_par  = np.empty(Np)
    v_perp = np.empty(Np)
    mu     = np.empty(Np)

    e = 1.602176634e-19
    keV_to_J = 1.602176634e-16

    for i in range(Np):
        T_local = T0_i * max(1.0 - rho[i]**2, 1e-4)**1.5
        vth = np.sqrt(2.0 * T_local * keV_to_J / mi)

        v_par[i]  = np.random.randn() * vth
        v_perp[i] = np.abs(np.random.randn() * vth)

        R_loc = R0 + a * rho[i] * np.cos(theta[i])
        B_loc = B0 * R0 / max(R_loc, 0.1)
        mu[i] = mi * v_perp[i]**2 / (2.0 * B_loc)

    return rho, theta, phi, v_par, v_perp, mu


# ═══════════════════════════════════════════════════════════════
#  RK4 guiding-center orbit integrator (CPU via Numba)
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _gc_derivatives(rho_i, theta_i, phi_i, v_par_i, mu_i,
                    R0, a, B0, mi, q0, qa, Ip, mu0, keV_to_J):
    """
    Compute time derivatives for guiding-center motion.
    Returns (drho/dt, dtheta/dt, dphi/dt, dv_par/dt).
    """
    e = 1.602176634e-19

    r     = rho_i * a
    cos_th = np.cos(theta_i)
    sin_th = np.sin(theta_i)
    R      = R0 + r * cos_th
    R      = max(R, 0.5)

    # Magnetic field
    Bt = B0 * R0 / R
    q  = q0 + (qa - q0) * rho_i**2
    # Bp from enclosed current
    I_enc = Ip * rho_i**2
    Bp = mu0 * I_enc / (2.0 * np.pi * max(r, 1e-6))
    B_mag = np.sqrt(Bt**2 + Bp**2)

    # v_perp from μ conservation
    v_perp = np.sqrt(max(2.0 * mu_i * B_mag / mi, 0.0))

    # ── Drift velocities ──

    # ∇B drift:  v_∇B = (m v⊥²)/(2eB²) × |∇B|
    grad_B_over_B = 1.0 / R
    v_gradB = mi * v_perp**2 / (2.0 * e * B_mag) * grad_B_over_B

    # Curvature drift: v_κ = m v∥² / (e B R)
    v_curv = mi * v_par_i**2 / (e * B_mag * R)

    # Combined vertical drift (Z-direction → poloidal motion)
    v_drift_Z = v_gradB + v_curv

    # E×B drift from ambipolar field Er = -(T/e) d ln(n)/dr
    n_loc = 1.0e20 * max(1.0 - rho_i**2, 1e-4)**0.5
    T_loc = 15.0 * max(1.0 - rho_i**2, 1e-4)**1.5
    dp_dr = -2.0 * rho_i / a * n_loc * T_loc * keV_to_J * 2.0
    Er = -dp_dr / (n_loc * e) if n_loc > 0 else 0.0
    v_ExB = Er / B_mag

    # ── Equations of motion ──

    # Toroidal advance: dφ/dt = v∥ Bt / (B R)
    dphi = v_par_i * Bt / (B_mag * R)

    # Poloidal advance
    dtheta = 0.0
    if r > 1e-4:
        dtheta = (v_par_i * Bp / (B_mag * r)
                  + v_ExB / r
                  + v_drift_Z * cos_th / r)

    # Radial drift from vertical drifts
    drho = v_drift_Z * sin_th / a

    # Mirror force: dv∥/dt = -(μ/m) ∂B/∂s
    dv_par = -mu_i / mi * (-B_mag * sin_th / R)

    # Collisional pitch-angle scattering (simplified Lorentz operator)
    # ν_D ≈ ν_ii ≈ n Z⁴ e⁴ lnΛ / (4π ε₀² m² v_th³)
    vth = np.sqrt(max(2.0 * T_loc * keV_to_J / mi, 1e-10))
    nu_D = n_loc * e**4 * 17.0 / (
        4.0 * np.pi * (8.854e-12)**2 * mi**2 * max(vth, 1e-3)**3)
    nu_D = min(nu_D, 1e6)  # cap for numerical stability

    return drho, dtheta, dphi, dv_par, nu_D


@njit(parallel=True, cache=True)
def pic_push_particles(rho, theta, phi, v_par, v_perp, mu,
                       dt, Nt, R0, a, B0, mi, q0, qa, kappa,
                       Ip, mu0):
    """
    RK4 guiding-center orbit integrator with collisional scattering.
    """
    Np = len(rho)
    e = 1.602176634e-19
    keV_to_J = 1.602176634e-16

    Nsave = min(Np, 500)
    save_interval = max(1, Nt // 100)
    Nsnapshots = Nt // save_interval + 1
    rho_hist   = np.zeros((Nsave, Nsnapshots))
    theta_hist = np.zeros((Nsave, Nsnapshots))
    phi_hist   = np.zeros((Nsave, Nsnapshots))
    snap_idx = 0

    for step in range(Nt):
        for i in prange(Np):
            rho_i   = rho[i]
            theta_i = theta[i]
            phi_i   = phi[i]
            vp_i    = v_par[i]
            mu_i    = mu[i]

            # ── RK4 stages ──
            # k1
            dr1, dt1, dp1, dv1, nu_D = _gc_derivatives(
                rho_i, theta_i, phi_i, vp_i, mu_i,
                R0, a, B0, mi, q0, qa, Ip, mu0, keV_to_J)

            # k2
            dr2, dt2, dp2, dv2, _ = _gc_derivatives(
                rho_i + 0.5*dt*dr1, theta_i + 0.5*dt*dt1,
                phi_i + 0.5*dt*dp1, vp_i + 0.5*dt*dv1, mu_i,
                R0, a, B0, mi, q0, qa, Ip, mu0, keV_to_J)

            # k3
            dr3, dt3, dp3, dv3, _ = _gc_derivatives(
                rho_i + 0.5*dt*dr2, theta_i + 0.5*dt*dt2,
                phi_i + 0.5*dt*dp2, vp_i + 0.5*dt*dv2, mu_i,
                R0, a, B0, mi, q0, qa, Ip, mu0, keV_to_J)

            # k4
            dr4, dt4, dp4, dv4, _ = _gc_derivatives(
                rho_i + dt*dr3, theta_i + dt*dt3,
                phi_i + dt*dp3, vp_i + dt*dv3, mu_i,
                R0, a, B0, mi, q0, qa, Ip, mu0, keV_to_J)

            # RK4 update
            rho[i]   += dt/6.0 * (dr1 + 2*dr2 + 2*dr3 + dr4)
            theta[i] += dt/6.0 * (dt1 + 2*dt2 + 2*dt3 + dt4)
            phi[i]   += dt/6.0 * (dp1 + 2*dp2 + 2*dp3 + dp4)
            v_par[i] += dt/6.0 * (dv1 + 2*dv2 + 2*dv3 + dv4)

            # Collisional pitch-angle scattering (Monte Carlo)
            if nu_D > 0:
                dv_coll = np.sqrt(nu_D * dt) * v_par[i] * np.random.randn() * 0.01
                v_par[i] += dv_coll

            # Update v_perp from μ conservation
            R_loc = R0 + a * rho[i] * np.cos(theta[i])
            R_loc = max(R_loc, 0.5)
            Bt = B0 * R0 / R_loc
            q  = q0 + (qa - q0) * rho[i]**2
            r_loc = rho[i] * a
            I_enc = Ip * rho[i]**2
            Bp = mu0 * I_enc / (2.0 * np.pi * max(r_loc, 1e-6))
            B_mag = np.sqrt(Bt**2 + Bp**2)
            v_perp[i] = np.sqrt(max(2.0 * mu[i] * B_mag / mi, 0.0))

            # Reflecting boundary conditions
            if rho[i] < 0.001:
                rho[i] = 0.001
                v_par[i] = abs(v_par[i]) * 0.8
            if rho[i] > 0.99:
                rho[i] = 0.99
                v_par[i] = -abs(v_par[i]) * 0.8

            # Periodicity
            theta[i] = theta[i] % (2.0 * np.pi)
            phi[i]   = phi[i] % (2.0 * np.pi)

        # Save snapshots
        if step % save_interval == 0 and snap_idx < Nsnapshots:
            for k in range(Nsave):
                rho_hist[k, snap_idx]   = rho[k]
                theta_hist[k, snap_idx] = theta[k]
                phi_hist[k, snap_idx]   = phi[k]
            snap_idx += 1

    return rho, theta, phi, v_par, v_perp, rho_hist, theta_hist, phi_hist


# ═══════════════════════════════════════════════════════════════
#  CIC density deposition
# ═══════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def deposit_to_grid(rho_p, theta_p, Nr, Ntheta, a):
    """CIC (Cloud-In-Cell) charge/density deposition onto (ρ, θ) grid."""
    density = np.zeros((Nr, Ntheta))
    dr  = 1.0 / (Nr - 1)
    dth = 2.0 * np.pi / (Ntheta - 1)
    Np  = len(rho_p)

    for i in prange(Np):
        ri = rho_p[i] / dr
        ti = theta_p[i] / dth

        ir = int(ri)
        it = int(ti)

        if ir < 0 or ir >= Nr - 1 or it < 0 or it >= Ntheta - 1:
            continue

        wr = ri - ir
        wt = ti - it

        density[ir,     it]     += (1 - wr) * (1 - wt)
        density[ir + 1, it]     += wr * (1 - wt)
        density[ir,     it + 1] += (1 - wr) * wt
        density[ir + 1, it + 1] += wr * wt

    # Normalise by cell volume (Jacobian ~ ρ for polar)
    for ir in range(Nr):
        rho_val  = max(ir * dr, 0.01)
        cell_vol = rho_val * dr * dth * a**2
        for it in range(Ntheta):
            density[ir, it] /= max(cell_vol, 1e-10)

    # Scale to physical density
    total = 0.0
    for ir in range(Nr):
        for it in range(Ntheta):
            total += density[ir, it]
    if total > 0:
        density *= (Np / total)

    return density
