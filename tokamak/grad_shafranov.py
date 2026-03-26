"""
Grad-Shafranov equilibrium solver for tokamak plasmas.

Solves the Grad-Shafranov (GS) equation on a 2D (R, Z) grid:

    Δ*ψ ≡ R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z² = -μ₀ R² p'(ψ) - F(ψ) F'(ψ)

where:
    ψ   = poloidal flux [Wb]
    p   = plasma pressure profile [Pa]
    F   = R·Bφ = toroidal field function [T·m]

Uses the Shafranov-Soloviev analytic family as the first-guess and source
profiles, then iterates via Picard fixed-point iteration with a scipy sparse
direct solver for the elliptic sub-problem.

Boundary condition: ψ = 0 on the boundary of the rectangular domain
(Dirichlet, representing zero flux at the LCFS).

Reference:
    Shafranov, V.D. (1966), Reviews of Plasma Physics, Vol. 2, p. 103.
    Soloviev, L.S. (1968), Soviet Physics JETP 26, 400.
    Freidberg, J.P. (2014), Ideal MHD, Cambridge University Press.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ─────────────────────────────────────────────────────────────────────────────
#  Soloviev analytic profiles  (linear in ψ  →  exact GS solution exists)
# ─────────────────────────────────────────────────────────────────────────────

def _soloviev_analytic(R, Z, R0, a, kappa, delta_tri, B0, mu0, Ip):
    """
    Compute the Soloviev analytic flux function.

    The Soloviev family gives an exact solution for ψ in the form:

        ψ_S(R,Z) = c1*(R²-R0²)/2 + c2*(R⁴/8 - R0²*R²/2 + ...) + c3*Z²

    We use the standard ITER-style formulation from Freidberg (2014) §7.5
    which fits an up-down symmetric D-shaped equilibrium.

    Returns ψ on the (R, Z) meshgrid.  The LCFS is the contour ψ = ψ_lcfs
    passing through (R0+a, 0), (R0-a*δ, ±κa).
    """
    # Elongation and triangularity parameters after Cerfon & Freidberg (2010)
    # Phys. Plasmas 17, 112502, Eq.(9)-(16)
    eps = a / R0           # inverse aspect ratio
    N1 = -(1.0 + np.arcsin(delta_tri))**2 / (eps * kappa**2)  # noqa: F841 (kept for reference)

    # Basis functions ψ_1 … ψ_4  (Cerfon & Freidberg Eq.(7))
    x = R / R0   # normalised major radius

    psi1 = 1.0
    psi2 = x**2
    psi3 = x**2 * np.log(x) - (Z / R0)**2
    psi4 = x**4 - 4.0 * x**2 * (Z / R0)**2

    # For an up-down symmetric plasma without current in the X-point region
    # we set the 4-parameter family with the particular solution for the
    # Soloviev source.
    #   ψ_particular = R⁴/8  (source ~ A R²  with no Z-dependent term)
    psi_particular = R**4 / 8.0

    # Boundary conditions: the LCFS passes through the three points
    #   P1 = (R0+a,  0)        outer equatorial point
    #   P2 = (R0-a,  0)        inner equatorial point
    #   P3 = (R0-δa, +κa)      upper X-point / top
    # and the curvature condition at P1 expressing kappa.
    # This is the 4-unknown system (c1,c2,c3,c4) solved by 4 BCs.

    def eval_basis(R_, Z_):
        x_ = R_ / R0
        b1 = 1.0
        b2 = x_**2
        b3 = x_**2 * np.log(x_) - (Z_ / R0)**2
        b4 = x_**4 - 4.0 * x_**2 * (Z_ / R0)**2
        bp = R_**4 / 8.0
        return b1, b2, b3, b4, bp

    R1, Z1 = R0 + a, 0.0
    R2, Z2 = R0 - a, 0.0
    R3, Z3 = R0 - delta_tri * a, kappa * a

    b1_1, b2_1, b3_1, b4_1, bp_1 = eval_basis(R1, Z1)
    b1_2, b2_2, b3_2, b4_2, bp_2 = eval_basis(R2, Z2)
    b1_3, b2_3, b3_3, b4_3, bp_3 = eval_basis(R3, Z3)

    # Second derivative ∂²ψ_i/∂Z² at the outer equatorial midplane (Z=0):
    # used for the curvature BC that sets the elongation at the outer point.
    # For a D-shape: ∂²ψ/∂Z²|_{P1} = -1/(N1 * ψ0)  with ψ0=1 normalisation.
    # We follow Freidberg (2014) notation: set coefficient c1=1 and solve for c2,c3,c4.
    def ddzz_basis(R_, Z_=0.0):
        x_ = R_ / R0
        dd1 = 0.0
        dd2 = 0.0
        dd3 = -2.0 / R0**2
        dd4 = -8.0 * x_**2 / R0**2
        ddp = 0.0
        return dd1, dd2, dd3, dd4, ddp

    dz1_1, dz2_1, dz3_1, dz4_1, dzp_1 = ddzz_basis(R1)

    # System:  M * [c1, c2, c3, c4] = rhs
    #
    # BC1:  ψ(R1, 0) = 0  →  c1 b1_1 + c2 b2_1 + c3 b3_1 + c4 b4_1 + bp_1 = 0
    # BC2:  ψ(R2, 0) = 0  →  (same with R2)
    # BC3:  ψ(R3, Z3) = 0
    # BC4:  ∂²ψ/∂Z²(R1,0) = 0  (elongation curvature BC, since ψ=0 there)
    #        so  c1 dz1_1 + c2 dz2_1 + c3 dz3_1 + c4 dz4_1 + dzp_1 = 0
    M = np.array([
        [b1_1,  b2_1,  b3_1,  b4_1 ],
        [b1_2,  b2_2,  b3_2,  b4_2 ],
        [b1_3,  b2_3,  b3_3,  b4_3 ],
        [dz1_1, dz2_1, dz3_1, dz4_1],
    ])
    rhs_sys = np.array([-bp_1, -bp_2, -bp_3, -dzp_1])

    try:
        coeffs = np.linalg.solve(M, rhs_sys)
    except np.linalg.LinAlgError:
        # fallback: uniform-current cylinder
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])

    c1, c2, c3, c4 = coeffs

    psi = (c1 * psi1 + c2 * psi2 + c3 * psi3 + c4 * psi4
           + psi_particular)

    return psi


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-Shafranov operator  Δ*
# ─────────────────────────────────────────────────────────────────────────────

def _build_gs_operator(R_1d, Z_1d):
    """
    Assemble the sparse matrix for the GS operator Δ* on a uniform (R,Z) grid.

    Δ*ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²
         = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²

    Interior finite-difference stencil (second-order centred differences):

        Δ*ψ_{i,j} ≈
            (ψ_{i+1,j} - 2ψ_{i,j} + ψ_{i-1,j}) / dR²
          - (ψ_{i+1,j} - ψ_{i-1,j}) / (2R_i dR)
          + (ψ_{i,j+1} - 2ψ_{i,j} + ψ_{i,j-1}) / dZ²

    Dirichlet ψ=0 on all four edges.

    Returns:
        A  — sparse (NR*NZ) x (NR*NZ) matrix
        idx — mapping from 2D (i,j) → 1D index (row of A)
    """
    NR = len(R_1d)
    NZ = len(Z_1d)
    dR = R_1d[1] - R_1d[0]
    dZ = Z_1d[1] - Z_1d[0]

    N = NR * NZ

    def ij2k(i, j):
        return i * NZ + j

    rows, cols, vals = [], [], []

    for i in range(NR):
        R = R_1d[i]
        for j in range(NZ):
            k = ij2k(i, j)

            if i == 0 or i == NR - 1 or j == 0 or j == NZ - 1:
                # Dirichlet BC: ψ = 0
                rows.append(k); cols.append(k); vals.append(1.0)
            else:
                # Interior point — GS stencil
                # Coefficients
                aR_fwd = 1.0 / dR**2 - 1.0 / (2.0 * R * dR)
                aR_bwd = 1.0 / dR**2 + 1.0 / (2.0 * R * dR)
                aZ     = 1.0 / dZ**2
                aCenter = -2.0 / dR**2 - 2.0 / dZ**2

                rows.append(k); cols.append(ij2k(i+1, j)); vals.append(aR_fwd)
                rows.append(k); cols.append(ij2k(i-1, j)); vals.append(aR_bwd)
                rows.append(k); cols.append(ij2k(i, j+1)); vals.append(aZ)
                rows.append(k); cols.append(ij2k(i, j-1)); vals.append(aZ)
                rows.append(k); cols.append(k);             vals.append(aCenter)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A


# ─────────────────────────────────────────────────────────────────────────────
#  Profile functions  (expressed as functions of normalised ψ)
# ─────────────────────────────────────────────────────────────────────────────

def _make_profiles(cfg, psi_axis, psi_lcfs):
    """
    Return callables p(ψ), p'(ψ), F(ψ), F'(ψ) for the Picard RHS.

    We use simple polynomial profiles parameterised by normalised flux
        ψ_N = (ψ - ψ_axis) / (ψ_lcfs - ψ_axis)  ∈ [0,1]

    Pressure:  p(ψ_N) = p0 (1 - ψ_N)^α_p
    p'(ψ)   = dp/dψ_N * dψ_N/dψ = -α_p p0 (1-ψ_N)^(α_p-1) / (ψ_lcfs-ψ_axis)

    Toroidal field function:
        F(ψ) = R0 B0 [1 + f1 (1-ψ_N)]^0.5
    F F'(ψ) = d(F²/2)/dψ  = -f1 R0² B0² / (2 (ψ_lcfs-ψ_axis)) (for small f1)

    We keep things simple: choose α_p=2 and determine f1 from Ip.
    """
    mu0    = cfg.mu0
    R0     = cfg.R0
    B0     = cfg.B0
    n0     = cfg.n0
    T0_i   = cfg.T0_i
    T0_e   = cfg.T0_e
    keV_to_J = cfg.keV_to_J

    # Central pressure
    p0 = n0 * (T0_i + T0_e) * keV_to_J

    # Profile exponent
    alpha_p = 2.0

    # ψ normalisation scale  (handle sign)
    dpsi = psi_lcfs - psi_axis   # may be positive or negative
    if abs(dpsi) < 1.0e-30:
        dpsi = 1.0e-30

    def psi_n(psi):
        """Clip to [0,1] to avoid issues outside the LCFS."""
        return np.clip((psi - psi_axis) / dpsi, 0.0, 1.0)

    def p_prime(psi):
        """dp/dψ  (Pa/Wb)."""
        psn = psi_n(psi)
        return -alpha_p * p0 * (1.0 - psn)**(alpha_p - 1.0) / dpsi

    # Toroidal function: F² = (R0 B0)² + 2 μ₀ f1_param * (ψ - ψ_axis)
    # This is the linear-in-ψ Soloviev choice: F F' = const = -μ₀ C
    # We calibrate C so that the volume-averaged current matches Ip.
    # A rough estimate: C ≈ mu0 * Ip / (2π * R0 * a^2 * kappa)
    a     = cfg.a
    kappa = cfg.kappa
    C_param = mu0 * cfg.Ip / (2.0 * np.pi * R0 * a**2 * kappa)

    def FF_prime(psi):
        """F dF/dψ  (T²·m²/Wb)."""
        return -C_param * np.ones_like(psi)

    return p_prime, FF_prime


# ─────────────────────────────────────────────────────────────────────────────
#  Safety-factor estimate from the GS solution
# ─────────────────────────────────────────────────────────────────────────────

def _compute_q_profile(psi_2d, R_2d, Z_2d, R_1d, Z_1d,
                       psi_axis, psi_lcfs, F0, Nr_q=20):
    """
    Estimate q(ψ) using the flux-surface averaging formula

        q(ψ) = 1/(2π) ∮ F/(R²) dl / |∇ψ|

    approximated by summing along closed flux-surface contours.

    Returns arrays (psi_vals, q_vals) for Nr_q surfaces.
    """
    dR = R_1d[1] - R_1d[0]
    dZ = Z_1d[1] - Z_1d[0]

    # ∇ψ on the grid (central differences, interior only)
    dpsi_dR = np.gradient(psi_2d, dR, axis=0)
    dpsi_dZ = np.gradient(psi_2d, dZ, axis=1)
    grad_psi_sq = dpsi_dR**2 + dpsi_dZ**2
    grad_psi_sq = np.maximum(grad_psi_sq, 1.0e-20)

    psi_vals = np.linspace(psi_axis, psi_lcfs, Nr_q + 2)[1:-1]
    q_vals   = np.zeros(Nr_q)

    # Pre-compute loop-invariant quantities
    dpsi_band = 0.5 * abs(psi_lcfs - psi_axis) / Nr_q
    integrand = 1.0 / (R_2d**2 * np.sqrt(grad_psi_sq))
    scale = (F0 / (2.0 * np.pi)) * dR * dZ / (dpsi_band * 2.0)

    for idx, psi_s in enumerate(psi_vals):
        # Integrate F / (R² |∇ψ|) dA  over the annulus dψ thick
        # q = F/(2π) * ∮ dl/|∇ψ| / R²
        # We use the area-integration form:  q = (F/2π) * ∫∫ δ(ψ-ψ_s)/(R² |∇ψ|) dR dZ
        # Approximated by counting cells with |ψ - ψ_s| < dpsi_band
        mask = np.abs(psi_2d - psi_s) < dpsi_band
        if mask.sum() < 4:
            q_vals[idx] = np.nan
            continue

        # Weight by 1/|∇ψ| to approximate dl
        q_vals[idx] = scale * np.sum(integrand[mask])

    # Fill NaN with linear interpolation
    valid = np.isfinite(q_vals)
    if valid.sum() >= 2:
        q_vals = np.where(valid, q_vals,
                          np.interp(psi_vals, psi_vals[valid], q_vals[valid]))

    return psi_vals, q_vals


# ─────────────────────────────────────────────────────────────────────────────
#  Shafranov shift from the GS solution
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shafranov_shift(psi_2d, R_1d, Z_1d, psi_axis, psi_lcfs, R0):
    """
    Estimate the Shafranov shift Δ(ψ) = R_center(ψ) - R0  for several flux surfaces.

    R_center(ψ) is the R-coordinate of the flux-surface centroid on the midplane.
    """
    Z_mid_idx = len(Z_1d) // 2
    psi_mid = psi_2d[:, Z_mid_idx]   # psi along midplane (Z=0)

    Nr_s = 10
    psi_s_vals = np.linspace(psi_axis, psi_lcfs, Nr_s + 2)[1:-1]
    shift_vals = np.zeros(Nr_s)

    for idx, psi_s in enumerate(psi_s_vals):
        # Find the two R locations where psi_mid = psi_s (inner + outer)
        diff = psi_mid - psi_s
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) >= 2:
            # Interpolate each crossing
            i1, i2 = crossings[0], crossings[-1]
            f1 = -diff[i1] / (diff[i1 + 1] - diff[i1])
            f2 = -diff[i2] / (diff[i2 + 1] - diff[i2])
            R_in  = R_1d[i1]  + f1 * (R_1d[i1 + 1]  - R_1d[i1])
            R_out = R_1d[i2]  + f2 * (R_1d[i2 + 1]  - R_1d[i2])
            shift_vals[idx] = 0.5 * (R_in + R_out) - R0
        else:
            shift_vals[idx] = np.nan

    valid = np.isfinite(shift_vals)
    if valid.sum() < 2:
        shift_vals[:] = 0.0
    else:
        shift_vals = np.where(valid, shift_vals,
                              np.interp(psi_s_vals,
                                        psi_s_vals[valid],
                                        shift_vals[valid]))

    return psi_s_vals, shift_vals


# ─────────────────────────────────────────────────────────────────────────────
#  Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_gs(cfg):
    """
    Solve the Grad-Shafranov equation for an ITER-like equilibrium.

    Parameters
    ----------
    cfg : TokamakConfig
        Simulation configuration object (see tokamak/config.py).

    Returns
    -------
    dict with keys:
        psi_axis      — ψ at the magnetic axis [Wb]
        psi_lcfs      — ψ at the last closed flux surface [Wb]
        psi_grid      — 2D array, shape (NR, NZ) [Wb]
        R_grid        — 2D R meshgrid [m]
        Z_grid        — 2D Z meshgrid [m]
        R_1d          — 1D R array [m]
        Z_1d          — 1D Z array [m]
        q_profile_gs  — dict with 'psi' and 'q' arrays
        shafranov_shift — dict with 'psi' and 'shift' arrays [m]
        converged     — bool
        n_iter        — int, iterations performed
    """
    R0      = cfg.R0
    a       = cfg.a
    kappa   = cfg.kappa
    delta_tri = cfg.delta_tri
    B0      = cfg.B0
    Ip      = cfg.Ip
    mu0     = cfg.mu0
    Nr      = cfg.Nr
    Nz      = cfg.Ntheta   # use Ntheta as Nz for the (R,Z) grid

    # ── Grid ──
    R_min = R0 - a
    R_max = R0 + a
    Z_min = -kappa * a
    Z_max =  kappa * a

    R_1d = np.linspace(R_min, R_max, Nr)
    Z_1d = np.linspace(Z_min, Z_max, Nz)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing='ij')

    print(f"  [GS] Grid: {Nr}×{Nz}, R=[{R_min:.2f}, {R_max:.2f}] m, "
          f"Z=[{Z_min:.2f}, {Z_max:.2f}] m")

    # ── Initial guess: Soloviev analytic solution ──
    psi_old = _soloviev_analytic(R_2d, Z_2d, R0, a, kappa, delta_tri,
                                 B0, mu0, Ip)

    # Locate axis (extremum) and LCFS value
    # The axis is roughly at (R0, 0); the boundary value is ψ at (R0+a, 0)
    i_axis = np.argmin(np.abs(R_1d - R0))
    j_axis = len(Z_1d) // 2

    # LCFS: value at the outer equatorial point
    i_edge = np.argmin(np.abs(R_1d - (R0 + a)))
    psi_lcfs_val = psi_old[i_edge, j_axis]

    # Normalise so that ψ_lcfs = 0  (Dirichlet BC at boundary)
    psi_old = psi_old - psi_lcfs_val
    psi_axis_cur = psi_old[i_axis, j_axis]

    print(f"  [GS] Initial guess: ψ_axis={psi_axis_cur:.4f} Wb, ψ_lcfs=0.0000 Wb")

    # ── Build the sparse GS operator (does not change between iterations) ──
    A_op = _build_gs_operator(R_1d, Z_1d)

    # ── Picard iteration ──
    max_iter    = 50
    tol         = 1.0e-4   # relative change tolerance
    omega       = 0.7      # under-relaxation factor (improves stability)
    converged   = False
    n_iter      = 0
    rel_change  = 1.0      # initialise before loop for use in warning message

    # F0 = R0 * B0  (value on axis)
    F0 = R0 * B0

    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        # Current axis and LCFS values
        psi_axis_cur = psi_old[i_axis, j_axis]
        psi_lcfs_cur = 0.0   # Dirichlet

        # Build source profiles from current ψ
        p_prime, FF_prime = _make_profiles(cfg, psi_axis_cur, psi_lcfs_cur)

        # RHS of GS equation:  rhs = -μ₀ R² p'(ψ) - F F'(ψ)
        rhs_2d = (-mu0 * R_2d**2 * p_prime(psi_old)
                  - FF_prime(psi_old))

        # Apply Dirichlet BC: zero on all boundary rows
        rhs_2d[ 0,  :] = 0.0
        rhs_2d[-1,  :] = 0.0
        rhs_2d[:,  0 ] = 0.0
        rhs_2d[:, -1 ] = 0.0

        # Flatten and solve
        rhs_flat = rhs_2d.ravel()
        psi_new_flat = spsolve(A_op, rhs_flat)
        psi_new = psi_new_flat.reshape(Nr, Nz)

        # Enforce Dirichlet BC explicitly (spsolve may drift)
        psi_new[ 0,  :] = 0.0
        psi_new[-1,  :] = 0.0
        psi_new[:,  0 ] = 0.0
        psi_new[:, -1 ] = 0.0

        # Under-relaxed update
        psi_blended = omega * psi_new + (1.0 - omega) * psi_old

        # Convergence check: relative change in ψ_axis
        delta_axis = abs(psi_blended[i_axis, j_axis] - psi_axis_cur)
        rel_change = delta_axis / (abs(psi_axis_cur) + 1.0e-30)

        if iteration <= 5 or iteration % 10 == 0:
            print(f"  [GS] iter {iteration:3d}:  ψ_axis={psi_blended[i_axis, j_axis]:.4f},  "
                  f"Δ(rel)={rel_change:.2e}")

        psi_old = psi_blended

        if rel_change < tol and iteration >= 5:
            converged = True
            print(f"  [GS] Converged at iteration {iteration}  "
                  f"(rel_change={rel_change:.2e} < {tol:.0e})")
            break

    if not converged:
        print(f"  [GS] Warning: did not converge after {max_iter} iterations "
              f"(final Δ_rel={rel_change:.2e})")

    psi_final   = psi_old
    psi_axis_f  = float(psi_final[i_axis, j_axis])
    psi_lcfs_f  = 0.0   # by construction (Dirichlet)

    # ── Derived geometry: q profile and Shafranov shift ──
    q_psi, q_vals = _compute_q_profile(
        psi_final, R_2d, Z_2d, R_1d, Z_1d,
        psi_axis_f, psi_lcfs_f, F0
    )

    sh_psi, sh_vals = _compute_shafranov_shift(
        psi_final, R_1d, Z_1d, psi_axis_f, psi_lcfs_f, R0
    )

    return {
        'psi_axis'          : psi_axis_f,
        'psi_lcfs'          : psi_lcfs_f,
        'psi_grid'          : psi_final,
        'R_grid'            : R_2d,
        'Z_grid'            : Z_2d,
        'R_1d'              : R_1d,
        'Z_1d'              : Z_1d,
        'q_profile_gs'      : {'psi': q_psi,  'q': q_vals},
        'shafranov_shift'   : {'psi': sh_psi, 'shift': sh_vals},
        'converged'         : converged,
        'n_iter'            : n_iter,
    }
