"""
Main entry point — Tokamak Plasma Transport Simulation.

Usage:
    python -m tokamak             # auto-detect CUDA
    python -m tokamak --backend cpu
    python -m tokamak --backend cuda
    python -m tokamak --fvm2d     # 2D poloidal transport
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokamak.config import TokamakConfig
from tokamak.backend import init_backend
from tokamak.grids import build_grids
from tokamak.pic import init_particles, pic_push_particles, deposit_to_grid
from tokamak.fvm import run_fvm_transport
from tokamak.poisson import solve_poisson_2d
from tokamak.current import compute_current_density
from tokamak.export3d import export_3d_animation_data
from tokamak.profiling import Profiler
from tokamak.visualization import (
    plot_poloidal_contours, plot_axial_contours,
    plot_radial_profiles, plot_particle_orbits,
    plot_multiple_toroidal_sections, plot_transport_hierarchy,
)


def main():
    parser = argparse.ArgumentParser(description='Tokamak Plasma Transport Simulation')
    parser.add_argument('--backend', choices=['cuda', 'cpu', 'auto'],
                        default='auto', help='Compute backend (default: auto)')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip 3D data export')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip diagnostic plots')
    parser.add_argument('--no-profile', action='store_true',
                        help='Skip profiling report')
    parser.add_argument('--fvm2d', action='store_true',
                        help='Use 2D poloidal FVM transport')
    args = parser.parse_args()

    cfg = TokamakConfig()
    prof = Profiler()

    print("=" * 65)
    print("  TOKAMAK PLASMA TRANSPORT SIMULATION")
    print("  PIC + FVM · CUDA-accelerated · ITER-like parameters")
    print("=" * 65)
    print()

    # ── Backend ──
    init_backend(args.backend)
    print()

    # ── Step 0: Grad-Shafranov Equilibrium (if available) ──
    gs_data = None
    try:
        from tokamak.grad_shafranov import solve_gs
        print("━━━ PHASE 0: Grad-Shafranov Equilibrium ━━━")
        with prof.phase("Grad-Shafranov Solve"):
            gs_data = solve_gs(cfg)
        print(f"  [GS] Converged. ψ_axis={gs_data['psi_axis']:.4f}, "
              f"ψ_lcfs={gs_data['psi_lcfs']:.4f}")
        print()
    except ImportError:
        pass

    # ── Step 1: FVM Transport ──
    print("━━━ PHASE 1: Finite Volume Transport Solver ━━━")
    if args.fvm2d:
        try:
            from tokamak.fvm2d import run_fvm2d_transport
            with prof.phase("FVM 2D Transport"):
                fvm_data = run_fvm2d_transport(cfg, gs_data=gs_data)
        except ImportError:
            print("  [WARN] fvm2d module not found, falling back to 1D FVM")
            with prof.phase("FVM 1D Transport"):
                fvm_data = run_fvm_transport(cfg)
    else:
        with prof.phase("FVM 1D Transport"):
            fvm_data = run_fvm_transport(cfg)
    print()

    # ── Step 2: Derived quantities ──
    print("━━━ PHASE 2: Equilibrium Derived Quantities ━━━")
    with prof.phase("Current + Poisson"):
        print("  [CALC] Current density profiles...")
        j_phi, j_bs = compute_current_density(
            fvm_data['rho'], fvm_data['n'], fvm_data['Ti'], fvm_data['Te'], cfg)

        print("  [CALC] Poisson solver for electrostatic potential...")
        rho_grid, theta_grid, dr, dtheta, RHO, THETA = build_grids(cfg)
        pic_density = deposit_to_grid(
            np.sqrt(np.random.random(10000)) * 0.95,
            np.random.random(10000) * 2 * np.pi,
            cfg.Nr, cfg.Ntheta, cfg.a
        )
        phi_2d, phi_1d = solve_poisson_2d(pic_density, rho_grid, theta_grid,
                                           dr, dtheta, cfg)
    print()

    # ── Step 3: PIC Particle Orbits ──
    print("━━━ PHASE 3: Particle-In-Cell Orbit Integration ━━━")
    print(f"  [PIC] Initializing {cfg.Nparticles:,} guiding-center particles...")

    with prof.phase("PIC Orbit Integration"):
        rho_p, theta_p, phi_p, vpar, vperp, mu_p = init_particles(
            cfg.Nparticles, cfg.R0, cfg.a, cfg.n0, cfg.T0_i, cfg.mi,
            cfg.q0, cfg.qa, cfg.B0, cfg.Ip, cfg.mu0
        )
        print(f"  [PIC] Pushing particles for {cfg.Nt_pic} steps "
              f"(dt={cfg.dt:.1e}s, RK4)...")
        rho_p, theta_p, phi_p, vpar, vperp, rho_hist, theta_hist, phi_hist = \
            pic_push_particles(rho_p, theta_p, phi_p, vpar, vperp, mu_p,
                               cfg.dt, cfg.Nt_pic, cfg.R0, cfg.a, cfg.B0,
                               cfg.mi, cfg.q0, cfg.qa, cfg.kappa,
                               cfg.Ip, cfg.mu0,
                               N_TF=cfg.N_TF, ripple_amplitude=0.001)
    print(f"  [PIC] Complete.")
    print()

    # ── Step 4: Final density deposition ──
    print("━━━ PHASE 4: Charge Deposition (CIC) ━━━")
    with prof.phase("Charge Deposition"):
        final_density = deposit_to_grid(rho_p, theta_p, cfg.Nr, cfg.Ntheta, cfg.a)
    print(f"  Peak PIC density: {final_density.max():.2f} (normalized)")
    print()

    # ── Step 5: 3D Data Export ──
    if not args.no_export:
        print("━━━ PHASE 5: 3D Animation Data Export ━━━")
        with prof.phase("3D Export"):
            export_3d_animation_data(
                fvm_data, rho_hist, theta_hist, phi_hist,
                j_phi, j_bs, cfg,
                output_path='tokamak_3d_data.npz'
            )
        print()

    # ── Step 6: Plotting ──
    if not args.no_plots:
        print("━━━ PHASE 6: Generating Diagnostic Plots ━━━")
        with prof.phase("Diagnostic Plots"):
            plot_poloidal_contours(fvm_data, phi_2d, j_phi, j_bs, cfg,
                                   '01_poloidal_contours.png')
            plot_axial_contours(fvm_data, cfg, '02_axial_contours.png')
            plot_radial_profiles(fvm_data, j_phi, j_bs, cfg,
                                 '03_radial_profiles.png')
            plot_particle_orbits(rho_hist, theta_hist, phi_hist, cfg,
                                 '04_particle_orbits.png')
            plot_multiple_toroidal_sections(fvm_data, cfg,
                                            '05_toroidal_sections.png')
            plot_transport_hierarchy(fvm_data, cfg, '06_transport_hierarchy.png')

    # ── Profiling Report ──
    if not args.no_profile:
        prof.report()

    total_wall = sum(p['wall'] for p in prof.phases)
    print()
    print("=" * 65)
    print(f"  SIMULATION COMPLETE — Total time: {total_wall:.1f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
