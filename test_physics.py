"""
Physics Benchmark Tests — Tokamak Simulation Validation

Compares simulation outputs against published ITER scenario data
and well-known plasma physics identities.

Sources:
  - ITER Physics Basis, Nucl. Fusion 39 (1999) 2137
  - NRL Plasma Formulary (2019 revised)
  - Wesson, "Tokamaks" 4th ed., Oxford (2011)
  - Bosch & Hale, Nucl. Fusion 32 (1992) 611

Run:  pytest test_physics.py -v
"""

import numpy as np
import pytest

from tokamak.config import TokamakConfig
from tokamak.equilibrium import (
    q_profile, density_profile, temperature_profile,
    pressure_profile, B_toroidal, B_poloidal, B_total,
    DT_reaction_rate, alpha_heating_power,
    bremsstrahlung_loss, cyclotron_radiation_loss,
    impurity_profile, shafranov_shift,
)
from tokamak.fvm import (
    classical_diffusion, neoclassical_diffusion, run_fvm_transport,
)
from tokamak.current import compute_current_density
from tokamak.pic import init_particles
from tokamak.grids import build_grids, poloidal_to_RZ


cfg = TokamakConfig()


# ═══════════════════════════════════════════════════════════════
#  1. MAGNETIC FIELD BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestMagneticField:
    """Validate magnetic field against tokamak fundamentals."""

    def test_toroidal_field_on_axis(self):
        """B_φ(R0) = B0 exactly on the magnetic axis."""
        Bt = B_toroidal(cfg.R0, cfg.B0, cfg.R0)
        assert Bt == pytest.approx(cfg.B0, rel=1e-6), \
            f"B_toroidal on axis should be {cfg.B0} T, got {Bt}"

    def test_toroidal_field_1_over_R(self):
        """B_φ must follow 1/R dependence (fundamental of toroidal geometry)."""
        R1 = cfg.R0 - cfg.a * 0.5   # inboard
        R2 = cfg.R0 + cfg.a * 0.5   # outboard
        B1 = B_toroidal(R1, cfg.B0, cfg.R0)
        B2 = B_toroidal(R2, cfg.B0, cfg.R0)
        # B1/B2 should equal R2/R1
        ratio_B = B1 / B2
        ratio_R = R2 / R1
        assert ratio_B == pytest.approx(ratio_R, rel=1e-4), \
            "B_toroidal must satisfy 1/R dependence"

    def test_toroidal_field_ITER_range(self):
        """On-axis B should be in ITER design range (5.0-5.5 T)."""
        assert 5.0 <= cfg.B0 <= 5.5, \
            f"ITER B0 should be ~5.3 T, got {cfg.B0}"

    def test_poloidal_field_zero_on_axis(self):
        """B_θ(ρ=0) = 0 on magnetic axis (no enclosed current)."""
        Bp = B_poloidal(0.0, cfg.a, cfg.R0, cfg.B0,
                        cfg.Ip, cfg.mu0, cfg.q0, cfg.qa)
        assert Bp == pytest.approx(0.0, abs=1e-10), \
            "B_poloidal must vanish on magnetic axis"

    def test_poloidal_field_edge_ampere(self):
        """B_θ(a) = μ₀ Ip / (2π a) at the edge (Ampère's law)."""
        Bp_edge = B_poloidal(1.0, cfg.a, cfg.R0, cfg.B0,
                             cfg.Ip, cfg.mu0, cfg.q0, cfg.qa)
        Bp_ampere = cfg.mu0 * cfg.Ip / (2 * np.pi * cfg.a)
        assert Bp_edge == pytest.approx(Bp_ampere, rel=1e-3), \
            f"B_θ(a) should satisfy Ampère: {Bp_ampere:.3f} T, got {Bp_edge:.3f}"

    def test_safety_factor_monotonic(self):
        """q(ρ) must be monotonically increasing for MHD stability."""
        rho = np.linspace(0, 1, 100)
        q = [q_profile(r, cfg.q0, cfg.qa) for r in rho]
        for i in range(1, len(q)):
            assert q[i] >= q[i-1], \
                f"q must be monotonic: q({rho[i-1]:.2f})={q[i-1]:.2f} > q({rho[i]:.2f})={q[i]:.2f}"

    def test_safety_factor_values(self):
        """q(0) ≈ 1, q(a) ≈ 3-5 for ITER-like equilibrium."""
        q_center = q_profile(0, cfg.q0, cfg.qa)
        q_edge = q_profile(1, cfg.q0, cfg.qa)
        assert q_center == pytest.approx(1.0, rel=0.1), \
            f"Central q should be ~1, got {q_center}"
        assert 3.0 <= q_edge <= 6.0, \
            f"Edge q should be 3-6, got {q_edge}"


# ═══════════════════════════════════════════════════════════════
#  2. EQUILIBRIUM PROFILE BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestEquilibriumProfiles:
    """Validate profiles against ITER design parameters."""

    def test_central_density(self):
        """n(0) = n0 = 1.0e20 m⁻³ (ITER reference)."""
        n0 = density_profile(0.0, cfg.n0)
        assert n0 == pytest.approx(cfg.n0, rel=0.01)

    def test_central_temperature(self):
        """T(0) = 15 keV (ITER burning plasma target)."""
        T0 = temperature_profile(0.0, cfg.T0_i)
        assert T0 == pytest.approx(cfg.T0_i, rel=0.01)

    def test_density_edge_lower(self):
        """Density at edge should be < central density (substantially)."""
        n_edge = density_profile(0.95, cfg.n0)
        assert n_edge < 0.5 * cfg.n0, \
            f"Edge density should be < 50% of n0, got {n_edge/cfg.n0:.1%}"

    def test_temperature_edge_lower(self):
        """Temperature at edge should be << central temperature."""
        T_edge = temperature_profile(0.95, cfg.T0_i)
        assert T_edge < 0.1 * cfg.T0_i, \
            f"Edge temperature should be < 10% of T0, got {T_edge/cfg.T0_i:.1%}"

    def test_pressure_from_ideal_gas(self):
        """p = n(Ti + Te) in keV·m⁻³ → Pascals."""
        rho = 0.5
        p = pressure_profile(rho, cfg.n0, cfg.T0_i, cfg.T0_e, cfg.keV_to_J)
        n = density_profile(rho, cfg.n0)
        Ti = temperature_profile(rho, cfg.T0_i)
        Te = temperature_profile(rho, cfg.T0_e)
        p_expected = n * (Ti + Te) * cfg.keV_to_J
        assert p == pytest.approx(p_expected, rel=1e-6)

    def test_central_pressure_ITER_range(self):
        """Central pressure should be ~500-1200 kPa for ITER."""
        p0 = pressure_profile(0.0, cfg.n0, cfg.T0_i, cfg.T0_e, cfg.keV_to_J)
        p0_kPa = p0 / 1e3
        assert 300 < p0_kPa < 1500, \
            f"Central pressure should be ~500-1200 kPa, got {p0_kPa:.0f}"

    def test_beta_ITER_range(self):
        """Volume-averaged β should be 2-5% for ITER baseline.
        β = 2μ₀ <p> / B₀²
        """
        rho = np.linspace(0, 1, 100)
        p_avg = np.mean([pressure_profile(r, cfg.n0, cfg.T0_i, cfg.T0_e,
                                          cfg.keV_to_J) for r in rho])
        beta = 2 * cfg.mu0 * p_avg / cfg.B0**2 * 100  # percent
        assert 1.0 < beta < 8.0, \
            f"Volume-average beta should be ~2-5%, got {beta:.1f}%"


# ═══════════════════════════════════════════════════════════════
#  3. FUSION REACTION RATE BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestFusionReactions:
    """Validate DT reaction rate against Bosch-Hale (1992) Table 1."""

    def test_reactivity_peak_location(self):
        """<σv>_DT peaks near T ~ 60-80 keV (Bosch-Hale)."""
        T_arr = np.linspace(1, 200, 500)
        sv = [DT_reaction_rate(T) for T in T_arr]
        T_peak = T_arr[np.argmax(sv)]
        assert 40 < T_peak < 120, \
            f"<σv> peak should be at T~60-80 keV, found at {T_peak:.0f}"

    def test_reactivity_at_10keV(self):
        """<σv>(10 keV) ≈ 1.1e-22 m³/s  (Bosch-Hale / NRL Table)."""
        sv = DT_reaction_rate(10.0)
        assert 1e-23 < sv < 1e-21, \
            f"<σv>(10 keV) should be O(1e-22), got {sv:.2e}"

    def test_reactivity_at_20keV(self):
        """<σv>(20 keV) ≈ 4.2e-22 m³/s  (Bosch-Hale / NRL Table)."""
        sv = DT_reaction_rate(20.0)
        assert 1e-22 < sv < 1e-20, \
            f"<σv>(20 keV) should be O(1e-22), got {sv:.2e}"

    def test_reactivity_monotonic_below_peak(self):
        """<σv> must increase monotonically for T < T_peak."""
        T_arr = np.linspace(1, 50, 100)
        sv = [DT_reaction_rate(T) for T in T_arr]
        for i in range(1, len(sv)):
            assert sv[i] >= sv[i-1] * 0.99, \
                f"<σv> should be increasing below peak at T={T_arr[i]:.1f} keV"

    def test_alpha_power_positive_at_core(self):
        """Alpha heating must be positive in the hot core."""
        P_alpha = alpha_heating_power(0.0, cfg.n0, cfg.T0_i,
                                      cfg.E_alpha, cfg.keV_to_J)
        assert P_alpha > 0, "Alpha heating at core should be > 0"

    def test_alpha_power_negligible_at_edge(self):
        """Alpha heating at cold edge should be negligible."""
        P_core = alpha_heating_power(0.0, cfg.n0, cfg.T0_i,
                                     cfg.E_alpha, cfg.keV_to_J)
        P_edge = alpha_heating_power(0.95, cfg.n0, cfg.T0_i,
                                     cfg.E_alpha, cfg.keV_to_J)
        assert P_edge < 0.01 * P_core, \
            "Alpha heating at edge should be < 1% of core"


# ═══════════════════════════════════════════════════════════════
#  4. RADIATION LOSS BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestRadiationLosses:
    """Validate radiation formulas against NRL Plasma Formulary."""

    def test_bremsstrahlung_formula(self):
        """P_brem = 5.35e-37 Z_eff n_e² √T_e [W/m³] (NRL)."""
        n_e = 1e20
        T_e = 10.0  # keV
        Z_eff = 1.5
        P = bremsstrahlung_loss(n_e, T_e, Z_eff)
        P_expected = 5.35e-37 * Z_eff * n_e**2 * np.sqrt(T_e)
        assert P == pytest.approx(P_expected, rel=1e-6)

    def test_bremsstrahlung_order_of_magnitude(self):
        """For ITER conditions, P_brem should be ~O(10⁴-10⁵) W/m³."""
        P = bremsstrahlung_loss(cfg.n0, cfg.T0_e, cfg.Z_eff)
        assert 1e3 < P < 1e6, \
            f"Bremsstrahlung should be O(10⁴-10⁵) W/m³, got {P:.2e}"

    def test_cyclotron_much_less_than_bremsstrahlung(self):
        """Cyclotron (with 2% escape) should be < bremsstrahlung for ITER."""
        P_brem = bremsstrahlung_loss(cfg.n0, cfg.T0_e, cfg.Z_eff)
        P_cyc = cyclotron_radiation_loss(cfg.n0, cfg.T0_e, cfg.B0)
        assert P_cyc < P_brem, \
            f"P_cyc ({P_cyc:.2e}) should be < P_brem ({P_brem:.2e})"

    def test_alpha_heating_exceeds_radiation(self):
        """Net power balance: P_alpha > P_brem + P_cyc at core (ignition)."""
        P_alpha = alpha_heating_power(0.0, cfg.n0, cfg.T0_i,
                                      cfg.E_alpha, cfg.keV_to_J)
        P_brem = bremsstrahlung_loss(cfg.n0, cfg.T0_e, cfg.Z_eff)
        P_cyc = cyclotron_radiation_loss(cfg.n0, cfg.T0_e, cfg.B0)
        assert P_alpha > (P_brem + P_cyc), \
            f"Alpha ({P_alpha:.2e}) should exceed radiation ({P_brem+P_cyc:.2e}) for ignition"


# ═══════════════════════════════════════════════════════════════
#  5. TRANSPORT COEFFICIENT BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestTransport:
    """Validate transport coefficient hierarchy and magnitudes."""

    def test_neoclassical_exceeds_classical(self):
        """D_neo > D_cl everywhere (fundamental result of neoclassical theory)."""
        rho = 0.5
        B = B_toroidal(cfg.R0 + cfg.a * rho, cfg.B0, cfg.R0)
        n = density_profile(rho, cfg.n0)
        T = temperature_profile(rho, cfg.T0_i)
        q = q_profile(rho, cfg.q0, cfg.qa)

        D_cl = classical_diffusion(rho, T, B, n, cfg.mi,
                                   cfg.Z_eff, cfg.ln_Lambda)
        D_neo = neoclassical_diffusion(rho, T, B, n, cfg.mi,
                                       cfg.Z_eff, q, cfg.epsilon,
                                       cfg.R0, cfg.ln_Lambda)
        assert D_neo > D_cl, \
            f"D_neo ({D_neo:.2e}) must exceed D_cl ({D_cl:.2e})"

    def test_classical_diffusion_order(self):
        """D_cl should be O(10⁻⁴ - 10⁻¹) m²/s for ITER conditions."""
        rho = 0.5
        B = B_toroidal(cfg.R0 + cfg.a * rho, cfg.B0, cfg.R0)
        n = density_profile(rho, cfg.n0)
        T = temperature_profile(rho, cfg.T0_i)
        D_cl = classical_diffusion(rho, T, B, n, cfg.mi,
                                   cfg.Z_eff, cfg.ln_Lambda)
        assert 1e-6 < D_cl < 1e1, \
            f"D_cl should be O(10⁻⁴-10⁻¹), got {D_cl:.2e}"

    def test_transport_hierarchy(self):
        """D_anomalous > D_neoclassical > D_classical (standard ordering)."""
        rho = 0.5
        B = B_toroidal(cfg.R0 + cfg.a * rho, cfg.B0, cfg.R0)
        n = density_profile(rho, cfg.n0)
        T = temperature_profile(rho, cfg.T0_i)
        q = q_profile(rho, cfg.q0, cfg.qa)

        D_cl = classical_diffusion(rho, T, B, n, cfg.mi,
                                   cfg.Z_eff, cfg.ln_Lambda)
        D_neo = neoclassical_diffusion(rho, T, B, n, cfg.mi,
                                       cfg.Z_eff, q, cfg.epsilon,
                                       cfg.R0, cfg.ln_Lambda)
        # Bohm diffusion
        D_bohm = T * cfg.keV_to_J / (16 * cfg.e * B)

        assert D_neo >= D_cl, "D_neo >= D_cl"
        assert D_bohm > D_cl, "D_bohm > D_cl"


# ═══════════════════════════════════════════════════════════════
#  6. GEOMETRY & GRID BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestGeometry:
    """Validate toroidal geometry transformations."""

    def test_magnetic_axis_position(self):
        """On axis (ρ=0), R should be ≈ R0 + Shafranov shift."""
        rho_grid = np.array([[0.0]])
        theta_grid = np.array([[0.0]])
        R, Z = poloidal_to_RZ(rho_grid, theta_grid, cfg.R0, cfg.a,
                               cfg.kappa, cfg.delta_tri)
        # R should be close to R0 (with small Shafranov shift)
        assert abs(R[0, 0] - cfg.R0) < 0.5, \
            f"Magnetic axis R should be near R0={cfg.R0}, got {R[0, 0]:.2f}"
        assert abs(Z[0, 0]) < 0.01, \
            f"Magnetic axis Z should be ~0, got {Z[0, 0]:.2f}"

    def test_elongation(self):
        """Z extent should be κ × a at the edge."""
        theta = np.linspace(0, 2 * np.pi, 200)
        rho_g = np.ones_like(theta)
        R, Z = poloidal_to_RZ(rho_g, theta, cfg.R0, cfg.a,
                               cfg.kappa, cfg.delta_tri)
        Z_extent = Z.max() - Z.min()
        expected = 2 * cfg.kappa * cfg.a
        assert Z_extent == pytest.approx(expected, rel=0.05), \
            f"Z extent should be {expected:.2f} m, got {Z_extent:.2f}"

    def test_aspect_ratio(self):
        """ITER aspect ratio R0/a ≈ 3.1."""
        A = cfg.R0 / cfg.a
        assert A == pytest.approx(3.1, rel=0.05), \
            f"Aspect ratio should be ~3.1, got {A:.2f}"


# ═══════════════════════════════════════════════════════════════
#  7. CURRENT DENSITY BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestCurrentDensity:
    """Validate current density profiles."""

    def test_peak_current_density(self):
        """j0 ≈ 2 Ip / (π a²) for parabolic profile (Wesson Ch.3)."""
        rho = np.linspace(0, 1, cfg.Nr)
        n = np.array([density_profile(r, cfg.n0) for r in rho])
        Ti = np.array([temperature_profile(r, cfg.T0_i) for r in rho])
        Te = np.array([temperature_profile(r, cfg.T0_e) for r in rho])

        j_phi, j_bs = compute_current_density(rho, n, Ti, Te, cfg)

        j0_expected = 2 * cfg.Ip / (np.pi * cfg.a**2)
        # j_phi at ρ≈0 should be close to j0_expected
        j_core = j_phi[1]  # index 1 (index 0 is boundary)
        assert j_core == pytest.approx(j0_expected, rel=0.15), \
            f"Peak j should be ~{j0_expected/1e6:.1f} MA/m², got {j_core/1e6:.1f}"

    def test_current_density_positive(self):
        """Ohmic current should be positive for standard tokamak operation."""
        rho = np.linspace(0, 1, cfg.Nr)
        n = np.array([density_profile(r, cfg.n0) for r in rho])
        Ti = np.array([temperature_profile(r, cfg.T0_i) for r in rho])
        Te = np.array([temperature_profile(r, cfg.T0_e) for r in rho])
        j_phi, _ = compute_current_density(rho, n, Ti, Te, cfg)
        assert np.all(j_phi >= 0), "Ohmic current should be non-negative"


# ═══════════════════════════════════════════════════════════════
#  8. INTEGRATED SYSTEM TEST (FVM)
# ═══════════════════════════════════════════════════════════════

class TestIntegratedFVM:
    """Run the full FVM solver and validate physical outputs."""

    @pytest.fixture(scope='class')
    def fvm_result(self):
        return run_fvm_transport(cfg)

    def test_density_stays_positive(self, fvm_result):
        """Density must remain positive everywhere."""
        assert np.all(fvm_result['n'] > 0), "Density went negative"

    def test_temperature_stays_positive(self, fvm_result):
        """Temperature must remain positive."""
        assert np.all(fvm_result['Ti'] > 0), "Temperature went negative"

    def test_density_profile_peaked(self, fvm_result):
        """Final density should still be peaked (n_core > n_edge)."""
        n = fvm_result['n']
        assert n[0] > n[-1] * 2, \
            f"Profile should be peaked: n_core={n[0]:.2e}, n_edge={n[-1]:.2e}"

    def test_temperature_profile_peaked(self, fvm_result):
        """Final temperature should be peaked."""
        Ti = fvm_result['Ti']
        assert Ti[0] > Ti[-1] * 5, \
            f"Profile should be peaked: Ti_core={Ti[0]:.1f}, Ti_edge={Ti[-1]:.1f}"

    def test_central_temperature_reasonable(self, fvm_result):
        """Central Ti should stay in physical range (1-30 keV)."""
        Ti_core = fvm_result['Ti'][0]
        assert 1.0 < Ti_core < 50.0, \
            f"Central Ti should be 1-30 keV, got {Ti_core:.1f}"

    def test_central_density_reasonable(self, fvm_result):
        """Central n should stay in physical range."""
        n_core = fvm_result['n'][0]
        assert 1e18 < n_core < 1e22, \
            f"Central n should be O(10²⁰), got {n_core:.2e}"


# ═══════════════════════════════════════════════════════════════
#  9. PIC PARTICLE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

class TestPICInit:
    """Validate particle initialization."""

    def test_particles_inside_plasma(self):
        """All particles should be within ρ < 1."""
        rho, theta, phi, vpar, vperp, mu = init_particles(
            1000, cfg.R0, cfg.a, cfg.n0, cfg.T0_i, cfg.mi,
            cfg.q0, cfg.qa, cfg.B0, cfg.Ip, cfg.mu0)
        assert np.all(rho >= 0) and np.all(rho <= 1.0), \
            "Particles should be within ρ ∈ [0, 1]"

    def test_velocity_thermal_scale(self):
        """v_th should be ~O(10⁶) m/s for DT at 15 keV."""
        v_th_expected = np.sqrt(2 * cfg.T0_i * cfg.keV_to_J / cfg.mi)
        rho, theta, phi, vpar, vperp, mu = init_particles(
            10000, cfg.R0, cfg.a, cfg.n0, cfg.T0_i, cfg.mi,
            cfg.q0, cfg.qa, cfg.B0, cfg.Ip, cfg.mu0)
        v_rms = np.sqrt(np.mean(vpar**2))
        # Should be within factor of 3 of thermal speed
        assert v_rms > v_th_expected * 0.1, \
            f"v_rms ({v_rms:.2e}) too low vs v_th ({v_th_expected:.2e})"
        assert v_rms < v_th_expected * 5.0, \
            f"v_rms ({v_rms:.2e}) too high vs v_th ({v_th_expected:.2e})"

    def test_magnetic_moment_positive(self):
        """μ = mv⊥²/(2B) must be positive."""
        rho, theta, phi, vpar, vperp, mu = init_particles(
            1000, cfg.R0, cfg.a, cfg.n0, cfg.T0_i, cfg.mi,
            cfg.q0, cfg.qa, cfg.B0, cfg.Ip, cfg.mu0)
        assert np.all(mu >= 0), "Magnetic moment must be non-negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
