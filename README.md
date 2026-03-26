# 🔥 Tokamak Plasma Transport Simulation

**Hybrid PIC + FVM numerical simulation of a deuterium–tritium fusion plasma in an ITER-like tokamak reactor.**

This code models radial heat and particle transport, guiding-center particle orbits, and electrostatic equilibrium using two complementary numerical methods — Finite Volume (FVM) for fluid transport and Particle-In-Cell (PIC) for kinetic orbit physics — with optional CUDA acceleration.

---

## 📐 Physics Overview

The simulation targets ITER-like parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| R₀ | 6.2 m | Major radius |
| a | 2.0 m | Minor radius |
| B₀ | 5.3 T | On-axis toroidal field |
| Iₚ | 15 MA | Plasma current |
| n₀ | 1.0 × 10²⁰ m⁻³ | Central electron density |
| T₀ | 15 keV | Central ion/electron temperature |
| κ | 1.7 | Elongation |
| δ | 0.33 | Triangularity |

### Equations Solved

**FVM (Radial Transport)**
```
∂n/∂t = (1/r) ∂/∂r(r D ∂n/∂r) + S_fuel − n/τ_p

∂T/∂t = (1/r) ∂/∂r(r χ ∂T/∂r) + P_α/(n·e) − P_rad/(n·e) − T/τ_E
```

- **Transport**: Three-regime neoclassical diffusion (banana / plateau / Pfirsch-Schlüter) selected via collisionality `ν*`, plus gyro-Bohm anomalous transport
- **Heating**: DT alpha heating using Bosch-Hale / NRL Plasma Formulary reactivity `⟨σv⟩_DT` (log-log table interpolation, validated 0.2–300 keV)
- **Radiation**: Bremsstrahlung (`5.35×10⁻³⁷ Z_eff n² √T`) and cyclotron radiation (2% escape fraction)
- **Confinement**: Particle loss `n/τ_p` and energy loss `T/τ_E` (H-mode confinement scaling)

**PIC (Guiding-Center Orbits)**
```
dρ/dt = v_∥ b̂·∇ρ + v_D·∇ρ
dθ/dt = v_∥ / (qR) + drifts
dφ/dt = v_∥ / R + E×B drift
```

- RK4 integration of 100,000 guiding-center ions
- ∇B drift, curvature drift, E×B drift, mirror force
- Monte Carlo pitch-angle scattering (Lorentz collision operator)
- Produces banana and passing orbit topology

---

## 🗂️ Project Architecture

```
fusion/
├── tokamak/                   # Main simulation package
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point
│   ├── config.py              # ITER-like parameters (TokamakConfig)
│   ├── backend.py             # CUDA/CPU backend abstraction (CuPy/NumPy)
│   ├── equilibrium.py         # Analytic profiles, DT rate, radiation
│   ├── grids.py               # 2D poloidal grid with D-shaped geometry
│   ├── fvm.py                 # 1D radial FVM transport solver
│   ├── pic.py                 # Guiding-center PIC orbit integrator
│   ├── poisson.py             # 2D Poisson solver (quasi-neutrality)
│   ├── current.py             # Ohmic + bootstrap current density
│   ├── export3d.py            # 3D toroidal mesh export (.npz)
│   └── visualization.py       # 6 diagnostic plot panels
├── animate_tokamak.py         # 3D PyVista animation script
├── test_physics.py            # 41 physics benchmark tests
└── README.md
```

### Module Summary

| Module | Method | What It Solves |
|--------|--------|----------------|
| `fvm.py` | Finite Volume | 1D radial transport: `n(ρ)` and `T(ρ)` with diffusion, alpha heating, radiation, confinement losses |
| `pic.py` | Particle-In-Cell | 100k guiding-center ion orbits in full 3D toroidal geometry `(ρ, θ, φ)` via RK4 |
| `poisson.py` | Iterative solver | 2D electrostatic potential Φ from charge-density imbalance |
| `equilibrium.py` | Analytic models | Density/temperature profiles, q(ρ), B-field, DT reactivity, radiation formulas |
| `current.py` | Neoclassical | Ohmic current `j_φ` and bootstrap current `j_bs` from pressure gradients |

---

## 🚀 Quick Start

### Requirements

```
numpy
numba
matplotlib
scipy
```

Optional: `cupy` (CUDA acceleration), `pyvista` (3D visualization)

### Run the Simulation

```bash
# Auto-detect CUDA, full simulation with plots and 3D export
python -m tokamak

# Force CPU backend
python -m tokamak --backend cpu

# Skip plots (headless/CI)
python -m tokamak --backend cpu --no-plots

# Skip 3D data export
python -m tokamak --no-export
```

### Run Physics Tests

```bash
pytest test_physics.py -v
```

All **41 tests** validate against published benchmarks:

| Category | Tests | Validates Against |
|----------|-------|-------------------|
| Magnetic field | 7 | Toroidal 1/R, Ampère's law, q-profile |
| Equilibrium profiles | 7 | ITER density/temperature/pressure design values |
| Fusion reactions | 6 | Bosch-Hale (1992) / NRL Formulary DT reactivity |
| Radiation losses | 4 | NRL bremsstrahlung formula, power balance |
| Transport coefficients | 3 | Classical < neoclassical < anomalous hierarchy |
| Geometry | 3 | ITER aspect ratio, elongation |
| Current density | 2 | Wesson Ch.3 peak j₀ formula |
| Integrated FVM | 6 | Stability, peaked profiles, physical output range |
| PIC initialization | 3 | Thermal velocity, particle bounds, μ > 0 |

### 3D Animation

After running the simulation (which exports `tokamak_3d_data.npz`):

```bash
# Interactive 3D window
python animate_tokamak.py

# Export to video
python animate_tokamak.py --output tokamak.mp4
python animate_tokamak.py --output tokamak.gif
```

---

## 📊 Simulation Phases

When you run the simulation, it executes in six sequential phases:

```
═════════════════════════════════════════════════════════════════
  TOKAMAK PLASMA TRANSPORT SIMULATION
  PIC + FVM · CUDA-accelerated · ITER-like parameters
═════════════════════════════════════════════════════════════════

━━━ PHASE 1: Finite Volume Transport Solver ━━━
  [FVM] Initializing radial transport solver...
  [FVM] Running 2000 steps (dt=1.0e-04 s)...
  [FVM] Estimated fusion power: XXX MW
  [FVM] Energy confinement time: X.XX s
  [FVM] Central Ti=XX.X keV, n0=X.XXe+20 m⁻³

━━━ PHASE 2: Equilibrium Derived Quantities ━━━
  [CALC] Current density profiles...
  [CALC] Poisson solver for electrostatic potential...

━━━ PHASE 3: Particle-In-Cell Orbit Integration ━━━
  [PIC] Initializing 100,000 guiding-center particles...
  [PIC] Pushing particles for 400 steps (dt=1.0e-07s, RK4)...

━━━ PHASE 4: Charge Deposition (CIC) ━━━

━━━ PHASE 5: 3D Animation Data Export ━━━
  [EXPORT] Building 3D toroidal mesh...

━━━ PHASE 6: Generating Diagnostic Plots ━━━
```

### Diagnostic Outputs

The simulation generates **6 diagnostic plots**:

| Plot | Content |
|------|---------|
| `01_poloidal_contours.png` | Density, temperature, potential, and current in poloidal cross-section |
| `02_axial_contours.png` | Scalar fields viewed along the magnetic axis |
| `03_radial_profiles.png` | n(ρ), T(ρ), p(ρ), q(ρ), and j(ρ) radial profiles |
| `04_particle_orbits.png` | Banana and passing orbit trajectories from PIC |
| `05_toroidal_sections.png` | Temperature contours at multiple toroidal angles |
| `06_transport_hierarchy.png` | D_classical vs D_neoclassical vs D_anomalous |

---

## ⚙️ Configuration

All parameters are defined in `tokamak/config.py` as a `TokamakConfig` class. Key tunable parameters:

```python
# Grid resolution
Nr     = 128          # Radial grid points
Ntheta = 128          # Poloidal grid points
Nparticles = 100_000  # PIC particle count

# Timesteps
dt      = 1.0e-7      # PIC time step [s]
Nt_pic  = 400         # PIC integration steps
dt_fvm  = 1.0e-4      # FVM time step [s]
Nt_fvm  = 2000        # FVM diffusion steps

# Transport model
chi_anomalous_mult = 0.5   # Gyro-Bohm anomalous transport multiplier
alpha_n = 0.5              # Density peaking exponent
alpha_T = 1.5              # Temperature peaking exponent
```

---

## 🧪 Physical Accuracy

### ✅ What this code models correctly

- **Magnetic geometry**: 1/R toroidal field, Ampère's law for B_θ, D-shaped flux surfaces with Shafranov shift
- **Transport**: Three-regime neoclassical diffusion (Hinton-Hazeltine formulation) with gyro-Bohm anomalous transport
- **Fusion**: Bosch-Hale DT reactivity, validated against NRL Plasma Formulary tabulated data
- **Alpha heating**: `P_α = ¼ n_D n_T ⟨σv⟩ E_α` with self-consistent density/temperature feedback
- **Radiation**: Bremsstrahlung + cyclotron with physical escape fraction
- **Orbit physics**: RK4 guiding-center integrator producing banana and passing orbits
- **Confinement**: Particle `n/τ_p` and energy `T/τ_E` loss terms with H-mode scaling

### ⚠️ Simplifications

- **1D transport** — FVM is radial only; turbulence (ITG, TEM) requires gyrokinetic codes (GENE, GS2)
- **Analytic equilibrium** — parametric profiles instead of numerical Grad-Shafranov solution
- **No MHD stability** — does not check for sawteeth, ELMs, or kink modes
- **Reduced collisions** — Monte Carlo pitch-angle scattering, not full Fokker-Planck
- **No SOL/divertor** — edge uses exponential decay boundary conditions

---

## 📚 References

1. **ITER Physics Basis**, Nucl. Fusion **39** (1999) 2137
2. **Bosch & Hale**, "Improved formulas for fusion cross-sections and thermal reactivities", Nucl. Fusion **32** (1992) 611
3. **NRL Plasma Formulary** (2019 revised edition)
4. **Wesson**, *Tokamaks*, 4th ed., Oxford University Press (2011)
5. **Hinton & Hazeltine**, "Theory of plasma transport in toroidal confinement systems", Rev. Mod. Phys. **48** (1976) 239

---

## 📄 License

Academic / educational use.
