"""
Fusion Performance Diagnostics — Q-factor, Lawson criterion, energy balance.

Computes and reports key fusion performance metrics from FVM transport output.
All physics follows ITER-relevant formulas and NRL Plasma Formulary conventions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from tokamak.config import TokamakConfig
from tokamak.equilibrium import (
    density_profile, temperature_profile,
    DT_reaction_rate, bremsstrahlung_loss, cyclotron_radiation_loss,
    B_toroidal, B_poloidal,
)

# ── Reuse visualization colour palette ──────────────────────────────────────
FACECOLOR = '#0a0a14'
TICK_COL  = '#667788'
LABEL_COL = '#778899'
TITLE_COL = '#ccddee'
SPINE_COL = '#222840'

energy_cmap = LinearSegmentedColormap.from_list('energy_green',
    [(0, '#050e08'), (0.3, '#0a3a18'), (0.6, '#1a8a38'),
     (0.85, '#55cc66'), (1.0, '#bbffcc')])

lawson_cmap = LinearSegmentedColormap.from_list('lawson_hot',
    [(0, '#0a0508'), (0.25, '#3a0a15'), (0.5, '#8a1a15'),
     (0.75, '#dd4422'), (1.0, '#ffee88')])


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0e0e1a')
    ax.set_title(title, fontsize=11, color=TITLE_COL, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=LABEL_COL)
    ax.set_ylabel(ylabel, fontsize=9, color=LABEL_COL)
    ax.tick_params(labelsize=7, colors=TICK_COL)
    for s in ax.spines.values():
        s.set_color(SPINE_COL)
    ax.grid(True, alpha=0.1, color='#334466')


# ═══════════════════════════════════════════════════════════════════════════
#  Volume integration helper
# ═══════════════════════════════════════════════════════════════════════════

def _vol_shell(rho_i, cfg):
    """
    Toroidal volume element for shell at normalised radius rho_i.
    V = 2π R0 * 2π a² * ρ dρ  (circular cross-section approximation,
    elongation factor κ included).
    """
    dr = 1.0 / (cfg.Nr - 1)
    return (2.0 * np.pi * cfg.R0) * (2.0 * np.pi * cfg.a**2 * cfg.kappa
                                      * rho_i * dr)


# ═══════════════════════════════════════════════════════════════════════════
#  Fusion performance
# ═══════════════════════════════════════════════════════════════════════════

def compute_fusion_performance(fvm_data, cfg) -> dict:
    """
    Compute global fusion performance metrics from FVM transport output.

    Parameters
    ----------
    fvm_data : dict
        Output of run_fvm_transport — keys: rho, n, Ti, Te, ...
    cfg : TokamakConfig

    Returns
    -------
    dict with keys:
        Q_factor, P_fusion_MW, P_alpha_MW, P_heat_MW,
        tau_E, tau_E_H98, n_tau_T, beta_N,
        lawson_criterion_met (bool), ignition_margin,
        W_total_MJ, n0_avg, T0_avg_keV, beta_global
    """
    rho  = fvm_data['rho']
    n    = fvm_data['n']
    Ti   = fvm_data['Ti']
    Te   = fvm_data['Te']
    Nr   = len(rho)
    dr   = rho[1] - rho[0] if Nr > 1 else 1.0 / cfg.Nr

    # ── Volume-integrated quantities ────────────────────────────────────────
    P_fusion_W   = 0.0   # total fusion power (alpha + neutron)
    P_alpha_W    = 0.0   # alpha heating power (20% of fusion)
    P_brem_W     = 0.0   # bremsstrahlung radiated power
    P_cyc_W      = 0.0   # cyclotron radiated power
    W_total_J    = 0.0   # stored plasma energy (3/2 ∫ p dV)
    mean_p       = 0.0   # volume-averaged pressure
    vol_total    = 0.0   # total plasma volume

    # DT fusion energy per event: E_fusion = E_alpha + E_neutron
    # E_alpha = 3.5 MeV, E_neutron = 14.1 MeV, total = 17.6 MeV
    E_fusion_keV = 17600.0   # keV
    E_alpha_keV  = cfg.E_alpha  # 3500 keV from config

    for i in range(Nr):
        rr    = rho[i]
        n_loc = n[i]
        Ti_loc = Ti[i]
        Te_loc = Te[i]
        R_loc  = cfg.R0 + cfg.a * rr
        B_loc  = B_toroidal(R_loc, cfg.B0, cfg.R0)

        dV = _vol_shell(max(rr, 0.001), cfg)

        # --- Fusion power density ---
        nD = 0.5 * n_loc  # 50-50 D-T
        nT = 0.5 * n_loc
        sv = DT_reaction_rate(Ti_loc)

        p_fus_density = nD * nT * sv * E_fusion_keV * cfg.keV_to_J   # W/m³
        p_alpha_density = nD * nT * sv * E_alpha_keV * cfg.keV_to_J  # W/m³

        P_fusion_W += p_fus_density * dV
        P_alpha_W  += p_alpha_density * dV

        # --- Radiation losses ---
        P_brem_W += bremsstrahlung_loss(n_loc, Te_loc, cfg.Z_eff) * dV
        P_cyc_W  += cyclotron_radiation_loss(n_loc, Te_loc, B_loc) * dV

        # --- Stored energy ---
        p_loc     = n_loc * (Ti_loc + Te_loc) * cfg.keV_to_J   # Pa
        W_total_J += 1.5 * p_loc * dV
        mean_p    += p_loc * dV
        vol_total += dV

    mean_p /= max(vol_total, 1.0)

    # ── External (non-alpha) heating power ──────────────────────────────────
    # P_loss = P_alpha + P_external = W / tau_E  →  P_external = P_loss - P_alpha
    # Use ITER H-mode scaling for tau_E to find P_loss, then back-calculate.
    # First compute tau_E from the H98 scaling self-consistently.

    # Volume-averaged density in 10^19 m^-3 for H98 scaling
    n_avg = 0.0
    for i in range(Nr):
        dV = _vol_shell(max(rho[i], 0.001), cfg)
        n_avg += n[i] * dV
    n_avg /= max(vol_total, 1.0)

    n19 = n_avg / 1.0e19   # in units of 10^19 m^-3

    # ITER H98(y,2) energy confinement scaling:
    #   τ_E = 0.0562 H98 Ip^0.93 R0^1.97 a^0.19 κ^0.78 n19^0.41 B0^0.15 A^0.19 / P_loss^0.69
    # A = average DT mass number = 2.5
    H98 = 1.0
    A_mass = 2.5
    Ip_MA  = cfg.Ip / 1.0e6   # convert A to MA

    # P_loss = P_alpha + P_external; solve self-consistently:
    # τ_E = W / P_loss  and  τ_E = C * P_loss^(-0.69)
    # => W / P_loss = C / P_loss^0.69  => P_loss^0.31 = C / W  => P_loss = (C/W)^(1/0.31)
    C_H98 = (0.0562 * H98
             * Ip_MA**0.93
             * cfg.R0**1.97
             * cfg.a**0.19
             * cfg.kappa**0.78
             * max(n19, 0.1)**0.41
             * cfg.B0**0.15
             * A_mass**0.19)

    # Self-consistent solution: P_loss * tau_E = W, tau_E = C * P_loss^-0.69
    # P_loss^1.69 = C * (W in MW, since C uses MW convention)?
    # H98 scaling uses P_loss in MW and tau_E in seconds.
    W_MJ = W_total_J / 1.0e6

    # P_loss_MW^(1-0.69) = C_H98 / W_MJ  => P_loss_MW^0.31 = C_H98 / W_MJ
    # But C_H98 * P_loss_MW^-0.69 = W_MJ / P_loss_MW
    # => C_H98 = W_MJ * P_loss_MW^(0.69 - 1) = W_MJ / P_loss_MW^0.31
    # => P_loss_MW = (W_MJ / C_H98)^(1/0.31)
    P_loss_MW = (W_MJ / C_H98) ** (1.0 / 0.31)
    tau_E_H98 = W_MJ / max(P_loss_MW, 1e-3)   # seconds

    P_alpha_MW = P_alpha_W / 1.0e6
    P_fusion_MW = P_fusion_W / 1.0e6

    # External heating = P_loss - P_alpha (clamped to ≥ 1 MW to avoid div-by-zero)
    P_heat_MW = max(P_loss_MW - P_alpha_MW, 1.0)

    # Q-factor
    Q_factor = P_fusion_MW / max(P_heat_MW, 1e-6)

    # tau_E from stored energy / loss power
    tau_E = W_MJ / max(P_loss_MW, 1e-6)   # seconds (same as H98 by construction)

    # ── Lawson parameter nτT ──────────────────────────────────────────────
    # Use peak (on-axis) values for the Lawson figure of merit
    n0_peak  = n[0]
    T0_peak  = Ti[0]   # keV
    n_tau_T  = n0_peak * tau_E * T0_peak   # m^-3 s keV

    # DT Lawson criterion threshold: n τ_E T > 3e21 m^-3 s keV
    LAWSON_THRESHOLD = 3.0e21
    lawson_criterion_met = bool(n_tau_T > LAWSON_THRESHOLD)

    # ── Normalized beta β_N ──────────────────────────────────────────────
    # β = 2μ₀ <p> / B²
    beta_global = 2.0 * cfg.mu0 * mean_p / (cfg.B0**2)

    # Troyon factor: g = Ip / (a B0)  [MA / (m T)]
    troyon_factor = cfg.Ip / (cfg.a * cfg.B0 * 1.0e6)   # MA·m^-1·T^-1  (dimensionless Troyon g)
    beta_N = beta_global / max(troyon_factor, 1e-9) * 100.0   # in % units (standard convention)

    # ── Ignition margin ──────────────────────────────────────────────────
    # Ratio of alpha heating to energy loss power (ignition when > 1)
    P_rad_MW = (P_brem_W + P_cyc_W) / 1.0e6
    ignition_margin = P_alpha_MW / max(P_loss_MW, 1e-6)

    return {
        'Q_factor':              Q_factor,
        'P_fusion_MW':           P_fusion_MW,
        'P_alpha_MW':            P_alpha_MW,
        'P_heat_MW':             P_heat_MW,
        'P_loss_MW':             P_loss_MW,
        'P_rad_MW':              P_rad_MW,
        'tau_E':                 tau_E,
        'tau_E_H98':             tau_E_H98,
        'n_tau_T':               n_tau_T,
        'beta_N':                beta_N,
        'beta_global':           beta_global,
        'lawson_criterion_met':  lawson_criterion_met,
        'ignition_margin':       ignition_margin,
        'W_total_MJ':            W_MJ,
        'n0_avg':                n_avg,
        'n0_peak':               n0_peak,
        'T0_avg_keV':            T0_peak,
        'vol_total_m3':          vol_total,
        'n19':                   n19,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Console report
# ═══════════════════════════════════════════════════════════════════════════

def print_performance_report(perf: dict, cfg):
    """Pretty-print fusion performance metrics matching simulation console style."""
    Q  = perf['Q_factor']
    Pf = perf['P_fusion_MW']
    Pa = perf['P_alpha_MW']
    Ph = perf['P_heat_MW']
    tE = perf['tau_E']
    tH = perf['tau_E_H98']
    nt = perf['n_tau_T']
    bN = perf['beta_N']
    bG = perf['beta_global'] * 100.0   # percent
    lm = perf['lawson_criterion_met']
    ig = perf['ignition_margin']
    W  = perf['W_total_MJ']
    Pr = perf['P_rad_MW']
    n0 = perf['n0_peak']
    T0 = perf['T0_avg_keV']

    LAWSON_THRESHOLD = 3.0e21

    print("━━━ FUSION PERFORMANCE DIAGNOSTICS ━━━")
    print()
    print("  ── Power Balance ─────────────────────────────────────────")
    print(f"  [CALC] Fusion power          P_fusion = {Pf:8.1f} MW")
    print(f"  [CALC] Alpha heating         P_alpha  = {Pa:8.1f} MW   (20% of fusion)")
    print(f"  [CALC] External heating      P_heat   = {Ph:8.1f} MW")
    print(f"  [CALC] Radiation losses      P_rad    = {Pr:8.1f} MW")
    print(f"  [CALC] Stored energy         W        = {W:8.1f} MJ")
    print()
    print("  ── Fusion Gain ──────────────────────────────────────────")
    print(f"  [CALC] Q-factor              Q        = {Q:8.2f}"
          f"   ({'IGNITION' if Q > 30 else 'BURNING' if Q > 5 else 'break-even' if Q >= 1 else 'sub-breakeven'})")
    print(f"  [CALC] Ignition margin               = {ig:8.3f}"
          f"   ({'>> ignition' if ig > 1 else 'approaching' if ig > 0.7 else 'below ignition'})")
    print()
    print("  ── Confinement ──────────────────────────────────────────")
    print(f"  [CALC] Confinement time      tau_E    = {tE:8.3f} s")
    print(f"  [CALC] H98 scaling (H98=1)   tau_H98  = {tH:8.3f} s")
    print(f"  [CALC] H-factor              H        = {tE/max(tH, 1e-9):8.2f}")
    print()
    print("  ── Lawson Criterion ─────────────────────────────────────")
    print(f"  [CALC] Peak density          n0       = {n0:.3e} m⁻³")
    print(f"  [CALC] Peak temperature      T0       = {T0:8.1f} keV")
    print(f"  [CALC] nτT (Lawson param)             = {nt:.3e} m⁻³ s keV")
    print(f"  [CALC] Lawson threshold               = {LAWSON_THRESHOLD:.1e} m⁻³ s keV")
    criterion_str = "MET" if lm else "NOT MET"
    margin_pct = (nt / LAWSON_THRESHOLD - 1.0) * 100.0
    print(f"  [CALC] Lawson criterion      [{criterion_str}]   "
          f"({margin_pct:+.0f}% vs threshold)")
    print()
    print("  ── Stability ─────────────────────────────────────────────")
    print(f"  [CALC] Global beta           β        = {bG:8.2f} %")
    print(f"  [CALC] Normalised beta       β_N      = {bN:8.2f}"
          f"   (ITER target ~1.8, Troyon limit ~3.5)")
    stability_str = ("stable" if bN < 2.0 else
                     "near Troyon limit" if bN < 3.0 else "EXCEEDS Troyon limit")
    print(f"  [CALC] Beta stability        [{stability_str}]")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Energy balance
# ═══════════════════════════════════════════════════════════════════════════

def compute_energy_balance(fvm_data, cfg) -> dict:
    """
    Compute radially-integrated power balance.

    Returns dict with P_ohmic, P_alpha, P_rad, P_transport, P_net (all in MW).
    """
    rho  = fvm_data['rho']
    n    = fvm_data['n']
    Ti   = fvm_data['Ti']
    Te   = fvm_data['Te']
    Nr   = len(rho)

    E_alpha_keV  = cfg.E_alpha

    P_alpha_W    = 0.0
    P_brem_W     = 0.0
    P_cyc_W      = 0.0
    P_transport_W = 0.0

    for i in range(Nr):
        rr    = rho[i]
        n_loc = n[i]
        Ti_loc = Ti[i]
        Te_loc = Te[i]
        R_loc  = cfg.R0 + cfg.a * rr
        B_loc  = B_toroidal(R_loc, cfg.B0, cfg.R0)
        dV     = _vol_shell(max(rr, 0.001), cfg)

        nD = 0.5 * n_loc
        sv = DT_reaction_rate(Ti_loc)
        P_alpha_W += nD * nD * sv * E_alpha_keV * cfg.keV_to_J * dV

        P_brem_W += bremsstrahlung_loss(n_loc, Te_loc, cfg.Z_eff) * dV
        P_cyc_W  += cyclotron_radiation_loss(n_loc, Te_loc, B_loc) * dV

    # Ohmic heating: P_ohm = η j² integrated
    # Use Spitzer resistivity: η = 5.2e-5 Z_eff ln_Λ / T_e^1.5 [Ω m]
    P_ohmic_W = 0.0
    if 'D_cl' in fvm_data:
        from tokamak.equilibrium import q_profile
        for i in range(Nr):
            rr     = rho[i]
            Te_loc = Te[i]
            n_loc  = n[i]
            dV     = _vol_shell(max(rr, 0.001), cfg)
            # Spitzer resistivity [Ω m]
            eta_sp = (5.2e-5 * cfg.Z_eff * cfg.ln_Lambda
                      / max(Te_loc, 0.1)**1.5)
            # Approximate j from enclosed current density using parabolic profile
            r_m    = max(rr * cfg.a, 1e-3)
            j_loc  = (cfg.mu0 * cfg.Ip * rr**2
                      / (2.0 * np.pi * r_m**2 + 1e-10))  # rough estimate
            j_loc  = min(j_loc, 1.0e7)   # cap at 10 MA/m²
            P_ohmic_W += eta_sp * j_loc**2 * dV

    # Transport losses: residual = P_in - P_rad
    P_alpha_MW = P_alpha_W / 1.0e6
    P_brem_MW  = P_brem_W  / 1.0e6
    P_cyc_MW   = P_cyc_W   / 1.0e6
    P_rad_MW   = P_brem_MW + P_cyc_MW
    P_ohmic_MW = P_ohmic_W / 1.0e6

    # Transport power = alpha heating (input) - radiation losses
    P_transport_MW = P_alpha_MW - P_rad_MW
    P_net_MW = P_alpha_MW + P_ohmic_MW - P_rad_MW

    return {
        'P_ohmic':     P_ohmic_MW,
        'P_alpha':     P_alpha_MW,
        'P_rad':       P_rad_MW,
        'P_brem':      P_brem_MW,
        'P_cyc':       P_cyc_MW,
        'P_transport': P_transport_MW,
        'P_net':       P_net_MW,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Energy balance plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_energy_balance(fvm_data, perf, cfg,
                        filename='07_energy_balance.png'):
    """
    Three-panel diagnostic figure:
      1. Energy balance pie chart (power channels)
      2. Lawson diagram (nτT vs T0) with ITER operating point
      3. Q-factor gauge / bar

    Saved to filename using the dark theme matching visualization.py.
    """
    print("  [PLOT] Generating energy balance diagnostics...")

    rho = fvm_data['rho']
    Ti  = fvm_data['Ti']

    eb = compute_energy_balance(fvm_data, cfg)

    fig = plt.figure(figsize=(20, 7), facecolor=FACECOLOR)
    fig.suptitle('FUSION PERFORMANCE — Energy Balance & Lawson Diagnostics',
                 fontsize=15, color='white', fontweight='300', y=0.98)

    gs = GridSpec(1, 3, figure=fig, wspace=0.35,
                  left=0.05, right=0.96, top=0.88, bottom=0.10)

    # ── Panel 1: Power balance pie ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0e0e1a')

    P_a  = max(eb['P_alpha'], 0.01)
    P_oh = max(eb['P_ohmic'], 0.01)
    P_br = max(eb['P_brem'],  0.01)
    P_cy = max(eb['P_cyc'],   0.01)
    # Transport loss = alpha - rad (stored + transported)
    P_tr = max(P_a - P_br - P_cy, 0.01)

    labels = ['Alpha\nHeating', 'Ohmic\nHeating',
              'Bremsstrahlung', 'Cyclotron\nRad.', 'Transport\nLoss']
    sizes  = [P_a, P_oh, P_br, P_cy, P_tr]
    colors = ['#33cc66', '#55aaff', '#ff6633', '#ffaa22', '#aa55dd']
    explode = [0.06, 0.0, 0.04, 0.04, 0.03]

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=120,
        textprops={'color': LABEL_COL, 'fontsize': 8},
        wedgeprops={'edgecolor': '#0a0a14', 'linewidth': 1.5},
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color(TITLE_COL)

    # Add power values as a text box
    total_in = P_a + P_oh
    info = (f"P_alpha  = {P_a:.1f} MW\n"
            f"P_ohmic  = {P_oh:.2f} MW\n"
            f"P_brem   = {P_br:.1f} MW\n"
            f"P_cycl   = {P_cy:.2f} MW\n"
            f"P_transp = {P_tr:.1f} MW")
    ax1.text(0.0, -1.55, info, transform=ax1.transData,
             fontsize=7, color=LABEL_COL,
             ha='center', va='bottom',
             bbox=dict(facecolor='#0e1520', edgecolor=SPINE_COL,
                       boxstyle='round,pad=0.4'))
    ax1.set_title('Power Balance Channels', fontsize=11, color=TITLE_COL, pad=10)

    # ── Panel 2: Lawson diagram ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2,
              title='Lawson Diagram  (DT Fusion)',
              xlabel='Ion Temperature  T₀ [keV]',
              ylabel='nτT  [m⁻³ s keV]')

    # Draw Lawson contour curve: nτT_min(T) for Q=1 from empirical fit
    # For DT: nτ_E * T_i ≥ 3×10²¹ m⁻³ s keV (flat threshold approx.)
    T_range = np.linspace(4, 80, 400)
    # Refined triple product threshold from Glasstone & Lovberg:
    # nτT_min ~ (12 k T) / (<σv> E_alpha * tau_burn_factor)
    nτT_ignition = np.zeros_like(T_range)
    nτT_breakeven = np.zeros_like(T_range)
    for k, Tk in enumerate(T_range):
        sv = DT_reaction_rate(Tk)
        E_alpha_J = cfg.E_alpha * cfg.keV_to_J
        # Ignition: alpha heating equals confinement loss
        # nτ_E > 12 T_J / (<σv> E_alpha)  — classical ignition condition
        T_J = Tk * cfg.keV_to_J
        if sv > 0:
            nτ_ign = 12.0 * T_J / (sv * E_alpha_J)
            nτT_ignition[k]  = nτ_ign * Tk   # multiply by T to get nτT
            nτT_breakeven[k] = nτ_ign * Tk * 5.0 / 4.0  # Q=1: slightly higher
        else:
            nτT_ignition[k]  = 1e25
            nτT_breakeven[k] = 1e25

    ax2.semilogy(T_range, nτT_ignition,  color='#55cc66', lw=2.0,
                 label='Ignition (Q→∞)', alpha=0.9)
    ax2.semilogy(T_range, nτT_breakeven, color='#ffaa22', lw=1.5,
                 linestyle='--', label='Break-even (Q=1)', alpha=0.8)

    # Flat Lawson threshold line
    ax2.axhline(3.0e21, color='#ff4444', lw=1.0, linestyle=':',
                alpha=0.6, label='nτT = 3×10²¹ (approx.)')

    # Current operating point
    n_tau_T = perf['n_tau_T']
    T0_op   = perf['T0_avg_keV']
    marker_color = '#00ffaa' if perf['lawson_criterion_met'] else '#ff6644'
    ax2.scatter([T0_op], [n_tau_T], s=180, color=marker_color,
                zorder=10, marker='*', label=f'This simulation', edgecolors='white',
                linewidth=0.8)
    ax2.annotate(f'  Q={perf["Q_factor"]:.1f}',
                 (T0_op, n_tau_T), color=marker_color,
                 fontsize=8, va='center')

    # Shade ignition region
    T_shade = T_range[(nτT_ignition > 1e18) & (nτT_ignition < 1e24)]
    nτ_shade = nτT_ignition[(nτT_ignition > 1e18) & (nτT_ignition < 1e24)]
    if len(T_shade) > 1:
        ax2.fill_betweenx([1e18, nτ_shade.min()], T_shade.min(), T_shade.max(),
                          alpha=0.05, color='#33cc66')

    ax2.set_xlim(4, 80)
    ax2.set_ylim(1e19, 1e24)
    ax2.legend(fontsize=7, loc='upper right',
               facecolor='#0e0e20', edgecolor=SPINE_COL,
               labelcolor=LABEL_COL)

    # ── Panel 3: Q-factor gauge ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3,
              title='Fusion Gain  Q-factor',
              xlabel='',
              ylabel='Q = P_fusion / P_external')

    Q_val = perf['Q_factor']
    Q_display = min(Q_val, 50.0)   # cap display at 50 for aesthetics

    # Milestone Q values
    milestones = [
        (0.0,  '#334466', ''),
        (1.0,  '#ffaa22', 'Break-even'),
        (5.0,  '#33aaff', 'ITER target'),
        (10.0, '#55cc66', 'DEMO target'),
        (30.0, '#ff66aa', 'Near-ignition'),
        (50.0, '#ffffff', 'Ignition'),
    ]

    # Draw gauge bars
    bar_colors  = []
    bar_heights = []
    bar_labels  = []
    bar_qs      = [m[0] for m in milestones[1:]]

    for i in range(len(milestones) - 1):
        q_lo, col_lo, lbl_lo = milestones[i]
        q_hi, col_hi, lbl_hi = milestones[i + 1]
        bar_colors.append(col_lo)
        bar_heights.append(q_hi - q_lo)
        bar_labels.append(lbl_hi)

    bottoms = [milestones[i][0] for i in range(len(milestones) - 1)]

    for i, (bh, bc, bot, bl) in enumerate(
            zip(bar_heights, bar_colors, bottoms, bar_labels)):
        alpha_val = 0.35 if Q_display <= bot else 0.75
        ax3.bar(0.5, bh, bottom=bot, width=0.6,
                color=bc, alpha=alpha_val,
                edgecolor=SPINE_COL, linewidth=0.5)
        ax3.text(1.22, bot + bh * 0.5, bl,
                 ha='left', va='center', fontsize=8,
                 color=LABEL_COL if Q_display <= bot else TITLE_COL)
        ax3.text(-0.22, bot + bh * 0.5, f'Q={bot + bh:.0f}',
                 ha='right', va='center', fontsize=7, color=TICK_COL)

    # Draw the actual Q value indicator
    ax3.axhline(Q_display, color='#ffffff', lw=2.5, alpha=0.9,
                linestyle='-', zorder=10)
    ax3.scatter([0.5], [Q_display], s=250, color='#ffffff',
                zorder=11, marker='D', edgecolors='#aaccff', linewidth=1.0)

    Q_label = (f'Q = {Q_val:.1f}'
               if Q_val < 50 else f'Q = {Q_val:.0f} (→∞)')
    ax3.text(0.5, Q_display + 1.5, Q_label,
             ha='center', va='bottom', fontsize=11,
             color='white', fontweight='bold')

    ax3.set_xlim(-0.5, 1.8)
    ax3.set_ylim(-1, 53)
    ax3.set_xticks([])
    ax3.set_yticks([m[0] for m in milestones])
    ax3.tick_params(axis='y', labelsize=7, colors=TICK_COL)
    ax3.set_facecolor('#0e0e1a')
    for s in ax3.spines.values():
        s.set_color(SPINE_COL)

    # Annotation box at bottom
    ann = (f"P_fusion = {perf['P_fusion_MW']:.0f} MW\n"
           f"P_ext    = {perf['P_heat_MW']:.0f} MW\n"
           f"τ_E      = {perf['tau_E']:.2f} s\n"
           f"β_N      = {perf['beta_N']:.2f}")
    ax3.text(0.5, -0.5, ann, ha='center', va='top',
             fontsize=7.5, color=LABEL_COL, transform=ax3.transData,
             bbox=dict(facecolor='#0e1520', edgecolor=SPINE_COL,
                       boxstyle='round,pad=0.5'))

    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")
