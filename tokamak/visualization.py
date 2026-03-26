"""
Visualization — diagnostic plots for tokamak simulation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from tokamak.config import TokamakConfig
from tokamak.grids import poloidal_to_RZ
from tokamak.equilibrium import (
    density_profile, temperature_profile, pressure_profile,
    B_toroidal, q_profile, impurity_profile,
)

# ── Colormaps ──
plasma_cmap = LinearSegmentedColormap.from_list('plasma_hot',
    [(0, '#0a0a2e'), (0.2, '#1a0a4e'), (0.4, '#4a1a7e'),
     (0.6, '#aa3a3e'), (0.8, '#ee7722'), (1.0, '#ffee88')])

density_cmap = LinearSegmentedColormap.from_list('density_blue',
    [(0, '#050520'), (0.25, '#0a1555'), (0.5, '#1a4599'),
     (0.75, '#3a88dd'), (1.0, '#aaddff')])

temp_cmap = LinearSegmentedColormap.from_list('temp_fire',
    [(0, '#0a0508'), (0.2, '#3a0a15'), (0.4, '#8a1a15'),
     (0.6, '#dd4422'), (0.8, '#ff9944'), (1.0, '#ffeecc')])

pressure_cmap = LinearSegmentedColormap.from_list('press_violet',
    [(0, '#08050e'), (0.25, '#2a1058'), (0.5, '#6a20a8'),
     (0.75, '#aa55dd'), (1.0, '#eeccff')])

impurity_cmap = LinearSegmentedColormap.from_list('impurity_gold',
    [(0, '#0a0a05'), (0.3, '#2a2a05'), (0.5, '#6a5a10'),
     (0.75, '#ccaa22'), (1.0, '#ffee66')])

current_cmap = LinearSegmentedColormap.from_list('current_teal',
    [(0, '#050a0a'), (0.25, '#0a2a3a'), (0.5, '#1a6a7a'),
     (0.75, '#33aabb'), (1.0, '#99eeff')])

FACECOLOR = '#0a0a14'
TICK_COL  = '#667788'
LABEL_COL = '#778899'
TITLE_COL = '#ccddee'
SPINE_COL = '#222840'


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0e0e1a')
    ax.set_title(title, fontsize=11, color=TITLE_COL, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=LABEL_COL)
    ax.set_ylabel(ylabel, fontsize=9, color=LABEL_COL)
    ax.tick_params(labelsize=7, colors=TICK_COL)
    for s in ax.spines.values():
        s.set_color(SPINE_COL)
    ax.grid(True, alpha=0.1, color='#334466')


# ═══════════════════════════════════════════════════════════════
#  Poloidal cross-section contours
# ═══════════════════════════════════════════════════════════════

def plot_poloidal_contours(fvm_data, phi_2d, j_phi, j_bs, cfg, filename):
    print("  [PLOT] Generating poloidal cross-section contours...")

    rho = fvm_data['rho']
    theta = np.linspace(0, 2*np.pi, cfg.Ntheta)
    RHO, THETA = np.meshgrid(rho, theta, indexing='ij')
    R_grid, Z_grid = poloidal_to_RZ(RHO, THETA, cfg.R0, cfg.a,
                                     cfg.kappa, cfg.delta_tri)

    # Build 2D fields
    n_2d  = np.zeros_like(RHO)
    T_2d  = np.zeros_like(RHO)
    p_2d  = np.zeros_like(RHO)
    nz_2d = np.zeros_like(RHO)
    j_2d  = np.zeros_like(RHO)

    for i in range(cfg.Nr):
        for j in range(cfg.Ntheta):
            rr = rho[i]; th = theta[j]
            rr_eff = max(0, min(0.999, rr + 0.05*(1-rr)*np.cos(th)))
            n_2d[i,j]  = density_profile(rr_eff, cfg.n0)
            T_2d[i,j]  = temperature_profile(rr_eff, cfg.T0_i)
            p_2d[i,j]  = pressure_profile(rr_eff, cfg.n0, cfg.T0_i,
                                           cfg.T0_e, cfg.keV_to_J)
            nz_2d[i,j] = (impurity_profile(rr_eff, cfg.n0, cfg.Z_imp)
                          * cfg.n0 * 0.02)
            balloon = 1.0 + 0.15*rr**2*np.cos(th)
            T_2d[i,j] *= balloon
            p_2d[i,j] *= balloon
            if i < len(j_phi):
                j_2d[i,j] = j_phi[i] + j_bs[i]

    phi_plot = np.zeros_like(RHO)
    pnr, pnt = phi_2d.shape
    for i in range(cfg.Nr):
        for j in range(cfg.Ntheta):
            phi_plot[i,j] = phi_2d[min(i,pnr-1), min(j,pnt-1)]

    fig = plt.figure(figsize=(22, 15), facecolor=FACECOLOR)
    fig.suptitle('SECCIONES POLOIDALES — Contornos de Equilibrio y Transporte',
                 fontsize=16, color='white', fontweight='300', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.25,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    panels = [
        (n_2d/1e20, 'Densidad  n [10²⁰ m⁻³]', density_cmap, 20),
        (T_2d,      'Temperatura  Tᵢ [keV]',   temp_cmap,    20),
        (p_2d/1e3,  'Presión  p [kPa]',         pressure_cmap,20),
        (phi_plot,  'Potencial Electrostático  Φ [V]', 'RdBu_r', 20),
        (nz_2d/1e18, f'Impurezas C⁶⁺  n_z [10¹⁸ m⁻³]', impurity_cmap, 18),
        (j_2d/1e6,  'Densidad de Corriente  j [MA/m²]', current_cmap, 20),
    ]

    for idx, (data, title, cmap, nlev) in enumerate(panels):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        ax.set_facecolor(FACECOLOR); ax.set_aspect('equal')
        levels = np.linspace(np.nanmin(data)*0.95, np.nanmax(data)*1.05, nlev)
        if levels[-1] <= levels[0]:
            levels = np.linspace(0, 1, nlev)
        cf = ax.contourf(R_grid, Z_grid, data, levels=levels,
                         cmap=cmap, extend='both')
        ax.contour(R_grid, Z_grid, data, levels=levels[::2],
                   colors='white', linewidths=0.3, alpha=0.3)
        for rr in [0.2, 0.4, 0.6, 0.8, 1.0]:
            idx_r = int(rr * (cfg.Nr - 1))
            if idx_r < cfg.Nr:
                ax.plot(R_grid[idx_r,:], Z_grid[idx_r,:],
                        'w-', linewidth=0.4, alpha=0.25)
        ax.plot(R_grid[-1,:], Z_grid[-1,:], 'w-', linewidth=1.0, alpha=0.5)
        ax.plot(cfg.R0+cfg.a*0.1, 0, 'w+', markersize=8,
                markeredgewidth=1.5, alpha=0.5)
        cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7, colors='#8899aa')
        cb.outline.set_edgecolor('#334455')
        ax.set_title(title, fontsize=11, color=TITLE_COL, pad=10)
        ax.set_xlabel('R [m]', fontsize=9, color=LABEL_COL)
        ax.set_ylabel('Z [m]', fontsize=9, color=LABEL_COL)
        ax.tick_params(labelsize=7, colors=TICK_COL)
        for s in ax.spines.values(): s.set_color(SPINE_COL)

    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  Axial (toroidal midplane) contours
# ═══════════════════════════════════════════════════════════════

def plot_axial_contours(fvm_data, cfg, filename):
    print("  [PLOT] Generating axial/toroidal midplane contours...")

    Nphi = 256
    phi_arr = np.linspace(0, 2*np.pi, Nphi)
    rho_arr = fvm_data['rho']
    RHO_ax, PHI_ax = np.meshgrid(rho_arr, phi_arr, indexing='ij')
    R_ax = cfg.R0 + cfg.a * RHO_ax
    X_ax = R_ax * np.cos(PHI_ax)
    Y_ax = R_ax * np.sin(PHI_ax)

    n_ax  = np.zeros_like(RHO_ax)
    T_ax  = np.zeros_like(RHO_ax)
    Bt_ax = np.zeros_like(RHO_ax)
    nz_ax = np.zeros_like(RHO_ax)
    q_ax  = np.zeros_like(RHO_ax)
    beta_ax = np.zeros_like(RHO_ax)

    for i in range(cfg.Nr):
        for j in range(Nphi):
            rr = rho_arr[i]; ph = phi_arr[j]
            R = cfg.R0 + cfg.a * rr
            n_b = density_profile(rr, cfg.n0)
            T_b = temperature_profile(rr, cfg.T0_i)
            B_t = B_toroidal(R, cfg.B0, cfg.R0)
            q_l = q_profile(rr, cfg.q0, cfg.qa)
            mode = 1.0 + 0.08*rr**2*np.cos(ph) + 0.03*rr**3*np.cos(3*ph)
            n_ax[i,j]  = n_b * mode
            T_ax[i,j]  = T_b * mode
            Bt_ax[i,j] = B_t * (1.0 + 0.015*np.cos(cfg.N_TF*ph))
            nz_ax[i,j] = impurity_profile(rr, cfg.n0, cfg.Z_imp)*cfg.n0*0.02*mode
            q_ax[i,j]  = q_l
            p = n_b * (T_b + 0.9*T_b) * cfg.keV_to_J
            beta_ax[i,j] = 2*cfg.mu0*p / B_t**2 * 100

    fig = plt.figure(figsize=(22, 15), facecolor=FACECOLOR)
    fig.suptitle('SECCIONES AXIALES (Plano Medio Toroidal, Z = 0)',
                 fontsize=16, color='white', fontweight='300', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.25,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    panels = [
        (n_ax/1e20, 'Densidad  n [10²⁰ m⁻³]', density_cmap, 20),
        (T_ax,      'Temperatura  Tᵢ [keV]',   temp_cmap,    20),
        (Bt_ax,     'Campo Toroidal  B_φ [T]',  'cividis',    20),
        (q_ax,      'Factor de Seguridad  q(ρ)','viridis',    16),
        (nz_ax/1e18,f'Impurezas C⁶⁺  n_z [10¹⁸ m⁻³]',impurity_cmap,18),
        (beta_ax,   'Beta  β [%]',              pressure_cmap,20),
    ]

    for idx, (data, title, cmap, nlev) in enumerate(panels):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        ax.set_facecolor(FACECOLOR); ax.set_aspect('equal')
        levels = np.linspace(np.nanmin(data), np.nanmax(data), nlev)
        if levels[-1] <= levels[0]:
            levels = np.linspace(0, 1, nlev)
        cf = ax.contourf(X_ax, Y_ax, data, levels=levels,
                         cmap=cmap, extend='both')
        ax.contour(X_ax, Y_ax, data, levels=levels[::3],
                   colors='white', linewidths=0.2, alpha=0.2)
        theta_ring = np.linspace(0, 2*np.pi, 200)
        for rr in [0, 1]:
            R_ring = cfg.R0 + cfg.a * rr
            ax.plot(R_ring*np.cos(theta_ring), R_ring*np.sin(theta_ring),
                    'w-', linewidth=0.5+rr*0.5, alpha=0.3)
        cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7, colors='#8899aa')
        cb.outline.set_edgecolor('#334455')
        ax.set_title(title, fontsize=11, color=TITLE_COL, pad=10)
        ax.set_xlabel('X [m]', fontsize=9, color=LABEL_COL)
        ax.set_ylabel('Y [m]', fontsize=9, color=LABEL_COL)
        ax.tick_params(labelsize=7, colors=TICK_COL)
        for s in ax.spines.values(): s.set_color(SPINE_COL)

    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  Radial profiles
# ═══════════════════════════════════════════════════════════════

def plot_radial_profiles(fvm_data, j_phi, j_bs, cfg, filename):
    print("  [PLOT] Generating radial profile comparison...")
    rho = fvm_data['rho']

    fig, axes = plt.subplots(2, 4, figsize=(24, 11), facecolor=FACECOLOR)
    fig.suptitle('PERFILES RADIALES — Equilibrio, Transporte y Corriente',
                 fontsize=15, color='white', fontweight='300', y=0.97)
    for ax in axes.flat:
        _style_ax(ax, xlabel='ρ = r/a')

    # 1. Density
    ax = axes[0,0]
    ax.fill_between(rho, fvm_data['n']/1e20, alpha=0.3, color='#4090ff')
    ax.plot(rho, fvm_data['n']/1e20, color='#4090ff', linewidth=2)
    n_init = np.array([density_profile(r, cfg.n0) for r in rho])
    ax.plot(rho, n_init/1e20, '--', color='#88aadd', linewidth=1, alpha=0.6,
            label='Inicial')
    ax.set_ylabel('n [10²⁰ m⁻³]', fontsize=9, color=LABEL_COL)
    ax.set_title('Densidad', fontsize=11, color=TITLE_COL)
    ax.legend(fontsize=8, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    # 2. Temperature
    ax = axes[0,1]
    ax.fill_between(rho, fvm_data['Ti'], alpha=0.3, color='#ff6b4a')
    ax.plot(rho, fvm_data['Ti'], color='#ff6b4a', linewidth=2, label='Tᵢ')
    ax.plot(rho, fvm_data['Te'], color='#ffaa44', linewidth=1.5, label='Tₑ')
    ax.set_ylabel('T [keV]', fontsize=9, color=LABEL_COL)
    ax.set_title('Temperatura', fontsize=11, color=TITLE_COL)
    ax.legend(fontsize=8, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    # 3. Safety factor
    ax = axes[0,2]
    q_arr = np.array([q_profile(r, cfg.q0, cfg.qa) for r in rho])
    ax.plot(rho, q_arr, color='#00e5a0', linewidth=2)
    ax.fill_between(rho, q_arr, alpha=0.2, color='#00e5a0')
    ax.axhline(y=1.0, color='#ff4444', ls='--', alpha=0.5, lw=1)
    ax.axhline(y=2.0, color='#ffaa44', ls='--', alpha=0.4, lw=1)
    ax.text(0.05, 1.05, 'q=1 (sawteeth)', fontsize=7, color='#ff6666',
            alpha=0.7)
    ax.set_ylabel('q', fontsize=9, color=LABEL_COL)
    ax.set_title('Factor de Seguridad', fontsize=11, color=TITLE_COL)

    # 4. Pressure
    ax = axes[0,3]
    p_arr = fvm_data['n']*(fvm_data['Ti']+fvm_data['Te'])*cfg.keV_to_J/1e3
    ax.fill_between(rho, p_arr, alpha=0.3, color='#b48eff')
    ax.plot(rho, p_arr, color='#b48eff', linewidth=2)
    ax.set_ylabel('p [kPa]', fontsize=9, color=LABEL_COL)
    ax.set_title('Presión', fontsize=11, color=TITLE_COL)

    # 5. Transport
    ax = axes[1,0]
    ax.semilogy(rho[1:], fvm_data['D_cl'][1:], color='#4090ff', lw=1.5,
                label='D_clásico')
    ax.semilogy(rho[1:], fvm_data['D_neo'][1:], color='#ff6b4a', lw=1.5,
                label='D_neoclásico')
    ax.semilogy(rho[1:], fvm_data['chi_n'][1:], color='#00e5a0', lw=2,
                label='D_total')
    ax.set_ylabel('D [m²/s]', fontsize=9, color=LABEL_COL)
    ax.set_title('Coeficientes de Difusión', fontsize=11, color=TITLE_COL)
    ax.legend(fontsize=7, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')
    ax.set_ylim(1e-4, 1e2)

    # 6. Impurity
    ax = axes[1,1]
    ax.fill_between(rho, fvm_data['n_z']/1e18, alpha=0.3, color='#ffd53e')
    ax.plot(rho, fvm_data['n_z']/1e18, color='#ffd53e', lw=2,
            label=f'C⁶⁺ (Z={cfg.Z_imp})')
    ax.set_ylabel('n_z [10¹⁸ m⁻³]', fontsize=9, color=LABEL_COL)
    ax.set_title(f'Impurezas (Z={cfg.Z_imp})', fontsize=11, color=TITLE_COL)
    ax.legend(fontsize=8, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    # 7. Current
    ax = axes[1,2]
    ax.plot(rho, j_phi/1e6, color='#33aabb', lw=2, label='j_ohmic')
    ax.plot(rho, j_bs/1e6, color='#ff6b4a', lw=1.5, label='j_bootstrap')
    ax.plot(rho, (j_phi+j_bs)/1e6, color='#ffffff', lw=1, alpha=0.6,
            label='j_total')
    ax.fill_between(rho, (j_phi+j_bs)/1e6, alpha=0.15, color='#33aabb')
    ax.set_ylabel('j [MA/m²]', fontsize=9, color=LABEL_COL)
    ax.set_title('Densidad de Corriente', fontsize=11, color=TITLE_COL)
    ax.legend(fontsize=7, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    # 8. Time evolution
    ax = axes[1,3]
    n_hist = fvm_data['n_hist']
    colors = plt.cm.cool(np.linspace(0, 1, len(n_hist)))
    for k, nh in enumerate(n_hist):
        ax.plot(rho, nh/1e20, color=colors[k], linewidth=0.8, alpha=0.7)
    ax.set_ylabel('n [10²⁰ m⁻³]', fontsize=9, color=LABEL_COL)
    ax.set_title('Evolución Temporal de n(ρ)', fontsize=11, color=TITLE_COL)
    sm = plt.cm.ScalarMappable(cmap='cool',
                               norm=plt.Normalize(0, cfg.Nt_fvm*cfg.dt_fvm*1e3))
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('t [ms]', fontsize=8, color=LABEL_COL)
    cb.ax.tick_params(labelsize=7, colors='#8899aa')
    cb.outline.set_edgecolor('#334455')

    plt.subplots_adjust(left=0.05, right=0.96, top=0.92, bottom=0.07,
                        hspace=0.32, wspace=0.30)
    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  Particle orbits
# ═══════════════════════════════════════════════════════════════

def plot_particle_orbits(rho_hist, theta_hist, phi_hist, cfg, filename):
    print("  [PLOT] Generating particle orbit projections...")

    fig = plt.figure(figsize=(22, 10), facecolor=FACECOLOR)
    fig.suptitle('ÓRBITAS DE PARTÍCULAS — Proyecciones del Centro de Giro',
                 fontsize=15, color='white', fontweight='300', y=0.97)
    gs = GridSpec(1, 3, figure=fig, wspace=0.25,
                  left=0.06, right=0.96, top=0.88, bottom=0.08)

    Nsave = rho_hist.shape[0]
    Nsnap = rho_hist.shape[1]

    # 1. Poloidal (R, Z)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_facecolor(FACECOLOR); ax1.set_aspect('equal')
    theta_fs = np.linspace(0, 2*np.pi, 200)
    for rr in np.arange(0.1, 1.01, 0.1):
        R_fs = cfg.R0 + cfg.a*rr*np.cos(theta_fs + cfg.delta_tri*np.sin(theta_fs))
        Z_fs = cfg.a*cfg.kappa*rr*np.sin(theta_fs)
        ax1.plot(R_fs, Z_fs, 'w-', lw=0.3, alpha=0.15)
    for i in range(min(Nsave, 80)):
        R_o = cfg.R0 + cfg.a*rho_hist[i,:Nsnap]*np.cos(theta_hist[i,:Nsnap])
        Z_o = cfg.a*cfg.kappa*rho_hist[i,:Nsnap]*np.sin(theta_hist[i,:Nsnap])
        rho0 = rho_hist[i,0]
        col = '#ff6b4a' if 0.3 < rho0 < 0.7 else '#4090ff'
        al  = 0.5 if 0.3 < rho0 < 0.7 else 0.35
        ax1.plot(R_o, Z_o, color=col, lw=0.4, alpha=al)
    ax1.set_title('Proyección Poloidal (R, Z)', fontsize=11, color=TITLE_COL)
    ax1.set_xlabel('R [m]', fontsize=9, color=LABEL_COL)
    ax1.set_ylabel('Z [m]', fontsize=9, color=LABEL_COL)
    ax1.tick_params(labelsize=7, colors=TICK_COL)
    for s in ax1.spines.values(): s.set_color(SPINE_COL)

    # 2. Toroidal (X, Y)
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_facecolor(FACECOLOR); ax2.set_aspect('equal')
    phi_ring = np.linspace(0, 2*np.pi, 200)
    for rr in [0,1]:
        R_r = cfg.R0 + cfg.a*rr
        ax2.plot(R_r*np.cos(phi_ring), R_r*np.sin(phi_ring),
                 'w-', lw=0.5, alpha=0.25)
    for i in range(min(Nsave, 80)):
        R_o = cfg.R0 + cfg.a*rho_hist[i,:Nsnap]*np.cos(theta_hist[i,:Nsnap])
        X_o = R_o*np.cos(phi_hist[i,:Nsnap])
        Y_o = R_o*np.sin(phi_hist[i,:Nsnap])
        col = '#ff6b4a' if 0.3 < rho_hist[i,0] < 0.7 else '#4090ff'
        ax2.plot(X_o, Y_o, color=col, lw=0.3, alpha=0.3)
    ax2.set_title('Proyección Toroidal (X, Y)', fontsize=11, color=TITLE_COL)
    ax2.set_xlabel('X [m]', fontsize=9, color=LABEL_COL)
    ax2.set_ylabel('Y [m]', fontsize=9, color=LABEL_COL)
    ax2.tick_params(labelsize=7, colors=TICK_COL)
    for s in ax2.spines.values(): s.set_color(SPINE_COL)

    # 3. Phase space (ρ, θ)
    ax3 = fig.add_subplot(gs[0,2], projection='polar')
    ax3.set_facecolor(FACECOLOR)
    for i in range(min(Nsave, 60)):
        col = '#ff6b4a' if 0.3 < rho_hist[i,0] < 0.7 else '#4090ff'
        ax3.plot(theta_hist[i,:Nsnap], rho_hist[i,:Nsnap],
                 color=col, lw=0.3, alpha=0.35)
    theta_c = np.linspace(0, 2*np.pi, 100)
    for rr in np.arange(0.2, 1.01, 0.2):
        ax3.plot(theta_c, np.full_like(theta_c, rr), 'w-', lw=0.3, alpha=0.15)
    ax3.set_title('Espacio de Fases (ρ, θ)', fontsize=11, color=TITLE_COL,
                  pad=15)
    ax3.tick_params(labelsize=7, colors=TICK_COL)
    ax3.set_rmax(1.0)
    ax3.grid(True, alpha=0.1, color='#334466')

    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  Multiple toroidal sections
# ═══════════════════════════════════════════════════════════════

def plot_multiple_toroidal_sections(fvm_data, cfg, filename):
    print("  [PLOT] Generating multi-toroidal-section contours...")

    Nsec = cfg.Nphi_sections
    phi_sections = np.linspace(0, 2*np.pi, Nsec, endpoint=False)

    fig = plt.figure(figsize=(24, 12), facecolor=FACECOLOR)
    fig.suptitle(f'SECCIONES TOROIDALES — Temperatura Tᵢ [keV] en {Nsec} cortes',
                 fontsize=15, color='white', fontweight='300', y=0.97)
    nrows, ncols = 2, Nsec // 2
    rho = fvm_data['rho']
    theta = np.linspace(0, 2*np.pi, cfg.Ntheta)
    RHO, THETA = np.meshgrid(rho, theta, indexing='ij')

    for s, phi_s in enumerate(phi_sections):
        ax = fig.add_subplot(nrows, ncols, s+1)
        ax.set_facecolor(FACECOLOR); ax.set_aspect('equal')
        ripple = 1.0 + 0.015 * np.cos(cfg.N_TF * phi_s)
        shafranov = 0.15 * cfg.a * (1.0 - RHO**2) * ripple
        R_g = (cfg.R0 + shafranov
               + cfg.a*RHO*np.cos(THETA + cfg.delta_tri*np.sin(THETA)))
        Z_g = cfg.a * cfg.kappa * RHO * np.sin(THETA)
        T_2d = np.zeros_like(RHO)
        for i in range(cfg.Nr):
            for j in range(cfg.Ntheta):
                rr, th = rho[i], theta[j]
                T_b = temperature_profile(rr, cfg.T0_i)
                q_l = q_profile(rr, cfg.q0, cfg.qa)
                mode = 1.0 + 0.10*rr**2*np.cos(phi_s - th/max(q_l,0.5))
                mode += 0.04*rr**3*np.cos(3*phi_s - 2*th)
                balloon = 1.0 + 0.12*rr**2*np.cos(th)
                T_2d[i,j] = T_b * mode * balloon * ripple
        levels = np.linspace(0, cfg.T0_i*1.3, 22)
        cf = ax.contourf(R_g, Z_g, T_2d, levels=levels,
                         cmap=temp_cmap, extend='both')
        ax.contour(R_g, Z_g, T_2d, levels=levels[::3],
                   colors='white', linewidths=0.2, alpha=0.15)
        for rr in [0.3, 0.6, 0.9]:
            idx_r = int(rr*(cfg.Nr-1))
            ax.plot(R_g[idx_r,:], Z_g[idx_r,:], 'w-', lw=0.4, alpha=0.2)
        ax.set_title(f'φ = {np.degrees(phi_s):.0f}°', fontsize=10,
                     color='#aabbcc', pad=6)
        ax.tick_params(labelsize=6, colors='#556677')
        for sp in ax.spines.values(): sp.set_color('#1a1a30')
        if s == ncols-1 or s == Nsec-1:
            cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6, colors='#8899aa')
            cb.outline.set_edgecolor('#334455')

    plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04,
                        hspace=0.22, wspace=0.18)
    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")


# ═══════════════════════════════════════════════════════════════
#  Transport hierarchy
# ═══════════════════════════════════════════════════════════════

def plot_transport_hierarchy(fvm_data, cfg, filename):
    print("  [PLOT] Generating transport hierarchy comparison...")
    rho = fvm_data['rho']; Nr = len(rho)
    D_cl = fvm_data['D_cl']; D_neo = fvm_data['D_neo']
    D_total = fvm_data['chi_n']
    dr = rho[1]-rho[0]; n = fvm_data['n']

    Gamma_cl = np.zeros(Nr); Gamma_neo = np.zeros(Nr)
    Gamma_total = np.zeros(Nr)
    for i in range(1, Nr-1):
        dn_dr = (n[i+1]-n[i-1]) / (2*dr*cfg.a)
        Gamma_cl[i]    = -D_cl[i]*dn_dr
        Gamma_neo[i]   = -D_neo[i]*dn_dr
        Gamma_total[i] = -D_total[i]*dn_dr

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=FACECOLOR)
    fig.suptitle('JERARQUÍA DE TRANSPORTE: Γ_Banana >> Γ_PS >> Γ_Clásico',
                 fontsize=15, color='white', fontweight='300', y=0.97)
    for ax in axes:
        _style_ax(ax, xlabel='ρ = r/a')

    # Panel 1
    ax = axes[0]
    ax.semilogy(rho[2:], D_cl[2:], color='#4090ff', lw=2, label='D_clásico')
    ax.semilogy(rho[2:], D_neo[2:], color='#ff6b4a', lw=2, label='D_neoclásico')
    ax.semilogy(rho[2:], D_total[2:], color='#00e5a0', lw=2.5,
                label='D_total (+ anómalo)')
    ax.fill_between(rho[2:], D_cl[2:], D_neo[2:], alpha=0.1, color='#ff6b4a')
    ax.fill_between(rho[2:], D_neo[2:], D_total[2:], alpha=0.1, color='#00e5a0')
    ax.set_ylabel('D [m²/s]', fontsize=10, color=LABEL_COL)
    ax.set_title('Coeficientes de Difusión', fontsize=12, color=TITLE_COL)
    ax.legend(fontsize=9, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')
    ax.set_ylim(1e-4, 1e2)

    # Panel 2
    ax = axes[1]
    ax.plot(rho[2:-2], np.abs(Gamma_cl[2:-2]), color='#4090ff', lw=2,
            label='|Γ_clásico|')
    ax.plot(rho[2:-2], np.abs(Gamma_neo[2:-2]), color='#ff6b4a', lw=2,
            label='|Γ_neoclásico|')
    ax.plot(rho[2:-2], np.abs(Gamma_total[2:-2]), color='#00e5a0', lw=2.5,
            label='|Γ_total|')
    ax.set_yscale('log')
    ax.set_ylabel('|Γ| [m⁻² s⁻¹]', fontsize=10, color=LABEL_COL)
    ax.set_title('Flujos de Partículas', fontsize=12, color=TITLE_COL)
    ax.legend(fontsize=9, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    # Panel 3
    ax = axes[2]
    ratio_neo = np.zeros(Nr); ratio_total = np.zeros(Nr)
    for i in range(Nr):
        if D_cl[i] > 1e-10:
            ratio_neo[i]   = D_neo[i]/D_cl[i]
            ratio_total[i] = D_total[i]/D_cl[i]
    ax.semilogy(rho[2:], ratio_neo[2:], color='#ff6b4a', lw=2,
                label='D_neo / D_cl')
    ax.semilogy(rho[2:], ratio_total[2:], color='#00e5a0', lw=2,
                label='D_total / D_cl')
    ax.axhline(y=1, color='#4090ff', ls='--', alpha=0.4, lw=1)
    ax.axhspan(1, 10, alpha=0.05, color='#4090ff')
    ax.axhspan(10, 100, alpha=0.05, color='#ff6b4a')
    ax.axhspan(100, 10000, alpha=0.05, color='#00e5a0')
    ax.text(0.5, 3, 'Pfirsch-Schlüter', fontsize=8, color='#6688aa',
            ha='center')
    ax.text(0.5, 40, 'Banana', fontsize=8, color='#ff8866', ha='center')
    ax.text(0.5, 500, 'Anómalo', fontsize=8, color='#44cc88', ha='center')
    ax.set_ylabel('D / D_clásico', fontsize=10, color=LABEL_COL)
    ax.set_title('Factor de Amplificación', fontsize=12, color=TITLE_COL)
    ax.legend(fontsize=9, facecolor='#0e0e1a', edgecolor='#334455',
              labelcolor='#aabbcc')

    plt.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.10,
                        wspace=0.25)
    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] Saved: {filename}")
