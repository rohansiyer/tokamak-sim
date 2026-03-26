#!/usr/bin/env python3
"""
3D Tokamak Animation — loads tokamak_3d_data.npz and creates
an interactive 3D visualization using PyVista (VTK-based).

Usage:
    python animate_tokamak.py                        # interactive window
    python animate_tokamak.py --save tokamak.mp4     # save to video
    python animate_tokamak.py --save tokamak.gif     # save to GIF
    python animate_tokamak.py --field temperature     # select field
    python animate_tokamak.py --field density
    python animate_tokamak.py --field pressure
    python animate_tokamak.py --field B_magnitude
    python animate_tokamak.py --field fusion_rate
    python animate_tokamak.py --animate-orbits        # animated particle orbits

Requirements:
    pip install pyvista numpy
    # For video export: pip install imageio[ffmpeg]
"""

import argparse
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista not installed. Run: pip install pyvista")
    print("       For video export also: pip install imageio[ffmpeg]")
    exit(1)


def load_data(filepath='tokamak_3d_data.npz'):
    """Load simulation data."""
    print(f"Loading {filepath}...")
    data = np.load(filepath)
    print(f"  Arrays found: {list(data.keys())}")
    return data


def build_tokamak_mesh(data, field_name='temperature', phi_skip=1):
    """
    Build a PyVista StructuredGrid from the toroidal mesh data.
    
    The mesh is (Nr, Ntheta, Nphi) → close the torus by appending
    the first phi slice at the end.
    """
    X = data['X'][:, :, ::phi_skip]
    Y = data['Y'][:, :, ::phi_skip]
    Z = data['Z_cart'][:, :, ::phi_skip]
    field = data[field_name][:, :, ::phi_skip]

    Nr, Ntheta, Nphi = X.shape

    # Close the torus: append first phi slice
    X = np.concatenate([X, X[:, :, 0:1]], axis=2)
    Y = np.concatenate([Y, Y[:, :, 0:1]], axis=2)
    Z = np.concatenate([Z, Z[:, :, 0:1]], axis=2)
    field = np.concatenate([field, field[:, :, 0:1]], axis=2)

    # Close poloidal direction too
    X = np.concatenate([X, X[:, 0:1, :]], axis=1)
    Y = np.concatenate([Y, Y[:, 0:1, :]], axis=1)
    Z = np.concatenate([Z, Z[:, 0:1, :]], axis=1)
    field = np.concatenate([field, field[:, 0:1, :]], axis=1)

    # F-order for VTK
    grid = pv.StructuredGrid(X, Y, Z)
    grid[field_name] = field.flatten(order='F')

    return grid


def build_outer_surface(data, field_name='temperature', phi_skip=1):
    """Extract just the outer flux surface (ρ=1) as a surface mesh."""
    X = data['X'][-1, :, ::phi_skip]
    Y = data['Y'][-1, :, ::phi_skip]
    Z = data['Z_cart'][-1, :, ::phi_skip]
    field = data[field_name][-1, :, ::phi_skip]

    Ntheta, Nphi = X.shape

    # Close torus and poloidal
    X = np.concatenate([X, X[:, 0:1]], axis=1)
    Y = np.concatenate([Y, Y[:, 0:1]], axis=1)
    Z = np.concatenate([Z, Z[:, 0:1]], axis=1)
    field = np.concatenate([field, field[:, 0:1]], axis=1)

    X = np.concatenate([X, X[0:1, :]], axis=0)
    Y = np.concatenate([Y, Y[0:1, :]], axis=0)
    Z = np.concatenate([Z, Z[0:1, :]], axis=0)
    field = np.concatenate([field, field[0:1, :]], axis=0)

    grid = pv.StructuredGrid(X, Y, Z)
    grid[field_name] = field.flatten(order='F')
    return grid


def build_midplane_surface(data, field_name='temperature',
                           rho_idx_range=None):
    """Build a midplane (θ=0) surface coloured by a field."""
    if rho_idx_range is None:
        rho_idx_range = slice(None)

    Nr = data['X'].shape[0]
    Nphi = data['X'].shape[2]

    # θ=0 slice
    X = data['X'][rho_idx_range, 0, :]
    Y = data['Y'][rho_idx_range, 0, :]
    Z = data['Z_cart'][rho_idx_range, 0, :]
    field = data[field_name][rho_idx_range, 0, :]

    # Close toroidally
    X = np.concatenate([X, X[:, 0:1]], axis=1)
    Y = np.concatenate([Y, Y[:, 0:1]], axis=1)
    Z = np.concatenate([Z, Z[:, 0:1]], axis=1)
    field = np.concatenate([field, field[:, 0:1]], axis=1)

    grid = pv.StructuredGrid(X, Y, Z)
    grid[field_name] = field.flatten(order='F')
    return grid


def build_particle_trails(data, n_particles=50):
    """Build polylines for particle orbits."""
    pX = data['particle_X'][:n_particles]
    pY = data['particle_Y'][:n_particles]
    pZ = data['particle_Z'][:n_particles]

    lines = []
    for i in range(min(n_particles, pX.shape[0])):
        # Filter out zero points
        mask = (pX[i] != 0) | (pY[i] != 0) | (pZ[i] != 0)
        if mask.sum() < 2:
            continue
        pts = np.column_stack([pX[i][mask], pY[i][mask], pZ[i][mask]])
        n = len(pts)
        cells = np.zeros(n + 1, dtype=int)
        cells[0] = n
        cells[1:] = np.arange(n)
        line = pv.PolyData(pts)
        line.lines = cells
        lines.append(line)

    if lines:
        return lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]
    return None


def build_poloidal_cut(data, phi_index, field_name='temperature'):
    """Extract a poloidal cross-section at a given toroidal angle."""
    X = data['X'][:, :, phi_index]
    Y = data['Y'][:, :, phi_index]
    Z = data['Z_cart'][:, :, phi_index]
    field = data[field_name][:, :, phi_index]

    # Close poloidal
    X = np.concatenate([X, X[:, 0:1]], axis=1)
    Y = np.concatenate([Y, Y[:, 0:1]], axis=1)
    Z = np.concatenate([Z, Z[:, 0:1]], axis=1)
    field = np.concatenate([field, field[:, 0:1]], axis=1)

    grid = pv.StructuredGrid(X, Y, Z)
    grid[field_name] = field.flatten(order='F')
    return grid


FIELD_LABELS = {
    'temperature': 'Temperature [keV]',
    'density':     'Density [m⁻³]',
    'pressure':    'Pressure [Pa]',
    'B_magnitude': '|B| [T]',
    'current':     'Current j [A/m²]',
    'fusion_rate': 'Fusion rate [m⁻³s⁻¹]',
    'q_factor':    'Safety factor q',
}

FIELD_CMAPS = {
    'temperature': 'inferno',
    'density':     'viridis',
    'pressure':    'magma',
    'B_magnitude': 'cividis',
    'current':     'cool',
    'fusion_rate': 'hot',
    'q_factor':    'plasma',
}


# ═══════════════════════════════════════════════════════════════
#  Animated orbit visualization
# ═══════════════════════════════════════════════════════════════

def create_animated_orbit_visualization(data, n_particles=40, trail_length=8):
    """
    Animate guiding-center particle orbits in real time.

    Steps through particle_X/Y/Z snapshot arrays using a PyVista callback.
    Shows current positions as glowing spheres and fading trail lines.
    """
    print("\nBuilding animated orbit visualization...")

    pX = data['particle_X']
    pY = data['particle_Y']
    pZ = data['particle_Z']

    Npart_total, Nsnap = pX.shape
    n_particles = min(n_particles, Npart_total)

    # Filter to particles with valid data
    valid = []
    for i in range(Npart_total):
        mask = (pX[i] != 0) | (pY[i] != 0) | (pZ[i] != 0)
        if mask.sum() > Nsnap * 0.5:
            valid.append(i)
        if len(valid) >= n_particles:
            break

    if not valid:
        print("  ERROR: No valid particle data found.")
        return

    particle_ids = valid
    n_particles = len(particle_ids)
    print(f"  Animating {n_particles} particles over {Nsnap} snapshots "
          f"(trail length: {trail_length})")

    # Classify orbits: banana vs passing based on theta range
    # (Banana orbits have limited poloidal angle range)
    colors_per_particle = []
    for pid in particle_ids:
        # Use the particle data to determine type
        r_vals = np.sqrt(pX[pid]**2 + pY[pid]**2)
        z_vals = pZ[pid]
        mask = (pX[pid] != 0) | (pY[pid] != 0) | (pZ[pid] != 0)
        if mask.sum() < 5:
            colors_per_particle.append('#ffffff')
            continue
        z_range = np.ptp(z_vals[mask])
        r_range = np.ptp(r_vals[mask])
        # Banana orbits have larger Z excursion relative to R
        if z_range > 0.3 * r_range:
            colors_per_particle.append('#ff6b4a')  # red-orange for banana
        else:
            colors_per_particle.append('#4ac0ff')  # blue for passing

    # Set up plotter
    pl = pv.Plotter(window_size=[1920, 1080])
    pl.set_background('#0a0a1a')

    # Add semi-transparent outer surface for context
    print("  Building outer surface...")
    outer = build_outer_surface(data, 'temperature', phi_skip=1)
    pl.add_mesh(outer, scalars='temperature', cmap='inferno',
                opacity=0.12, smooth_shading=True, show_scalar_bar=False)

    # Add one poloidal cross-section for context
    Nphi = data['X'].shape[2]
    cut = build_poloidal_cut(data, 0, 'temperature')
    pl.add_mesh(cut, scalars='temperature', cmap='inferno',
                opacity=0.2, show_scalar_bar=False)

    # Initial particle point cloud
    snap_0 = 0
    pts_0 = np.column_stack([
        pX[particle_ids, snap_0],
        pY[particle_ids, snap_0],
        pZ[particle_ids, snap_0],
    ])
    point_cloud = pv.PolyData(pts_0)

    # Color array for particles
    particle_rgba = np.zeros((n_particles, 4), dtype=np.uint8)
    for idx, color_hex in enumerate(colors_per_particle):
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        particle_rgba[idx] = [r, g, b, 255]
    point_cloud['colors'] = particle_rgba

    particle_actor = pl.add_mesh(
        point_cloud, scalars='colors', rgb=True,
        point_size=10, render_points_as_spheres=True,
        show_scalar_bar=False
    )

    # Trail storage — list of line actors
    trail_actors = []

    # Time label
    time_actor = pl.add_text(
        f"Snapshot: 0/{Nsnap}  |  Particles: {n_particles}",
        position='upper_right', font_size=11, color='white',
        name='time_label'
    )

    # Title
    pl.add_text('TOKAMAK — Animated Guiding-Center Orbits',
                position='upper_left', font_size=12, color='white')

    # Legend
    pl.add_text('🔴 Banana orbits   🔵 Passing orbits',
                position='lower_left', font_size=10, color='#aaaaaa')

    # Camera
    R0 = float(data['config_R0'])
    pl.camera.position = (R0 * 2.2, R0 * 2.2, R0 * 1.8)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    # Animation state
    state = {'frame': 0, 'playing': True, 'speed': 1}

    def update_frame(frame_idx):
        """Update particle positions and trails for a given frame."""
        # Remove old trail actors
        for actor in trail_actors:
            pl.remove_actor(actor)
        trail_actors.clear()

        # Current positions
        pts = np.column_stack([
            pX[particle_ids, frame_idx],
            pY[particle_ids, frame_idx],
            pZ[particle_ids, frame_idx],
        ])
        point_cloud.points = pts

        # Draw trails (fading lines from past snapshots)
        trail_start = max(0, frame_idx - trail_length)
        if frame_idx > trail_start:
            for p_idx, pid in enumerate(particle_ids):
                trail_pts = np.column_stack([
                    pX[pid, trail_start:frame_idx + 1],
                    pY[pid, trail_start:frame_idx + 1],
                    pZ[pid, trail_start:frame_idx + 1],
                ])
                # Skip if degenerate
                if len(trail_pts) < 2:
                    continue
                # Check for zero (invalid) points
                valid = np.any(trail_pts != 0, axis=1)
                trail_pts = trail_pts[valid]
                if len(trail_pts) < 2:
                    continue

                n = len(trail_pts)
                cells = np.zeros(n + 1, dtype=int)
                cells[0] = n
                cells[1:] = np.arange(n)
                trail = pv.PolyData(trail_pts)
                trail.lines = cells

                color = colors_per_particle[p_idx]
                # Fade opacity based on trail position
                opacity = 0.3
                actor = pl.add_mesh(
                    trail, color=color, line_width=1.5,
                    opacity=opacity, show_scalar_bar=False
                )
                trail_actors.append(actor)

        # Update time label
        pl.add_text(
            f"Snapshot: {frame_idx + 1}/{Nsnap}  |  "
            f"Particles: {n_particles}  |  "
            f"{'▶ Playing' if state['playing'] else '⏸ Paused'}",
            position='upper_right', font_size=11, color='white',
            name='time_label'
        )

    def callback(step):
        """Timer callback for animation advancement."""
        if not state['playing']:
            return
        state['frame'] = (state['frame'] + state['speed']) % Nsnap
        update_frame(state['frame'])

    def toggle_play():
        state['playing'] = not state['playing']

    def speed_up():
        state['speed'] = min(state['speed'] + 1, 5)
        print(f"  Speed: {state['speed']}x")

    def slow_down():
        state['speed'] = max(state['speed'] - 1, 1)
        print(f"  Speed: {state['speed']}x")

    # Key bindings
    pl.add_key_event('space', toggle_play)
    pl.add_key_event('Right', speed_up)
    pl.add_key_event('Left', slow_down)

    print("\n  Controls:")
    print("    Space     = Play/Pause")
    print("    Right/Left = Speed up/down")
    print("    Left-click drag = Rotate")
    print("    Scroll = Zoom")
    print("    Right-click = Pan")
    print()

    # Start animation timer (update every 80ms = ~12.5fps)
    pl.add_callback(callback, interval=80)

    pl.show()


# ═══════════════════════════════════════════════════════════════
#  Static visualization (original)
# ═══════════════════════════════════════════════════════════════

def create_visualization(data, field_name='temperature', save_path=None):
    """Create the full 3D tokamak visualization."""
    print(f"\nBuilding 3D visualization (field: {field_name})...")

    cmap = FIELD_CMAPS.get(field_name, 'viridis')
    label = FIELD_LABELS.get(field_name, field_name)

    pl = pv.Plotter(window_size=[1920, 1080])
    pl.set_background('#0a0a1a')

    # 1. Outer surface (semi-transparent)
    print("  Building outer surface...")
    outer = build_outer_surface(data, field_name, phi_skip=1)
    pl.add_mesh(outer, scalars=field_name, cmap=cmap, opacity=0.3,
                smooth_shading=True, show_scalar_bar=False)

    # 2. Two poloidal cross-sections (90° apart)
    print("  Building poloidal cross-sections...")
    Nphi = data['X'].shape[2]
    for phi_idx in [0, Nphi // 4]:
        cut = build_poloidal_cut(data, phi_idx, field_name)
        pl.add_mesh(cut, scalars=field_name, cmap=cmap, opacity=0.8,
                    show_scalar_bar=False)

    # 3. Midplane surface
    print("  Building midplane surface...")
    midplane = build_midplane_surface(data, field_name)
    pl.add_mesh(midplane, scalars=field_name, cmap=cmap, opacity=0.6,
                show_scalar_bar=True,
                scalar_bar_args={'title': label, 'color': 'white',
                                 'title_font_size': 14, 'label_font_size': 10})

    # 4. Particle orbits
    print("  Building particle trails...")
    trails = build_particle_trails(data, n_particles=30)
    if trails is not None:
        pl.add_mesh(trails, color='#ff6b4a', line_width=1.5, opacity=0.6)

    # Camera
    R0 = float(data['config_R0'])
    pl.camera.position = (R0 * 2.5, R0 * 2.5, R0 * 1.5)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    pl.add_text('TOKAMAK FUSION REACTOR — 3D Visualization',
                position='upper_left', font_size=12, color='white')

    if save_path:
        print(f"\n  Rendering orbiting camera to {save_path}...")
        pl.open_movie(save_path) if save_path.endswith('.mp4') else None

        if save_path.endswith('.gif') or save_path.endswith('.mp4'):
            path = pl.generate_orbital_path(n_points=120, shift=R0*0.5)
            pl.open_gif(save_path) if save_path.endswith('.gif') else None
            if save_path.endswith('.mp4'):
                pl.open_movie(save_path, framerate=30)
            pl.orbit_on_path(path, write_frames=True)
            print(f"  Saved: {save_path}")
        else:
            pl.screenshot(save_path)
            print(f"  Screenshot saved: {save_path}")
    else:
        print("\n  Opening interactive window...")
        print("  Controls: Left-click drag = rotate, Scroll = zoom, "
              "Right-click = pan")
        pl.show()


def main():
    parser = argparse.ArgumentParser(
        description='3D Tokamak Animation from simulation data')
    parser.add_argument('--data', default='tokamak_3d_data.npz',
                        help='Path to .npz data file')
    parser.add_argument('--field', default='temperature',
                        choices=list(FIELD_LABELS.keys()),
                        help='Scalar field to visualize')
    parser.add_argument('--save', default=None,
                        help='Save to file (.mp4, .gif, or .png)')
    parser.add_argument('--animate-orbits', action='store_true',
                        help='Animate particle orbits over time')
    parser.add_argument('--n-particles', type=int, default=40,
                        help='Number of particles to animate (default: 40)')
    parser.add_argument('--trail-length', type=int, default=8,
                        help='Number of past snapshots for trail (default: 8)')
    args = parser.parse_args()

    data = load_data(args.data)

    if args.animate_orbits:
        create_animated_orbit_visualization(
            data, n_particles=args.n_particles,
            trail_length=args.trail_length
        )
    else:
        create_visualization(data, field_name=args.field, save_path=args.save)


if __name__ == '__main__':
    main()
