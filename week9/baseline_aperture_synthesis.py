import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from global_land_mask import globe
import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Earth-rotation aperture synthesis animation")
    parser.add_argument('--dec', type=float, default=55.0,
                         help="Declination of the target radio source, in degrees (default: 55.0)")
    parser.add_argument('--save', metavar='PATH',
                         help="Export the animation to PATH (.mp4 or .gif) instead of displaying it interactively")
    parser.add_argument('--fps', type=int, default=20, help="Frames per second for the exported file (default: 20)")
    parser.add_argument('--resolution', default='1280x720', metavar='WIDTHxHEIGHT',
                         help="Pixel resolution for the exported file, e.g. 1280x720 for 720p (default: 1280x720)")
    parser.add_argument('--baseline', type=str, default='GBT-EFF',help="Baseline to use for the animation (default: GBT-EFF)")
    return parser.parse_args()

args = parse_args()

# 1. Define Station Coordinates (Latitude, Longitude in degrees)
if args.baseline.upper() == 'GBT-EFF':
    # GBT-EFF baseline is a classic long baseline in radio astronomy, connecting the Green Bank Telescope in West Virginia, USA, with the Effelsberg 100-m Radio Telescope in Germany. This baseline provides a large separation for high-resolution imaging.
    # Station 1: Green Bank, WV, USA
    # Station 2: Effelsberg, Germany
    lat1, lon1 = np.radians(38.43), np.radians(-79.84)
    lat2, lon2 = np.radians(50.52), np.radians(6.88)
elif args.baseline.upper() == 'JBO-EFF':
    # JBO-EFF baseline is another classic long baseline in radio astronomy, connecting the Jodrell Bank Observatory in Cheshire, UK, with the Effelsberg 100-m Radio Telescope in Germany. This baseline also provides a large separation for high-resolution imaging.
    # Station 1: Jodrell Bank, Cheshire, UK
    # Station 2: Effelsberg, Germany
    lat1, lon1 = np.radians(53.236), np.radians(-2.307)
    lat2, lon2 = np.radians(50.52), np.radians(6.88)


R_earth = 6371.0  # Earth radius in km

# Convert geodetic coordinates to Earth-Centered Earth-Fixed (ECEF) XYZ
def geo_to_ecef(lat, lon):
    x = R_earth * np.cos(lat) * np.cos(lon)
    y = R_earth * np.cos(lat) * np.sin(lon)
    z = R_earth * np.sin(lat)
    return np.array([x, y, z])

pos1 = geo_to_ecef(lat1, lon1)
pos2 = geo_to_ecef(lat2, lon2)

# Baseline vector in Earth-fixed coordinates
bx, by, bz = pos1 - pos2

# Build a coarse land-mask grid so the rotating Earth disk shows recognizable
# continents. This makes the rotation of the Earth-fixed baseline visually
# intuitive instead of just an empty shaded circle.
grid_step_deg = 2.0
grid_lats = np.arange(-90, 90, grid_step_deg)
grid_lons = np.arange(-180, 180, grid_step_deg)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)
is_land = globe.is_land(grid_lat_mesh, grid_lon_mesh)

land_lat = np.radians(grid_lat_mesh[is_land])
land_lon = np.radians(grid_lon_mesh[is_land])
land_x, land_y, land_z = geo_to_ecef(land_lat, land_lon)

# 2. Define Source Position (Declination)
dec = np.radians(args.dec)  # Declination of the target radio source

# 3. Compute the u-v tracks over a full 24-hour rotation (Hour Angle from -12h to +12h)
num_points = 180
hour_angles = np.linspace(-np.pi, np.pi, num_points)

u_points = np.sin(hour_angles) * bx + np.cos(hour_angles) * by
v_points = -np.sin(dec) * np.cos(hour_angles) * bx + np.sin(dec) * np.sin(hour_angles) * by + np.cos(dec) * bz

# A station can only observe the source when the source is above its local
# horizon, i.e. when the station's position vector has a positive component
# along the source direction (the same w/line-of-sight test used for land
# masking). The baseline (and its u-v sample) is only valid when both
# stations satisfy this.
def station_w(pos, ha):
    return np.cos(dec) * np.cos(ha) * pos[0] - np.cos(dec) * np.sin(ha) * pos[1] + np.sin(dec) * pos[2]

w1_points = station_w(pos1, hour_angles)
w2_points = station_w(pos2, hour_angles)
baseline_visible = (w1_points > 0) & (w2_points > 0)

# Break the track lines with NaNs wherever the baseline is hidden, so tracing
# stops instead of drawing a spurious segment across the gap.
u_points_track = np.where(baseline_visible, u_points, np.nan)
v_points_track = np.where(baseline_visible, v_points, np.nan)

# 4. Set up the Matplotlib Figure for Animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(f"Earth-Rotation Aperture Synthesis (Source Dec = {args.dec:.1f}\u00b0)", fontsize=14, fontweight='bold')

# Left Axis: View from the Source (Projected Earth & Baseline)
ax1.set_title("Baseline (Viewed from Source)")
ax1.set_xlabel("East-West Projection (km)")
ax1.set_ylabel("North-South Projection (km)")
ax1.set_xlim(-R_earth*1.2, R_earth*1.2)
ax1.set_ylim(-R_earth*1.2, R_earth*1.2)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.5)

# Draw an outline of the Earth disk as seen from space
earth_outline = plt.Circle((0, 0), R_earth, color='skyblue', alpha=0.3, label='Earth Disk')
ax1.add_patch(earth_outline)

# Scatter of land-mask points, updated each frame to show the Earth rotating
# beneath the fixed source direction (only the near/sunlit-facing hemisphere
# as seen from the source is drawn).
land_scatter = ax1.scatter([], [], s=2, color='darkgreen', alpha=0.7, label='Land Mass')

# Animation elements for the left plot
baseline_line, = ax1.plot([], [], 'o-', color='crimson', lw=2.5, markersize=6, label='Baseline')
station1_dot, = ax1.plot([], [], 'o', color='navy', markersize=5)
station2_dot, = ax1.plot([], [], 'o', color='navy', markersize=5)

# Right Axis: The u-v Plane Track
ax2.set_title("Corresponding u-v Track")
ax2.set_xlabel("u (Spatial Frequency / Baseline Projection East in km)")
ax2.set_ylabel("v (Spatial Frequency / Baseline Projection North in km)")
max_uv = max(np.max(np.abs(u_points)), np.max(np.abs(v_points))) * 1.2
ax2.set_xlim(max_uv, -max_uv)  # East is traditionally inverted in radio astronomy (u points left)
ax2.set_ylim(-max_uv, max_uv)
ax2.set_aspect('equal')
ax2.grid(True, linestyle='--', alpha=0.5)

# Animation elements for the right plot
uv_track_line, = ax2.plot([], [], color='teal', alpha=0.4, lw=1.5, label='Full 24h Track')
uv_conjugate_line, = ax2.plot([], [], color='orange', alpha=0.4, lw=1.5, label='Conjugate (-u, -v)')
uv_current_dot, = ax2.plot([], [], 'o', color='teal', markersize=7, label='Current Sampling')
uv_conj_current_dot, = ax2.plot([], [], 'o', color='orange', markersize=7)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

# 5. Animation Initialization
def init():
    baseline_line.set_data([], [])
    station1_dot.set_data([], [])
    station2_dot.set_data([], [])
    land_scatter.set_offsets(np.empty((0, 2)))
    uv_track_line.set_data([], [])
    uv_conjugate_line.set_data([], [])
    uv_current_dot.set_data([], [])
    uv_conj_current_dot.set_data([], [])
    return baseline_line, station1_dot, station2_dot, land_scatter, uv_track_line, uv_current_dot

# 6. Animation Update Function
def update(frame):
    # Current indices up to the current frame
    ha = hour_angles[frame]
    
    # Calculate current positions of stations projected from the source's view
    # Station 1 projected coordinates
    u1 = np.sin(ha) * pos1[0] + np.cos(ha) * pos1[1]
    v1 = -np.sin(dec) * np.cos(ha) * pos1[0] + np.sin(dec) * np.sin(ha) * pos1[1] + np.cos(dec) * pos1[2]
    
    # Station 2 projected coordinates
    u2 = np.sin(ha) * pos2[0] + np.cos(ha) * pos2[1]
    v2 = -np.sin(dec) * np.cos(ha) * pos2[0] + np.sin(dec) * np.sin(ha) * pos2[1] + np.cos(dec) * pos2[2]
    
    # Only draw each station (and the baseline joining them) when the source
    # is above that station's local horizon.
    w1 = station_w(pos1, ha)
    w2 = station_w(pos2, ha)
    station1_dot.set_data([u1] if w1 > 0 else [], [v1] if w1 > 0 else [])
    station2_dot.set_data([u2] if w2 > 0 else [], [v2] if w2 > 0 else [])
    if w1 > 0 and w2 > 0:
        baseline_line.set_data([u1, u2], [v1, v2])
    else:
        baseline_line.set_data([], [])

    # Project land points into the same (u, v) view plane, using the w
    # (line-of-sight) component to keep only the hemisphere facing the source
    # -- this is what makes the land masses appear to rotate with the Earth.
    land_u = np.sin(ha) * land_x + np.cos(ha) * land_y
    land_v = (-np.sin(dec) * np.cos(ha) * land_x + np.sin(dec) * np.sin(ha) * land_y
              + np.cos(dec) * land_z)
    # w = u x v (right-handed line-of-sight axis); note the minus sign on the
    # Y term, required to be consistent with the u/v rotation above.
    land_w = (np.cos(dec) * np.cos(ha) * land_x - np.cos(dec) * np.sin(ha) * land_y
              + np.sin(dec) * land_z)
    visible = land_w > 0
    land_scatter.set_offsets(np.column_stack((land_u[visible], land_v[visible])))
    
    # Update growing u-v tracks (including the symmetric conjugate track -u, -v),
    # which stop tracing wherever the baseline is hidden behind the Earth
    uv_track_line.set_data(u_points_track[:frame+1], v_points_track[:frame+1])
    uv_conjugate_line.set_data(-u_points_track[:frame+1], -v_points_track[:frame+1])
    
    # Update current sample spot indicator (hidden when the baseline is not visible)
    if baseline_visible[frame]:
        uv_current_dot.set_data([u_points[frame]], [v_points[frame]])
        uv_conj_current_dot.set_data([-u_points[frame]], [-v_points[frame]])
    else:
        uv_current_dot.set_data([], [])
        uv_conj_current_dot.set_data([], [])
    
    return (baseline_line, station1_dot, station2_dot, land_scatter, uv_track_line,
            uv_conjugate_line, uv_current_dot, uv_conj_current_dot)

# 7. Run and Display Animation, or Export to a Presentation-Friendly File
ani = FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=50)
plt.tight_layout()

if args.save:
    ext = os.path.splitext(args.save)[1].lower()
    if ext == '.gif':
        writer = PillowWriter(fps=args.fps)
    elif ext == '.mp4':
        if shutil.which('ffmpeg') is None:
            raise SystemExit("Exporting to .mp4 requires ffmpeg to be installed and available on PATH "
                              "(e.g. `brew install ffmpeg`), or use a .gif path instead.")
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    else:
        raise SystemExit(f"Unsupported export format '{ext}'. Use a .mp4 or .gif file extension.")

    # H.264 (used by FFMpegWriter) requires even width/height, so round the
    # requested resolution and resize the figure to match it exactly at export.
    width_px, height_px = (int(v) for v in args.resolution.lower().split('x'))
    width_px -= width_px % 2
    height_px -= height_px % 2
    export_dpi = 100
    fig.set_size_inches(width_px / export_dpi, height_px / export_dpi)
    plt.tight_layout()

    print(f"Saving animation to {args.save} at {width_px}x{height_px} ...")
    ani.save(args.save, writer=writer, dpi=export_dpi)
    print("Done.")
else:
    plt.show()
