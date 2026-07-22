import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---- World / corridor params ----
L = 20.0                 # visible corridor length
H = 4.0                  # corridor height
wall_y_top = H / 2
wall_y_bot = -H / 2

# "Corridor moves left" speed (ego frame)
scroll_speed = 0.12

# Vehicle parameters
x_vehicle = 4.0          # vehicle stays fixed in ego frame
y_vehicle = 0.0
vehicle_speed_t = 0.018  # how fast it moves along the local Bezier in parameter t

# Spawning parameters for wall points
spawn_distance = 7.0     # keep wall points separated along x
alternate = True         # alternate top/bottom spawn for variety

# ---- Plot setup ----
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-H/2, H/2)
ax.set_aspect('equal')
ax.axis('off')

# Corridor walls
ax.plot([0, L], [wall_y_top, wall_y_top], color='black', linewidth=2)
ax.plot([0, L], [wall_y_bot, wall_y_bot], color='black', linewidth=2)

# Artists
curve_line, = ax.plot([], [], linewidth=3)
pts_scatter = ax.scatter([], [], s=60, zorder=5)
vehicle_dot, = ax.plot([x_vehicle], [y_vehicle], 'ko', markersize=9, zorder=6)

# ---- Bezier helpers ----
def quad_bezier(P0, P1, P2, t):
    # P0,P1,P2 are (2,) arrays; t can be scalar or (N,1)
    return (1 - t)**2 * P0 + 2*(1 - t)*t * P1 + t**2 * P2

def spawn_wall_point(x):
    global alternate
    y = wall_y_top if alternate else wall_y_bot
    alternate = not alternate
    return [x, y]

# Active wall points: we maintain exactly two ahead of the vehicle (when possible)
wall_points = [
    spawn_wall_point(L * 0.65),
    spawn_wall_point(L * 0.90),
]

t_vehicle = 0.0  # progress along current curve

def update(frame):
    global wall_points, t_vehicle

    # 1) Move wall points left (corridor scrolling past vehicle)
    for p in wall_points:
        p[0] -= scroll_speed

    # 2) Remove any wall point that has reached/passed the vehicle x
    wall_points = [p for p in wall_points if p[0] > x_vehicle + 0.2]

    # 3) Spawn new wall points at the far right so we always have 2 ahead
    #    and roughly maintain spacing.
    while len(wall_points) < 2:
        x_spawn = L
        # ensure spacing from last point if it exists
        if wall_points:
            x_spawn = max(L, wall_points[-1][0] + spawn_distance)
        wall_points.append(spawn_wall_point(x_spawn))

    # 4) Define the quadratic Bezier points:
    #    P0 = vehicle position (center-ish), P1/P2 = wall points (constraints)
    P0 = np.array([x_vehicle, y_vehicle], dtype=float)
    P1 = np.array(wall_points[0], dtype=float)
    P2 = np.array(wall_points[1], dtype=float)

    # 5) Draw curve
    t = np.linspace(0, 1, 250)[:, None]
    curve = quad_bezier(P0, P1, P2, t)
    curve_line.set_data(curve[:, 0], curve[:, 1])

    # 6) Move vehicle a bit along the curve (parameter space)
    t_vehicle += vehicle_speed_t
    if t_vehicle > 1.0:
        # once it "finishes" the segment, restart (receding horizon replans anyway)
        t_vehicle = 0.0

    vpos = quad_bezier(P0, P1, P2, t_vehicle)
    vehicle_dot.set_data([vpos[0]], [vpos[1]])

    # 7) Plot points (vehicle + wall control points)
    pts = np.vstack([P0, P1, P2])
    pts_scatter.set_offsets(pts)

    return curve_line, pts_scatter, vehicle_dot

ani = FuncAnimation(fig, update, frames=800, interval=30, blit=True)
plt.show()

