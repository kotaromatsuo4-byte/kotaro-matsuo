import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator, FuncFormatter

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,    
    "font.size": 12,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "savefig.dpi": 600,
    "image.cmap": "viridis",
})

surface_alpha_minus = 0.90
surface_alpha_plus = 0.00
arrow_interval = 5
view_elev = 22
view_azim = -55

"""
# ====================================
# settings for local minima landscape
# ====================================

# length parameters
l1 = 1.0
l2 = 1.0
l3 = 2.0
l4 = 2.0
d = 4.0

# graphical parameters
alpha_range = (-np.pi/2, 3*np.pi/2)
beta_range = (-np.pi/2, 3*np.pi/2)
gamma_range = (-np.pi/2, 3*np.pi/2)


# for states plot
flat_states = np.array([[np.pi, np.pi, 0.0],[0.0, np.pi, np.pi]])
local_minima_states = np.array([[0.0, 2*np.pi/3, 0.0]])

# for trajectory plot
base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "trajectory", "local_minima", "results")
angles_trajectory = np.loadtxt(os.path.join(root, "angles_trajectory.dat"), dtype=float, ndmin=2)

"""


# ====================================
# settings for collision landscape
# ====================================

# length parameters
l1 = 2.0
l2 = 2.0
l3 = 2.0
l4 = 3.0
d = 3.0

# graphical parameters
alpha_range = (-np.pi/2, 3*np.pi/2)
beta_range = (-np.pi/2, 3*np.pi/2)
gamma_range = (-np.pi/2, 3*np.pi/2)

# for states plot
flat_states = np.array([[0.0, 0.0, 0.0]])
#collision_states = np.array([[1.0471975511965976, 2.4278682746450277, -2.3731990578281437]])
collision_states = np.array([[1.300488048076, 2.475292170253, -2.135198953119]])

# for trajectory plot
base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "trajectory", "collision", "results")
angles_trajectory = np.loadtxt(os.path.join(root, "angles_trajectory.dat"), dtype=float, ndmin=2)


# ==========================
# functions
# ==========================

# π/2刻みの目盛りをπ表記にする
def pi_tick_label(value, position):
    n = int(np.rint(value / (np.pi / 2)))
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    coefficient = n // 2 if n % 2 == 0 else n
    numerator = r"\pi" if coefficient == 1 else rf"{coefficient}\pi"
    return rf"${sign}{numerator}$" if n % 2 == 0 else rf"${sign}{numerator}/2$"

# 2D cross product
def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

# collision detection between segments PQ and RS
def collision_detection(P, Q, R, S):
    s1 = cross2d(Q - P, R - P)
    s2 = cross2d(Q - P, S - P)
    s3 = cross2d(S - R, P - R)
    s4 = cross2d(S - R, Q - R)
    val1 = s1 * s2
    val2 = s3 * s4
    return np.maximum(val1, val2)

# collision check
def collision_check(alpha, beta, gamma):
    O_L = np.zeros(alpha.shape + (2,))
    O_R = np.zeros(alpha.shape + (2,))
    O_R[..., 0] = d
    A = np.stack([l1 * np.cos(alpha),l1 * np.sin(alpha)], axis=-1)
    B = np.stack([d + l4 * np.cos(beta),l4 * np.sin(beta)], axis=-1)
    C = np.stack([l1 * np.cos(alpha) + l2 * np.cos(gamma),l1 * np.sin(alpha) + l2 * np.sin(gamma)], axis=-1)
    collision1 = collision_detection(O_L, A, B, O_R)
    collision2 = collision_detection(A, C, B, O_R)
    collision3 = collision_detection(O_L, A, C, B)
    val = np.minimum.reduce([collision1,collision2,collision3])
    return val < 0

# 不連続部分をNaNにする関数
def wrap_and_cut(angle, threshold=np.pi):
    cut = np.zeros_like(angle, dtype=bool)
    jump_alpha = np.abs(np.diff(angle, axis=1)) > threshold
    jump_gamma = np.abs(np.diff(angle, axis=0)) > threshold
    cut[:, :-1] |= jump_alpha
    cut[:, 1:]  |= jump_alpha
    cut[:-1, :] |= jump_gamma
    cut[1:, :]  |= jump_gamma
    result = angle.copy()
    result[cut] = np.nan
    return result

# potential energy function
def potential_energy(alpha, beta, gamma, reference_height=0.0):
    U = np.abs(l1 * np.sin(alpha) - reference_height) + np.abs(l1 * np.sin(alpha) + l2 * np.sin(gamma) - reference_height) + np.abs(l4 * np.sin(beta) - reference_height)
    return U

# ==========================
# main
# ==========================

# 描画領域
a = np.linspace(alpha_range[0], alpha_range[1], 121)
g = np.linspace(gamma_range[0], gamma_range[1], 121)
alpha, gamma = np.meshgrid(a, g)

Cx = l1 * np.cos(alpha) + l2 * np.cos(gamma)
Cy = l1 * np.sin(alpha) + l2 * np.sin(gamma)
rx = Cx - d
ry = Cy
rho = np.sqrt(rx**2 + ry**2)
theta = np.arctan2(ry, rx)

c = (rho**2 + l4**2 - l3**2) / (2*l4*rho+1e-12)
phi = np.arccos(np.clip(c, -1, 1))

# feasible spaceを求める
feasible = np.abs(c) <= 1.0

# alpha・gammaからfeasibleな領域のbetaを計算・正規化
beta_plus = np.full_like(alpha, np.nan)
beta_minus = np.full_like(alpha, np.nan)
beta_plus[feasible] = theta[feasible] + phi[feasible]
beta_minus[feasible] = theta[feasible] - phi[feasible]
beta_plus = (beta_plus - beta_range[0]) % (2 * np.pi) + beta_range[0]
beta_minus = (beta_minus - beta_range[0]) % (2 * np.pi) + beta_range[0]

# feasibleな領域に対して衝突判定
collision_plus = np.full_like(alpha, False, dtype=bool)
collision_minus = np.full_like(alpha, False, dtype=bool)
collision_plus[feasible] = collision_check(alpha[feasible], beta_plus[feasible], gamma[feasible])
collision_minus[feasible] = collision_check(alpha[feasible], beta_minus[feasible], gamma[feasible])

# potential energyを計算
U_plus = potential_energy(alpha, beta_plus, gamma)
U_minus = potential_energy(alpha, beta_minus, gamma)
# plus/minusで共通のcolor scale
U_min = np.nanmin([np.nanmin(U_plus),np.nanmin(U_minus)])
U_max = np.nanmax([np.nanmax(U_plus),np.nanmax(U_minus)])
norm = Normalize(vmin=U_min, vmax=U_max)
cmap = colormaps[plt.rcParams["image.cmap"]]

# 衝突とpotentialを考慮した色のマップを作成
colors_plus = cmap(norm(U_plus))
colors_minus = cmap(norm(U_minus))
dark_factor = 0.3
colors_plus[collision_plus, :3] *= dark_factor
colors_minus[collision_minus, :3] *= dark_factor

# 不連続部分をNaNにする
beta_plus_plot = wrap_and_cut(beta_plus)
beta_minus_plot = wrap_and_cut(beta_minus)

# surface plot
fig = plt.figure()
# 経路・矢印・状態マーカーを曲面より手前に描く
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

# for surface plot
ax.plot_surface(alpha, beta_plus_plot, gamma, facecolors=colors_plus, rstride=1, cstride=1, linewidth=0,shade=False,alpha=surface_alpha_plus, zorder=1)
ax.plot_surface(alpha, beta_minus_plot, gamma, facecolors=colors_minus, rstride=1, cstride=1, linewidth=0,shade=False,alpha=surface_alpha_minus, zorder=1)

# colorbar
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

# states plot
angle_min = np.array([alpha_range[0], beta_range[0], gamma_range[0]])
flat_states = (flat_states - angle_min) % (2 * np.pi) + angle_min
ax.scatter(flat_states[:, 0],flat_states[:, 1],flat_states[:, 2],s=70,marker="s",label="Flat states",color="black",alpha=0.8, zorder=3)

#local_minima_states = (local_minima_states - angle_min) % (2 * np.pi) + angle_min
#ax.scatter(local_minima_states[:, 0],local_minima_states[:, 1],local_minima_states[:, 2],s=70,marker="^",label="Local minima",color="black",alpha=0.8, zorder=3)

collision_states = (collision_states - angle_min) % (2 * np.pi) + angle_min
ax.scatter(collision_states[:, 0],collision_states[:, 1],collision_states[:, 2],s=70,marker="x",label="Collision states",color="red",alpha=0.8, zorder=3)

# trajectory plot
angles_trajectory = (angles_trajectory - angle_min) % (2 * np.pi) + angle_min
trajectory_plot = angles_trajectory[::-1].copy()
jump = np.any(np.abs(np.diff(trajectory_plot, axis=0)) > np.pi, axis=1)
trajectory_plot = np.insert(trajectory_plot, np.flatnonzero(jump) + 1, np.nan, axis=0)
ax.plot(trajectory_plot[:, 0], trajectory_plot[:, 1], trajectory_plot[:, 2], color="black", linewidth=2.0, zorder=2)
initial_state = angles_trajectory[-1,:]
ax.scatter(initial_state[0],initial_state[1],initial_state[2],s=70,marker="o",label="Initial state",color="black",alpha=0.8, zorder=3)
for i in range(0, len(angles_trajectory) - 1, arrow_interval):
    state = angles_trajectory[(-1-i),:]
    next_state = angles_trajectory[(-2-i),:]
    delta_angle = next_state - state
    if np.any(np.abs(delta_angle) > np.pi):
        continue
    ax.quiver(state[0], state[1], state[2], delta_angle[0], delta_angle[1], delta_angle[2], color="black", arrow_length_ratio=0.2, normalize=False, linewidth=1.0, zorder=2)

# 軸など
ax.legend()
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_zlabel(r"$\gamma$")

# 表示範囲
ax.set_xlim(alpha_range)
ax.set_ylim(beta_range)
ax.set_zlim(gamma_range)
ax.view_init(elev=view_elev, azim=view_azim)

# π表示
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_major_locator(MultipleLocator(np.pi / 2))
    axis.set_major_formatter(FuncFormatter(pi_tick_label))


plt.show()
