import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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


# 不連続部分をNaNにする関数
def wrap_and_cut(angle, threshold=np.pi):
    # 不連続部分を記録
    cut = np.zeros_like(angle, dtype=bool)

    # alpha方向の隣接点差
    jump_alpha = np.abs(np.diff(angle, axis=1)) > threshold

    # beta方向の隣接点差
    jump_beta = np.abs(np.diff(angle, axis=0)) > threshold

    # jumpを挟む両側の点をNaN対象にする
    cut[:, :-1] |= jump_alpha
    cut[:, 1:]  |= jump_alpha

    cut[:-1, :] |= jump_beta
    cut[1:, :]  |= jump_beta

    result = angle.copy()
    result[cut] = np.nan

    return result

def potential_energy(alpha, beta, gamma, reference_height=0.0):
    U = np.abs(l1 * np.sin(alpha) - reference_height) + np.abs(l1 * np.sin(alpha) + l2 * np.sin(gamma) - reference_height) + np.abs(l4 * np.sin(beta) - reference_height)
    return U

l1 = 1.0
l2 = 1.0
l3 = 1.0
l4 = 1.0
d  = 2.0

# for surface plot
a = np.linspace(-np.pi, np.pi, 80)
b = np.linspace(-np.pi, np.pi, 80)

alpha, beta = np.meshgrid(a, b)

Ax = l1 * np.cos(alpha)
Ay = l1 * np.sin(alpha)

Bx = d + l4 * np.cos(beta)
By = l4 * np.sin(beta)

rx = Bx - Ax
ry = By - Ay

rho = np.sqrt(rx**2 + ry**2)
theta = np.arctan2(ry, rx)

c = (rho**2 + l2**2 - l3**2) / (2*l2*rho+1e-12)

# feasible spaceを求める
feasible = np.abs(c) <= 1.0

gamma_plus = np.full_like(alpha, np.nan)
gamma_minus = np.full_like(alpha, np.nan)

phi = np.arccos(np.clip(c, -1, 1))

# feasibleな領域に対してgammaを計算・正規化
gamma_plus[feasible] = theta[feasible] + phi[feasible]
gamma_minus[feasible] = theta[feasible] - phi[feasible]
gamma_plus = (gamma_plus + np.pi) % (2 * np.pi) - np.pi
gamma_minus = (gamma_minus + np.pi) % (2 * np.pi) - np.pi

# potential energyを計算
U_plus = potential_energy(alpha, beta, gamma_plus)
U_minus = potential_energy(alpha, beta, gamma_minus)

# plus/minusで共通のcolor scale
U_min = np.nanmin([np.nanmin(U_plus),np.nanmin(U_minus)])
U_max = np.nanmax([np.nanmax(U_plus),np.nanmax(U_minus)])
norm = Normalize(vmin=U_min, vmax=U_max)
cmap = colormaps[plt.rcParams["image.cmap"]]

# 不連続部分をNaNにする
gamma_plus_plot = wrap_and_cut(gamma_plus)
gamma_minus_plot = wrap_and_cut(gamma_minus)

# surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# for surface plot
ax.plot_surface(alpha, beta, gamma_plus_plot, facecolors=cmap(norm(U_plus)), rstride=1, cstride=1, linewidth=0,shade=False,alpha=1.0)
ax.plot_surface(alpha, beta, gamma_minus_plot, facecolors=cmap(norm(U_minus)), rstride=1, cstride=1, linewidth=0,shade=False,alpha=0.5)

# colorbar
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

# states plot
initial_state = np.array([np.pi/2,np.pi/2,np.pi/2])
ax.scatter(initial_state[0],initial_state[1],initial_state[2],s=70,marker="o",label="Initial state",color="black",alpha=0.8)
flat_states = np.array([[0.0, 0.0, 0.0],[np.pi, 0.0, np.pi]])
ax.scatter(flat_states[:, 0],flat_states[:, 1],flat_states[:, 2],s=70,marker="s",label="Flat states",color="black",alpha=0.8)

# 軸など
ax.legend()
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_zlabel(r"$\gamma$")

# 表示範囲
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_zlim(-np.pi, np.pi)

# π表示
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_zticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax.set_zticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


plt.show()