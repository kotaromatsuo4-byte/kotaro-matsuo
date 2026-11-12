import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})

l1 = 1.0
l2 = 1.2
l3 = 1.5
l4 = 1.0
d  = 1.5

a = np.linspace(-np.pi, np.pi, 300)
b = np.linspace(-np.pi, np.pi, 300)

alpha, beta = np.meshgrid(a, b)

Ax = l1 * np.cos(alpha)
Ay = l1 * np.sin(alpha)

Bx = d + l4 * np.cos(beta)
By = l4 * np.sin(beta)

rx = Bx - Ax
ry = By - Ay

rho = np.sqrt(rx**2 + ry**2)
theta = np.arctan2(ry, rx)

c = (rho**2 + l2**2 - l3**2) / (2*l2*rho)

# feasible spaceを求める
feasible = np.abs(c) <= 1.0

gamma_plus = np.full_like(alpha, np.nan)
gamma_minus = np.full_like(alpha, np.nan)

phi = np.arccos(np.clip(c, -1, 1))

# feasibleな領域に対してgammaを計算
gamma_plus[feasible] = theta[feasible] + phi[feasible]
gamma_minus[feasible] = theta[feasible] - phi[feasible]


# surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(alpha, beta, gamma_plus, alpha=1.0)
ax.plot_surface(alpha, beta, gamma_minus, alpha=1.0)

# states plot
initial_state = np.array([np.pi/2,np.pi/2,np.pi/2])
ax.scatter(initial_state[0],initial_state[1],initial_state[2],s=80,marker="o",label="Initial state")
flat_states = np.array([[0.0, 0.0, 0.0],[np.pi, 0.0, np.pi]])
ax.scatter(flat_states[:, 0],flat_states[:, 1],flat_states[:, 2],s=80,marker="^",label="Flat states")

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