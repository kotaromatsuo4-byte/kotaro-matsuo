import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "savefig.dpi": 600,
    "image.cmap": "viridis",
})

x = np.linspace(-np.pi, np.pi, 500)
y = np.linspace(-np.pi, np.pi, 500)

alpha, beta = np.meshgrid(x, y)

l2_l1=1
l3_l1=1
d_l1=1

F = (d_l1 + l3_l1 * np.cos(beta) - np.cos(alpha))**2 + (l3_l1 * np.sin(beta) - np.sin(alpha))**2 - l2_l1**2

def potential_energy(alpha, beta, reference_height=0.0):
    U = np.abs(np.sin(alpha) - reference_height) + np.abs(l3_l1 * np.sin(beta) - reference_height)
    return U

fig, ax = plt.subplots()

# 等高線
cs = ax.contour(alpha, beta, F, levels=[0])

path = cs.collections[0].get_paths()[0]
verts = path.vertices
a = verts[:,0]
b = verts[:,1]
U = potential_energy(a, b, reference_height=0.0)

plt.scatter(a,b,c=U,s=3)
plt.colorbar()

# states plot
initial_state = np.array([np.pi/2,np.pi/2])
ax.scatter(initial_state[0],initial_state[1],s=70,marker="o",label="Initial state",color="black",alpha=0.8)
flat_states = np.array([[0.0, 0.0],[np.pi, 0.0]])
ax.scatter(flat_states[:, 0],flat_states[:, 1],s=70,marker="s",label="Flat states",color="black",alpha=0.8)

"""
"o"  # ○
"s"  # □
"^"  # △
"v"  # ▽
"D"  # ◇
"*"  # ★
"x"  # x
"+"  # +
"""

# 座標軸
ax.legend()
ax.axhline(0, color='k', lw=0.8)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")

# 表示範囲
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# π表示
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

ax.set_aspect('equal')
ax.grid(True)

plt.show()
