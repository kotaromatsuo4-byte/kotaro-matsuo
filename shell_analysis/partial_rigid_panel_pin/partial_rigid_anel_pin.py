import numpy as np
from pathlib import Path
from SVD import modes

# モデル
node=np.loadtxt("00_node_initial.dat",dtype=float)
panel=np.loadtxt("01_face.dat",dtype=int)
member=np.loadtxt("02_member.dat",dtype=int)
fix_member_id=np.loadtxt("03_fix_crease_id.dat",delimiter=',',dtype=int)
fix=np.loadtxt("04_fix.dat",dtype=int)

"""

node = np.array([
    [0.0,0.0,0.0],
    [0.5,0.0,0.0],
    [0.5,0.5,0.7071],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0]
])
member = np.array([
    [0,1],[0,2],[0,4],[1,3],[1,4],[2,4],[3,4]
])

panel = np.array([
    [1,4,3],
    [0,4,1],
    [0,2,4]
])

fix_member_id = []

fix = np.array([
    [0,0],[0,1],[0,2],
    [4,0],[4,1],[4,2],
    [2,0],[2,1],[2,2]
])
"""
L0 = np.array([np.linalg.norm(node[i]-node[j]) for i,j in member])
phi0 = np.array([0 for i in fix_member_id])
#angle
def angle(a,b):
    c = np.dot(a,b) / (np.linalg.norm(a)* np.linalg.norm(b))
    c = np.clip(c,-1.0,1.0)
    return np.arccos(c)

#cot(x)
def cot(x):
    return 1/np.tan(x)

#find k,l for fold angle
def find_kl(panel, i, j):
    faces = []
    for tri in panel:
        if i in tri and j in tri:
            faces.append(tri)
    tri1, tri2 = faces
    k = [v for v in tri1 if v not in (i,j)][0]
    l = [v for v in tri2 if v not in (i,j)][0]

    return k, l

#extract free index
def index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])

#incompatibility and Jacobian_matrix
def G_dGdX(node,panel,fix_member_id,member,L0,phi0):
    
    fix_member_id = np.array(fix_member_id, dtype=int).ravel()
    phi0 = np.array(phi0, dtype=float).ravel()

    m = member.shape[0]
    f = len(fix_member_id)
    n = node.shape[0]

    G_bar = np.zeros((m, 1))
    G_panel = np.zeros((f, 1))

    dGdX = np.zeros((m+f, 3*n))

    #bar_length__derivative_fold_angle
    for x in range(m):
        i = member[x,0]
        j = member[x,1]
        #bar length
        L_ij_square = 0.5 * (np.linalg.norm(node[i]-node[j])**2-L0[x]**2)
        G_bar[x] += L_ij_square

        #derivative of bar length
        dL_d_i = (node[i]-node[j])
        dL_d_j = -(node[i]-node[j])

        dGdX[x,3*i:3*i+3]+=dL_d_i
        dGdX[x,3*j:3*j+3]+=dL_d_j

    #fold_angle__derivative_fold_angle
    for x in range(f):
        y =fix_member_id[x] 
        fix_member=member[y]
        #index
        i=fix_member[0]
        j=fix_member[1]
        #search for k,l
        k,l = find_kl(panel, i, j)

        #normal vector
        n_k=np.cross(node[k]-node[j],node[i]-node[j])/np.linalg.norm(np.cross(node[k]-node[j],node[i]-node[j]))
        n_l=np.cross(node[i]-node[j],node[l]-node[j])/np.linalg.norm(np.cross(node[i]-node[j],node[l]-node[j]))

        #fold angle
        phi_ij=np.arccos(np.clip(np.dot(n_k,n_l), -1.0, 1.0))
        G_panel[x] = phi_ij - phi0[x]

        #sector angle
        theta_j_ki = angle(node[k]-node[j],node[i]-node[j])
        theta_j_il = angle(node[l]-node[j],node[i]-node[j])
        theta_i_jk = angle(node[j]-node[i],node[k]-node[i])
        theta_i_lj = angle(node[l]-node[i],node[j]-node[i])

        #length of the perpendicular
        h_k=np.sin(theta_j_ki)*np.linalg.norm(node[k]-node[j])
        h_l=np.sin(theta_j_il)*np.linalg.norm(node[l]-node[j])

        #derivative of fold angle
        d_phi_d_k=n_k/h_k
        d_phi_d_l=n_l/h_l
        d_phi_d_i=-cot(theta_j_ki)/(cot(theta_j_ki)+cot(theta_i_jk))*d_phi_d_k-cot(theta_j_il)/(cot(theta_j_il)+cot(theta_i_lj))*d_phi_d_l
        d_phi_d_j=-cot(theta_i_jk)/(cot(theta_j_ki)+cot(theta_i_jk))*d_phi_d_k-cot(theta_i_lj)/(cot(theta_j_il)+cot(theta_i_lj))*d_phi_d_l

        dGdX[m+x,3*i:3*i+3]+=d_phi_d_i
        dGdX[m+x,3*j:3*j+3]+=d_phi_d_j
        dGdX[m+x,3*k:3*k+3]+=d_phi_d_k
        dGdX[m+x,3*l:3*l+3]+=d_phi_d_l

    G = np.vstack([G_bar, G_panel])

    #assemble
    return G,dGdX

G,dGdX = G_dGdX(node,panel,fix_member_id,member,L0,phi0)
soft_sigma,soft_modes = modes(dGdX, abs_sigma_tol=1e-6, gap_ratio=10.0, k_start=12, k_max=100)
print(soft_sigma)
for old_mode in Path(".").glob("soft_mode_*.dat"):
    old_mode.unlink()
for i in range(len(soft_sigma)):
    np.savetxt("soft_mode_"+str(i)+".dat", soft_modes[:,i].reshape(-1,3), delimiter=',', fmt='%.6f')
print(G)
np.savetxt("dG_dX.dat", dGdX, delimiter=',', fmt='%.6f')
