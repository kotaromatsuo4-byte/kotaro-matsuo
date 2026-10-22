import numpy as np

# segment-segment distance
def segment_distance(p1, p2, q1, q2):

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b

    eps = 1e-12

    sN = 0.0
    sD = D

    tN = 0.0
    tD = D

    if D < eps:

        sN = 0.0
        sD = 1.0
        tN = e
        tD = c

    else:

        sN = (b * e - c * d)
        tN = (a * e - b * d)

        if sN < 0.0:

            sN = 0.0
            tN = e
            tD = c

        elif sN > sD:

            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:

        tN = 0.0

        if -d < 0.0:
            sN = 0.0

        elif -d > a:
            sN = sD

        else:
            sN = -d
            sD = a

    elif tN > tD:

        tN = tD

        if (-d + b) < 0.0:
            sN = 0

        elif (-d + b) > a:
            sN = sD

        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD

    dP = w + sc * u - tc * v

    return np.linalg.norm(dP)

# edge-edge crossing check
def edge_edge_crossing_check(node0,node1,member,dist_tol,normal_tol,cos_tol):

    n_member = len(member)

    for i in range(n_member):

        ia, ib = member[i]

        set_i = set([ia, ib])

        for j in range(i + 1, n_member):

            ja, jb = member[j]

            set_j = set([ja, jb])

            shared = list(set_i & set_j)

            # CASE 1: non shared node
            if len(shared) == 0:

                p0a = node0[ia]
                p0b = node0[ib]
                q0a = node0[ja]
                q0b = node0[jb]

                p1a = node1[ia]
                p1b = node1[ib]
                q1a = node1[ja]
                q1b = node1[jb]

                dist0 = segment_distance(p0a, p0b,q0a, q0b)
                dist1 = segment_distance(p1a, p1b,q1a, q1b)

                if dist0 > dist_tol and dist1 > dist_tol:
                    continue

                u0 = p0b - p0a
                v0 = q0b - q0a
                n0 = np.cross(u0, v0)
                norm0 = np.linalg.norm(n0)
                if norm0 < normal_tol:
                    continue
                d0 = np.dot(q0a - p0a, n0)

                u1 = p1b - p1a
                v1 = q1b - q1a
                n1 = np.cross(u1, v1)
                norm1 = np.linalg.norm(n1)
                if norm1 < normal_tol:
                    continue
                d1 = np.dot(q1a - p1a, n1)

                d0 /=np.linalg.norm(n0)+1e-8
                d1 /=np.linalg.norm(n1)+1e-8
                if d0 * d1 < -1e-4: # non-shared crossing is strictly defined.
                    return 0

            # CASE 2: one shared node
            elif len(shared) == 1:

                s = shared[0]
                a = ib if ia == s else ia
                b = jb if ja == s else ja

                u0 = node0[a] - node0[s]
                v0 = node0[b] - node0[s]
                cos0= np.dot(u0, v0) / (np.linalg.norm(u0) * np.linalg.norm(v0) + 1e-8)

                u1 = node1[a] - node1[s]
                v1 = node1[b] - node1[s]
                cos1 = np.dot(u1, v1) / (np.linalg.norm(u1) * np.linalg.norm(v1) + 1e-8)

                near0 = cos0 > cos_tol
                near1 = cos1 > cos_tol

                if not (near0 or near1): # if both are not near, then it's a crossing
                    continue

                n0 = np.cross(u0, v0)
                n1 = np.cross(u1, v1)

                n0 /= np.linalg.norm(n0) + 1e-8
                n1 /= np.linalg.norm(n1) + 1e-8
                if np.dot(n0, n1) < -1e-2: # shared-node crossing is not strictly defined. Because things don't work out if you evaluate contact too strictly.
                    return 0

            # CASE 3: two shared nodes
            else:
                continue

    return 1