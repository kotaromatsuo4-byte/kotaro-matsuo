import numpy as np

# ゼロ厚さの棒を対象に、各節点がbeforeからafterへ直線移動すると仮定する。
# CASE 1：非平行な棒の内部同士の交差を、step端点を含めて禁止。端点接触・共線重なりは許容。
# CASE 2：2Dは重なりを許容し通過を禁止。3Dは探索中に棄却せず、成功経路で重なりを追加確認する。
# dimension=2はxz平面、dimension=3はxyz空間。閾値は全てmainから渡す。


"""非共有組を(n_nonshared, 4)の[ia, ib, ja, jb]、共有組を(n_shared, 3)の[s, a, b]で返す。両端共有の組は除く。"""
def prepare_pairs(member):
    member = np.asarray(member, dtype=int)
    nonshared_pairs, shared_pairs = [], []
    for i, (ia, ib) in enumerate(member):
        for ja, jb in member[i + 1:]:
            shared = {ia, ib} & {ja, jb}
            if len(shared) == 0:
                nonshared_pairs.append((ia, ib, ja, jb))
            elif len(shared) == 1:
                s = next(iter(shared))
                shared_pairs.append((s, ib if ia == s else ia, jb if ja == s else ja))
    return np.asarray(nonshared_pairs, dtype=int).reshape(-1, 4), np.asarray(shared_pairs, dtype=int).reshape(-1, 3)


"""移動範囲の箱が重なる非共有組だけを(n_candidate, 4)で返す。距離・移動量による追加の除外は行わない。"""
def filter_swept_boxes(node_before, node_after, nonshared_pairs, *, dimension, distance_tolerance):
    axes = [0, 2] if dimension == 2 else [0, 1, 2]
    before = np.asarray(node_before)[:, axes]
    after = np.asarray(node_after)[:, axes]
    lower, upper = np.minimum(before, after), np.maximum(before, after)
    ia, ib, ja, jb = nonshared_pairs.T
    lower_p, upper_p = np.minimum(lower[ia], lower[ib]), np.maximum(upper[ia], upper[ib])
    lower_q, upper_q = np.minimum(lower[ja], lower[jb]), np.maximum(upper[ja], upper[jb])
    keep = np.all(upper_p + distance_tolerance >= lower_q, axis=1) & np.all(upper_q + distance_tolerance >= lower_p, axis=1)
    return nonshared_pairs[keep]


"""昇順係数の多項式について[0, 1]内の実根を1次元配列で返す。全係数が数値的にゼロなら根を列挙しない。"""
def polynomial_roots(coefficients, *, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    coefficients = np.asarray(coefficients)
    nonzero = np.flatnonzero(np.abs(coefficients) > coefficient_tolerance)
    if len(nonzero) == 0 or nonzero[-1] == 0:
        return np.empty(0)
    roots = np.roots(coefficients[:nonzero[-1] + 1][::-1])
    real = roots.real[np.abs(roots.imag) <= root_imag_tolerance]
    return np.clip(real[(real >= -time_tolerance) & (real <= 1 + time_tolerance)], 0, 1)


"""候補時刻とstep両端を昇順にまとめた1次元配列を返す。数値的に近い根を統合する。"""
def event_partition(times, *, time_tolerance):
    interior = sorted(t for t in times if time_tolerance < t < 1 - time_tolerance)
    result = [0.]
    for time in interior:
        if time - result[-1] > time_tolerance:
            result.append(float(time))
    return np.asarray([*result, 1.])


"""直線変化するu,vの外積の係数を(3, 3)で返す。行はtの0〜2次、列はxyz成分。"""
def cross_coefficients(u0, u1, v0, v1):
    du, dv = u1 - u0, v1 - v0
    return np.stack((np.cross(u0, v0), np.cross(du, v0) + np.cross(u0, dv), np.cross(du, dv)))


"""4端点を共通の原点・長さで正規化して返す。(before, after, scale)で座標は各(4, 3)。"""
def normalize_pair(before, after):
    origin = before[0]
    before, after = before - origin, after - origin
    scale = max(np.max(np.linalg.norm(before, axis=1)), np.max(np.linalg.norm(after, axis=1)))
    return before / scale, after / scale, scale


"""固定した2本の非平行な棒が内部同士で交差すればTrue。pointsは(4, 3)で[A, B, C, D]。端点接触と共線重なりはFalse。"""
def proper_intersection(points, *, distance_tolerance, angle_tolerance, parameter_tolerance):
    a, b, c, d = points
    u, v, offset = b - a, d - c, c - a
    normal = np.cross(u, v)
    normal_squared = np.dot(normal, normal)
    if normal_squared <= angle_tolerance**2 * np.dot(u, u) * np.dot(v, v):
        return False
    s = np.dot(np.cross(offset, v), normal) / normal_squared
    t = np.dot(np.cross(offset, u), normal) / normal_squared
    inside = parameter_tolerance < s < 1 - parameter_tolerance and parameter_tolerance < t < 1 - parameter_tolerance
    return bool(inside and np.linalg.norm(a + s * u - c - t * v) <= distance_tolerance)


"""4端点の向きが変わる時刻を含む区切りを返す。2Dはxz、3Dの常時共面の場合は全投影を使う。"""
def orientation_partition(before, after, *, dimension, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    times = []
    for a, b, c in ((0, 1, 2), (0, 1, 3), (2, 3, 0), (2, 3, 1)):
        coefficients = cross_coefficients(before[b] - before[a], after[b] - after[a], before[c] - before[a], after[c] - after[a])
        components = (1,) if dimension == 2 else (0, 1, 2)
        for axis in components:
            times.extend(polynomial_roots(coefficients[:, axis], coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance))
    return event_partition(times, time_tolerance=time_tolerance)


"""CASE 1の1組を2Dで判定してboolを返す。before/afterは(4, 3)。step両端と各時間区間の内部を調べる。"""
def nonshared_crossing_2d(before, after, *, distance_tolerance, angle_tolerance, parameter_tolerance, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    before, after = np.array(before, dtype=float), np.array(after, dtype=float)
    before[:, 1], after[:, 1] = 0., 0.
    before, after, scale = normalize_pair(before, after)
    times = orientation_partition(before, after, dimension=2, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
    samples = np.concatenate((times, (times[:-1] + times[1:]) / 2))
    for time in samples:
        points = before + time * (after - before)
        if proper_intersection(points, distance_tolerance=distance_tolerance / scale, angle_tolerance=angle_tolerance, parameter_tolerance=parameter_tolerance):
            return True
    return False


"""CASE 1の1組を3Dで判定してboolを返す。before/afterは(4, 3)。共面の根とstep両端を調べ、常時共面なら時間区間も調べる。"""
def nonshared_crossing_3d(before, after, *, distance_tolerance, angle_tolerance, parameter_tolerance, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    before, after, scale = normalize_pair(np.asarray(before, dtype=float), np.asarray(after, dtype=float))
    u0, u1 = before[1] - before[0], after[1] - after[0]
    v0, v1 = before[3] - before[2], after[3] - after[2]
    w0, w1 = before[2] - before[0], after[2] - after[0]
    normal = cross_coefficients(u0, u1, v0, v1)
    dw = w1 - w0
    coefficients = np.array([np.dot(w0, normal[0]), np.dot(dw, normal[0]) + np.dot(w0, normal[1]), np.dot(dw, normal[1]) + np.dot(w0, normal[2]), np.dot(dw, normal[2])])
    if np.max(np.abs(coefficients)) <= coefficient_tolerance:
        times = orientation_partition(before, after, dimension=3, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
        samples = np.concatenate((times, (times[:-1] + times[1:]) / 2))
    else:
        roots = polynomial_roots(coefficients, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
        samples = np.concatenate(([0., 1.], roots))
    for time in samples:
        points = before + time * (after - before)
        if proper_intersection(points, distance_tolerance=distance_tolerance / scale, angle_tolerance=angle_tolerance, parameter_tolerance=parameter_tolerance):
            return True
    return False


"""共有組の2ベクトルを共通長さで正規化し(u0, u1, v0, v1)で返す。各ベクトルは(3,)。before/afterは[s, a, b]の(3, 3)。"""
def shared_vectors(before, after):
    u0, u1 = before[1] - before[0], after[1] - after[0]
    v0, v1 = before[2] - before[0], after[2] - after[0]
    scale = max(np.linalg.norm(u0), np.linalg.norm(u1), np.linalg.norm(v0), np.linalg.norm(v1))
    return u0 / scale, u1 / scale, v0 / scale, v1 / scale


"""2D共有組の状態を(side, overlap)で返す。sideはxz外積の符号、overlapは同じ向きの重なり。ゼロ長ベクトルは重なりとしない。"""
def shared_state_2d(u, v, *, angle_tolerance):
    area = u[0] * v[2] - u[2] * v[0]
    threshold = angle_tolerance * np.linalg.norm(u) * np.linalg.norm(v)
    side = 1 if area > threshold else -1 if area < -threshold else 0
    return side, bool(side == 0 and np.dot(u, v) > 0)


"""共有組の外積・内積の根から時間を区切る。戻り値は端点と各区間中点を含む昇順の1次元配列。"""
def shared_sample_times(u0, u1, v0, v1, *, dimension, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    normal = cross_coefficients(u0, u1, v0, v1)
    du, dv = u1 - u0, v1 - v0
    dot = np.array([np.dot(u0, v0), np.dot(du, v0) + np.dot(u0, dv), np.dot(du, dv)])
    polynomials = [normal[:, 1], dot] if dimension == 2 else [*normal.T, dot]
    times = []
    for coefficients in polynomials:
        times.extend(polynomial_roots(coefficients, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance))
    partition = event_partition(times, time_tolerance=time_tolerance)
    return np.sort(np.concatenate((partition, (partition[:-1] + partition[1:]) / 2)))


"""CASE 2を2Dで判定し(collision, sides_after)を返す。sides_afterは(n_shared,)の接近側履歴。入力履歴は変更しない。"""
def shared_crossing_2d(node_before, node_after, shared_pairs, sides, *, angle_tolerance, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    updated = sides.copy()
    for index, pair in enumerate(shared_pairs):
        before, after = np.array(node_before[pair], dtype=float), np.array(node_after[pair], dtype=float)
        before[:, 1], after[:, 1] = 0., 0.
        u0, u1, v0, v1 = shared_vectors(before, after)
        side, overlapping = shared_state_2d(u0, v0, angle_tolerance=angle_tolerance)
        previous_side = int(sides[index]) if overlapping else side
        times = shared_sample_times(u0, u1, v0, v1, dimension=2, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
        for time in times[1:]:
            u, v = u0 + time * (u1 - u0), v0 + time * (v1 - v0)
            side, overlap = shared_state_2d(u, v, angle_tolerance=angle_tolerance)
            if side != 0:
                if overlapping and previous_side != 0 and side != previous_side:
                    return True, updated
                previous_side, overlapping = side, False
            elif overlap:
                overlapping = True
            else:
                # 反対向きの共線状態は通過を許し、同方向の重なり履歴を解除する。
                previous_side, overlapping = 0, False
        updated[index] = previous_side
    return False, updated


"""3D共有組の同方向の重なりを追加確認し、検出時刻の1次元配列を返す。区間内の代表時刻も含み、貫通と回り込みの分類は行わない。"""
def shared_overlap_times_3d(before, after, *, angle_tolerance, coefficient_tolerance, root_imag_tolerance, time_tolerance):
    u0, u1, v0, v1 = shared_vectors(np.asarray(before), np.asarray(after))
    times = shared_sample_times(u0, u1, v0, v1, dimension=3, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
    overlaps = []
    for time in times:
        u, v = u0 + time * (u1 - u0), v0 + time * (v1 - v0)
        parallel = np.linalg.norm(np.cross(u, v)) <= angle_tolerance * np.linalg.norm(u) * np.linalg.norm(v)
        if parallel and np.dot(u, v) > 0:
            overlaps.append(time)
    return np.asarray(overlaps)


# ペアとCASE 2の履歴だけを保持する。mainのstep用dataclassには接触履歴を入れない。
class CollisionChecker:
    """モデル入力時にペアを作る。distance_toleranceは座標の長さ単位、それ以外の閾値は無次元。"""
    def __init__(self, member, node_initial, *, dimension, distance_tolerance, angle_tolerance, parameter_tolerance, coefficient_tolerance, root_imag_tolerance, time_tolerance):
        self.dimension = dimension
        self.nonshared_pairs, self.shared_pairs = prepare_pairs(member)
        self.distance_tolerance = distance_tolerance
        self.detail_options = dict(distance_tolerance=distance_tolerance, angle_tolerance=angle_tolerance, parameter_tolerance=parameter_tolerance, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
        self.shared_options = dict(angle_tolerance=angle_tolerance, coefficient_tolerance=coefficient_tolerance, root_imag_tolerance=root_imag_tolerance, time_tolerance=time_tolerance)
        self.reset(node_initial)

    """episode開始時に2Dの接近側履歴(n_shared,)を初期化する。初期状態から重なる組は側未定として0を保持する。"""
    def reset(self, node_initial):
        self.shared_sides = np.zeros(len(self.shared_pairs), dtype=int)
        if self.dimension == 2:
            node = np.asarray(node_initial)
            for index, (s, a, b) in enumerate(self.shared_pairs):
                side, _ = shared_state_2d(node[a] - node[s], node[b] - node[s], angle_tolerance=self.shared_options["angle_tolerance"])
                self.shared_sides[index] = side

    """候補移動を判定し、禁止する交差・通過があればTrueを返す。座標・step幅・履歴を変更しない。"""
    def check(self, node_before, node_after):
        node_before, node_after = np.asarray(node_before), np.asarray(node_after)
        candidates = filter_swept_boxes(node_before, node_after, self.nonshared_pairs, dimension=self.dimension, distance_tolerance=self.distance_tolerance)
        detail = nonshared_crossing_2d if self.dimension == 2 else nonshared_crossing_3d
        for pair in candidates:
            if detail(node_before[pair], node_after[pair], **self.detail_options):
                return True
        if self.dimension == 2:
            collision, _ = shared_crossing_2d(node_before, node_after, self.shared_pairs, self.shared_sides, **self.shared_options)
            return bool(collision)
        return False

    """mainがcheck=Falseを確認して採用した移動だけ、履歴へ反映する。候補評価中や棄却時には呼ばない。"""
    def accept(self, node_before, node_after):
        if self.dimension == 2:
            _, self.shared_sides = shared_crossing_2d(np.asarray(node_before), np.asarray(node_after), self.shared_pairs, self.shared_sides, **self.shared_options)

    """保存経路nodes:(n_state, n_node, 3)の3D共有組を追加確認する。step(0始まりの遷移番号), pair:[s,a,b], timeの辞書リストを返す。履歴は変更しない。"""
    def audit_shared_path(self, nodes):
        nodes = np.asarray(nodes)
        events = []
        for step, (before, after) in enumerate(zip(nodes[:-1], nodes[1:])):
            for pair in self.shared_pairs:
                times = shared_overlap_times_3d(before[pair], after[pair], **self.shared_options)
                for time in times:
                    events.append(dict(step=step, pair=pair.tolist(), time=float(time)))
        return events
