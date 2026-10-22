"""2次元リンケージ用の連続的な部材交差判定。
節点は ``node0`` から ``node1`` まで直線的に移動すると仮定する。
時刻 0 < t < 1 で部材同士が交差または重複すれば衝突（0）、
そうでなければ有効（1）を返す。t=1 で初めて生じる接触・重複は、
平坦終端における部材の重なりを許すため衝突に含めない。既存APIには
flat判定が渡されないので、この終端接触の許可はnode1の状態によらず適用される。
"""

import numpy as np


def _project_to_plane(node0, node1):
    """five-bar が運動するx-z平面の座標を取り出す。"""

    # 平面をstepごとに推定すると、平坦近傍で基底が不安定になる。
    # 現モデルでは運動平面が既知なので、x-z座標を明示的に用いる。
    return node0[:, [0, 2]], node1[:, [0, 2]]


def _cross2d(u, v):
    """2次元ベクトルの外積の符号付きスカラー値を返す。"""

    return u[0] * v[1] - u[1] * v[0]


def _orientation_coefficients(a0, a1, b0, b1, c0, c1):
    """cross(B(t)-A(t), C(t)-A(t)) の2次多項式係数を返す。"""

    u0 = b0 - a0
    du = (b1 - b0) - (a1 - a0)
    v0 = c0 - a0
    dv = (c1 - c0) - (a1 - a0)

    quadratic = _cross2d(du, dv)
    linear = _cross2d(du, v0) + _cross2d(u0, dv)
    constant = _cross2d(u0, v0)
    return np.array([quadratic, linear, constant], dtype=float)


def _roots_in_open_unit_interval(coefficients, time_tolerance):
    """実係数多項式の 0 < t < 1 にある実根を返す。"""

    coefficients = np.asarray(coefficients, dtype=float)
    coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
    coefficient_tolerance = 1.0e-12 * coefficient_scale

    first_nonzero = 0
    while (
        first_nonzero < len(coefficients)
        and abs(coefficients[first_nonzero]) <= coefficient_tolerance
    ):
        first_nonzero += 1

    # 恒等的に0、または定数だけの場合は孤立した根を持たない。
    coefficients = coefficients[first_nonzero:]
    if len(coefficients) <= 1:
        return []

    roots = []
    for root in np.roots(coefficients):
        if abs(root.imag) <= 1.0e-9:
            time = float(root.real)
            if time_tolerance < time < 1.0 - time_tolerance:
                roots.append(time)

    return roots


def _unique_sorted(values, tolerance):
    """近接した時刻を一つにまとめ、昇順に並べる。"""

    result = []
    for value in sorted(values):
        value = min(1.0, max(0.0, float(value)))
        if not result or value - result[-1] > tolerance:
            result.append(value)
    return result


def _point_on_segment(point, start, end, length_tolerance, area_tolerance):
    """点が閉線分上にあるかを判定する。"""

    if abs(_cross2d(end - start, point - start)) > area_tolerance:
        return False

    return (
        min(start[0], end[0]) - length_tolerance
        <= point[0]
        <= max(start[0], end[0]) + length_tolerance
        and min(start[1], end[1]) - length_tolerance
        <= point[1]
        <= max(start[1], end[1]) + length_tolerance
    )


def _segments_intersect(a, b, c, d, length_tolerance, area_tolerance):
    """二つの閉線分が交差、接触または重複するかを判定する。"""

    o1 = _cross2d(b - a, c - a)
    o2 = _cross2d(b - a, d - a)
    o3 = _cross2d(d - c, a - c)
    o4 = _cross2d(d - c, b - c)

    proper_intersection = (
        ((o1 > area_tolerance and o2 < -area_tolerance)
         or (o1 < -area_tolerance and o2 > area_tolerance))
        and ((o3 > area_tolerance and o4 < -area_tolerance)
             or (o3 < -area_tolerance and o4 > area_tolerance))
    )
    if proper_intersection:
        return True

    if abs(o1) <= area_tolerance and _point_on_segment(
        c, a, b, length_tolerance, area_tolerance
    ):
        return True
    if abs(o2) <= area_tolerance and _point_on_segment(
        d, a, b, length_tolerance, area_tolerance
    ):
        return True
    if abs(o3) <= area_tolerance and _point_on_segment(
        a, c, d, length_tolerance, area_tolerance
    ):
        return True
    if abs(o4) <= area_tolerance and _point_on_segment(
        b, c, d, length_tolerance, area_tolerance
    ):
        return True

    return False


def _node_at_time(node0, node1, time):
    """線形補間した時刻 ``time`` の節点位置を返す。"""

    return node0 + time * (node1 - node0)


def _nonshared_members_collide(
    node0,
    node1,
    first_member,
    second_member,
    length_tolerance,
    area_tolerance,
    time_tolerance,
):
    """節点を共有しない二部材が 0 < t < 1 で交差するかを調べる。"""

    ia, ib = first_member
    ja, jb = second_member

    a0, b0 = node0[ia], node0[ib]
    c0, d0 = node0[ja], node0[jb]
    a1, b1 = node1[ia], node1[ib]
    c1, d1 = node1[ja], node1[jb]

    event_times = [0.0, 1.0]

    # 線分交差の向き条件が変わり得る時刻をすべて集める。
    orientation_data = (
        (a0, a1, b0, b1, c0, c1),
        (a0, a1, b0, b1, d0, d1),
        (c0, c1, d0, d1, a0, a1),
        (c0, c1, d0, d1, b0, b1),
    )
    for data in orientation_data:
        coefficients = _orientation_coefficients(*data)
        event_times.extend(
            _roots_in_open_unit_interval(coefficients, time_tolerance)
        )

    # 共線時の線分範囲の重なりが変わり得る時刻も集める。
    endpoint_data = (
        (a0, a1, c0, c1),
        (a0, a1, d0, d1),
        (b0, b1, c0, c1),
        (b0, b1, d0, d1),
    )
    for p0, p1, q0, q1 in endpoint_data:
        for coordinate in range(2):
            constant = p0[coordinate] - q0[coordinate]
            linear = (
                (p1[coordinate] - p0[coordinate])
                - (q1[coordinate] - q0[coordinate])
            )
            event_times.extend(
                _roots_in_open_unit_interval(
                    [linear, constant], time_tolerance
                )
            )

    event_times = _unique_sorted(event_times, time_tolerance)

    # 各イベント時刻と、その間の区間を一つずつ検査する。
    # t=0,1そのものは検査しないため、終端で初めて生じる重なりは許される。
    times_to_check = [
        time
        for time in event_times
        if time_tolerance < time < 1.0 - time_tolerance
    ]
    times_to_check.extend(
        0.5 * (left + right)
        for left, right in zip(event_times[:-1], event_times[1:])
        if right - left > time_tolerance
    )

    for time in times_to_check:
        a = _node_at_time(a0, a1, time)
        b = _node_at_time(b0, b1, time)
        c = _node_at_time(c0, c1, time)
        d = _node_at_time(d0, d1, time)

        if _segments_intersect(
            a, b, c, d, length_tolerance, area_tolerance
        ):
            return True

    return False


def _shared_members_collide(
    node0,
    node1,
    first_member,
    second_member,
    shared_node,
    length_tolerance,
    area_tolerance,
    time_tolerance,
):
    """共有節点以外で隣接二部材が重なる時刻があるかを調べる。"""

    ia, ib = first_member
    ja, jb = second_member
    first_other = ib if ia == shared_node else ia
    second_other = jb if ja == shared_node else ja

    s0, s1 = node0[shared_node], node1[shared_node]
    a0, a1 = node0[first_other], node1[first_other]
    b0, b1 = node0[second_other], node1[second_other]

    coefficients = _orientation_coefficients(
        s0, s1, a0, a1, b0, b1
    )
    event_times = _roots_in_open_unit_interval(coefficients, time_tolerance)

    coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
    identically_collinear = np.max(np.abs(coefficients)) <= 1.0e-12 * coefficient_scale

    # 二部材が区間全体で共線なら、dot(u(t), v(t)) の符号が変わる
    # 時刻で区間を分割し、同じ向きになる時間区間を漏れなく調べる。
    if identically_collinear:
        u0 = a0 - s0
        du = (a1 - a0) - (s1 - s0)
        v0 = b0 - s0
        dv = (b1 - b0) - (s1 - s0)
        dot_coefficients = np.array(
            [
                np.dot(du, dv),
                np.dot(du, v0) + np.dot(u0, dv),
                np.dot(u0, v0),
            ]
        )
        partition_times = [0.0, 1.0]
        partition_times.extend(
            _roots_in_open_unit_interval(dot_coefficients, time_tolerance)
        )
        partition_times = _unique_sorted(partition_times, time_tolerance)
        event_times.extend(
            0.5 * (left + right)
            for left, right in zip(partition_times[:-1], partition_times[1:])
            if right - left > time_tolerance
        )

    for time in _unique_sorted(event_times, time_tolerance):
        s = _node_at_time(s0, s1, time)
        a = _node_at_time(a0, a1, time)
        b = _node_at_time(b0, b1, time)
        u = a - s
        v = b - s

        length_u = np.linalg.norm(u)
        length_v = np.linalg.norm(v)
        if length_u <= length_tolerance or length_v <= length_tolerance:
            continue

        collinear = abs(_cross2d(u, v)) <= area_tolerance
        same_direction = np.dot(u, v) > length_tolerance**2

        # 同じ半直線上に並ぶと、共有節点以外にも正の長さの重なりがある。
        if collinear and same_direction:
            return True

    return False


def edge_edge_crossing_check(
    node0,
    node1,
    member,
    dist_tol,
    normal_tol,
    cos_tol,
):
    """全部材対について平面内の連続交差判定を行う。

    返り値は既存実装と同じで、0 が衝突、1 が有効である。
    ``dist_tol``, ``normal_tol``, ``cos_tol`` は import の差し替えだけで
    利用できるように引数として残している。ここでは厚みのない2次元線分の
    厳密な交差を調べるため、判定値としては使用しない。
    """

    del dist_tol, normal_tol, cos_tol

    node0_2d, node1_2d = _project_to_plane(node0, node1)

    coordinate_span = np.ptp(np.vstack((node0_2d, node1_2d)), axis=0)
    geometry_scale = max(1.0, float(np.linalg.norm(coordinate_span)))
    length_tolerance = 1.0e-10 * geometry_scale
    area_tolerance = 1.0e-10 * geometry_scale**2
    time_tolerance = 1.0e-9

    n_member = len(member)
    for i in range(n_member):
        first_member = member[i]
        first_nodes = set(first_member)

        for j in range(i + 1, n_member):
            second_member = member[j]
            shared_nodes = first_nodes.intersection(second_member)

            if len(shared_nodes) == 0:
                collision = _nonshared_members_collide(
                    node0_2d,
                    node1_2d,
                    first_member,
                    second_member,
                    length_tolerance,
                    area_tolerance,
                    time_tolerance,
                )
            elif len(shared_nodes) == 1:
                collision = _shared_members_collide(
                    node0_2d,
                    node1_2d,
                    first_member,
                    second_member,
                    next(iter(shared_nodes)),
                    length_tolerance,
                    area_tolerance,
                    time_tolerance,
                )
            else:
                # 同じ二節点を結ぶ重複部材は、この探索では同一部材として扱う。
                collision = False

            if collision:
                return 0

    return 1
