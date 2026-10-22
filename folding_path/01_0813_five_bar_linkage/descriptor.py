import numpy as np

# 途中状態を、無次元化したxyz節点vectorで表す
def state_coordinate_descriptor(node, reference_point, length_scale):
    descriptor = node.copy()
    descriptor[:, 2] -= reference_point
    return (descriptor / length_scale).reshape(-1)

# 二つの途中状態について、xyz節点位置のRMS距離と最大節点距離を返す
def state_distance(descriptor1, descriptor2):
    difference = (
        np.asarray(descriptor1).reshape(-1, 3)
        - np.asarray(descriptor2).reshape(-1, 3)
    )
    node_distance = np.linalg.norm(difference, axis=1)
    rms_distance = float(np.sqrt(np.mean(node_distance**2)))
    max_distance = float(np.max(node_distance))
    return rms_distance, max_distance

# RMS距離と最大節点距離から、二つの途中状態を同一cellとみなすか判定する
def same_state(descriptor1, descriptor2, rms_tolerance, max_tolerance):
    rms_distance, max_distance = state_distance(descriptor1, descriptor2)
    return (
        rms_distance <= rms_tolerance
        and max_distance <= max_tolerance
    )

# 終端形状を、無次元化したxy節点vectorで表す
def terminal_coordinate_descriptor(node, length_scale):
    return (node[:, :2] / length_scale).reshape(-1)

# 二形状間のRMS距離と最大節点距離を返す
def shape_distance(descriptor1, descriptor2):
    difference = (np.asarray(descriptor1).reshape(-1, 2) - np.asarray(descriptor2).reshape(-1, 2))
    node_distance = np.linalg.norm(difference, axis=1)
    rms_distance = float(np.sqrt(np.mean(node_distance**2)))
    max_distance = float(np.max(node_distance))
    return rms_distance, max_distance

# RMS距離と最大節点距離から同一形状か判定する
def same_terminal_shape(descriptor1,descriptor2,rms_tolerance,max_tolerance):
    rms_distance, max_distance = shape_distance(descriptor1, descriptor2)
    return (rms_distance <= rms_tolerance and max_distance <= max_tolerance)


def _cross_2d(vector1, vector2):
    return float(vector1[0] * vector2[1] - vector1[1] * vector2[0])

# 終端のxy射影上で有限長重なる部材対を返す
def find_overlapping_member_pairs(terminal_node,member,length_scale,overlap_tolerance,collinearity_tolerance,parallel_tolerance):

    if collinearity_tolerance is None:
        collinearity_tolerance = overlap_tolerance

    projected_node = terminal_node[:, :2] / length_scale
    overlaps = []

    for member_i, (i0, i1) in enumerate(member):
        p0 = projected_node[i0]
        p1 = projected_node[i1]
        vector_i = p1 - p0
        length_i = np.linalg.norm(vector_i)

        for member_j in range(member_i + 1, len(member)):
            j0, j1 = member[member_j]

            q0 = projected_node[j0]
            q1 = projected_node[j1]
            vector_j = q1 - q0
            length_j = np.linalg.norm(vector_j)

            parallel_error = abs(_cross_2d(vector_i, vector_j)) / (
                length_i * length_j
            )
            if parallel_error > parallel_tolerance:
                continue

            distance_q0 = abs(_cross_2d(vector_i, q0 - p0)) / length_i
            distance_q1 = abs(_cross_2d(vector_i, q1 - p0)) / length_i
            if max(distance_q0, distance_q1) > collinearity_tolerance:
                continue

            axis = vector_i / length_i
            interval_i = sorted((np.dot(p0, axis), np.dot(p1, axis)))
            interval_j = sorted((np.dot(q0, axis), np.dot(q1, axis)))
            overlap_start = max(interval_i[0], interval_j[0])
            overlap_end = min(interval_i[1], interval_j[1])
            overlap_length = overlap_end - overlap_start

            if overlap_length <= overlap_tolerance:
                continue

            # 重複区間中央に対応する各部材上の物質点を記録する。
            sample_position = 0.5 * (overlap_start + overlap_end)
            sample_node = p0 + (
                sample_position - np.dot(p0, axis)
            ) * axis
            parameter_i = np.dot(sample_node - p0, vector_i) / length_i**2
            parameter_j = np.dot(sample_node - q0, vector_j) / length_j**2

            overlaps.append(
                {
                    "member_i": int(member_i),
                    "member_j": int(member_j),
                    "parameter_i": float(parameter_i),
                    "parameter_j": float(parameter_j),
                }
            )

    return overlaps

# 終端から遡り、potentialが閾値を初めて超える状態を返す
def find_last_nonflat_state_id(history,terminal_state_id,layer_threshold):
    state_id = terminal_state_id
    oldest_state_id = terminal_state_id

    while state_id is not None:
        state = history[state_id]
        oldest_state_id = state_id
        if state.potential > layer_threshold:
            return state_id
        state_id = state.parent_id

    return oldest_state_id


def terminal_continuous_descriptor(
    history,
    terminal_state_id,
    length_scale,
    layer_threshold,
    reference_point=0.0,
):
    """GPが回帰する連続終端descriptorを返す。

    前半は平坦終端のxy座標 ``2 * num_node`` 成分、後半は終端から遡って
    最初に非平坦閾値を超えた状態のz座標 ``num_node`` 成分である。後半を
    残すことで、完全平坦化で失われる重なり順を後から復元できる。
    """

    terminal_node = history[terminal_state_id].node
    observation_state_id = find_last_nonflat_state_id(
        history,
        terminal_state_id,
        layer_threshold,
    )
    observation_z = (
        history[observation_state_id].node[:, 2] - reference_point
    ) / length_scale
    return np.concatenate(
        (
            terminal_coordinate_descriptor(terminal_node, length_scale),
            observation_z,
        )
    )


def decode_terminal_continuous_descriptor(
    continuous_descriptor,
    member,
    num_node,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """連続GP出力を既存archiveと同じ形状・重なり順へ変換する。"""

    prediction = np.asarray(continuous_descriptor, dtype=float).reshape(-1)
    shape_dimension = 2 * num_node
    if prediction.size != 3 * num_node or not np.all(np.isfinite(prediction)):
        return prediction[:shape_dimension], ((-1, -1, -1),)

    shape_descriptor = prediction[:shape_dimension]
    observation_z = prediction[shape_dimension:]

    normalized_terminal_node = np.zeros((num_node, 3), dtype=float)
    normalized_terminal_node[:, :2] = shape_descriptor.reshape(num_node, 2)
    overlaps = find_overlapping_member_pairs(
        normalized_terminal_node,
        member,
        1.0,
        overlap_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )

    layer_key = []
    for overlap in overlaps:
        member_i = overlap["member_i"]
        member_j = overlap["member_j"]
        i0, i1 = member[member_i]
        j0, j1 = member[member_j]
        parameter_i = overlap["parameter_i"]
        parameter_j = overlap["parameter_j"]
        z_i = (
            (1.0 - parameter_i) * observation_z[i0]
            + parameter_i * observation_z[i1]
        )
        z_j = (
            (1.0 - parameter_j) * observation_z[j0]
            + parameter_j * observation_z[j1]
        )
        separation = z_i - z_j
        if separation > separation_tolerance:
            upper_member = int(member_i)
        elif separation < -separation_tolerance:
            upper_member = int(member_j)
        else:
            upper_member = -1
        layer_key.append((int(member_i), int(member_j), upper_member))

    return shape_descriptor, tuple(layer_key)

# 最後の非平坦状態における重複部材の上下関係を返す
def layer_order_descriptor(history,observation_state_id,member,overlaps,length_scale,separation_tolerance):

    node = history[observation_state_id].node
    layer_key = []

    for overlap in overlaps:
        member_i = overlap["member_i"]
        member_j = overlap["member_j"]
        i0, i1 = member[member_i]
        j0, j1 = member[member_j]
        parameter_i = overlap["parameter_i"]
        parameter_j = overlap["parameter_j"]

        point_i = (
            (1.0 - parameter_i) * node[i0]
            + parameter_i * node[i1]
        )
        point_j = (
            (1.0 - parameter_j) * node[j0]
            + parameter_j * node[j1]
        )
        separation = (point_i[2] - point_j[2]) / length_scale

        if separation > separation_tolerance:
            upper_member = member_i
        elif separation < -separation_tolerance:
            upper_member = member_j
        else:
            upper_member = -1

        layer_key.append((member_i, member_j, upper_member))

    return tuple(layer_key)

# 平坦終端の形状座標と重なり順を返す
def terminal_descriptor(
    history,
    terminal_state_id,
    member,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):


    terminal_node = history[terminal_state_id].node
    shape_descriptor = terminal_coordinate_descriptor(
        terminal_node,
        length_scale,
    )

    overlaps = find_overlapping_member_pairs(
        terminal_node,
        member,
        length_scale,
        overlap_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )
    observation_state_id = find_last_nonflat_state_id(
        history,
        terminal_state_id,
        layer_threshold,
    )
    layer_key = layer_order_descriptor(
        history,
        observation_state_id,
        member,
        overlaps,
        length_scale,
        separation_tolerance,
    )

    return shape_descriptor, layer_key # 例: ((0, 1, 0),) は、部材1に対して部材0が上
