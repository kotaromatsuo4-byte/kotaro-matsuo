from descriptor import (
    same_state,
    same_terminal_shape,
    state_coordinate_descriptor,
    terminal_continuous_descriptor,
    terminal_descriptor,
)

#  既存archiveと新規終端状態の比較。新しいcellが追加された場合はTrueを返す。
def find_elite_cell(elite_archive,shape_descriptor,layer_key,rms_tolerance,max_tolerance):
    # 既存archiveのすべてと比較して新規cellかどうか判定する
    for cell_index, cell in enumerate(elite_archive):

        # 重なり順が異なれば別cell
        if layer_key != cell["layer_key"]:
            continue

        # 終端形状が同じなら同一cell
        if same_terminal_shape(
            shape_descriptor,
            cell["shape_descriptor"],
            rms_tolerance,
            max_tolerance,
        ):
            return cell_index

    # 同じcellが存在しない
    return None

# elite_archiveを更新する。
def update_elite_archive(elite_archive,history,terminal_state_id,member,length_scale,layer_threshold,overlap_tolerance,separation_tolerance,collinearity_tolerance,parallel_tolerance,rms_tolerance,max_tolerance,reference_point=0.0):

    shape_descriptor, layer_key = terminal_descriptor(history,terminal_state_id,member,length_scale,layer_threshold,overlap_tolerance,separation_tolerance,collinearity_tolerance,parallel_tolerance)
    continuous_descriptor = terminal_continuous_descriptor(
        history,
        terminal_state_id,
        length_scale,
        layer_threshold,
        reference_point,
    )

    # 重なり順を判定できなかった終端は有効archiveに入れない
    if any(upper_member == -1
        for _, _, upper_member in layer_key
    ):
        return False

    cell_index = find_elite_cell(elite_archive,shape_descriptor,layer_key,rms_tolerance,max_tolerance)

    # 同じcellがなければ新規追加
    if cell_index is None:
        elite_archive.append(
            {
                "shape_descriptor": shape_descriptor,
                "layer_key": layer_key,
                "continuous_descriptor": continuous_descriptor,
                "terminal_state_id": terminal_state_id,
            }
        )
        return True

    # 同じcellなら、経路が短い場合だけeliteを更新
    old_terminal_state_id = elite_archive[cell_index]["terminal_state_id"]

    if (
        history[terminal_state_id].depth
        < history[old_terminal_state_id].depth
    ):
        elite_archive[cell_index]["terminal_state_id"] = terminal_state_id
        elite_archive[cell_index]["continuous_descriptor"] = continuous_descriptor

    return False

# 新しい途中状態と同じ距離cellに属するfrontier cellのindexを返す
def find_frontier_cell(frontier_archive,state_descriptor,rms_tolerance,max_tolerance):

    for cell_index, cell in enumerate(frontier_archive):
        if same_state(
            state_descriptor,
            cell["state_descriptor"],
            rms_tolerance,
            max_tolerance,
        ):
            return cell_index

    return None

# frontier_archiveを更新する。既存prototypeは動かさず、新規cellだけを追加する。
def update_frontier_archive(frontier_archive,history,state_ids,reference_point,length_scale,rms_tolerance,max_tolerance):
    
    for state_id in state_ids:
        state = history[state_id]

        # root nodeや終端nodeはfrontierに入れない
        if state_id == 0 or state.terminal_reason is not None:
            continue

        descriptor = state_coordinate_descriptor(state.node,reference_point,length_scale)

        cell_index = find_frontier_cell(
            frontier_archive,
            descriptor,
            rms_tolerance,
            max_tolerance,
        )

        # 既存prototypeの距離閾値内に入らない場合だけ、新しいcellを追加する
        if cell_index is None:
            frontier_archive.append(
                {
                    "state_descriptor": descriptor,
                    "state_id": state_id,
                }
            )
