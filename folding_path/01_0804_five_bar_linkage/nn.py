"""終端予測NNを用いた摂動候補の生成・学習・選択。

NNへはSVD modeやその係数を直接渡さず、両者から計算した物理的な
節点変位を渡す。これによりmodeの符号反転や基底回転の影響を避ける。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as torch_nn

from descriptor import (
    find_last_nonflat_state_id,
    find_overlapping_member_pairs,
    state_coordinate_descriptor,
    terminal_descriptor,
)
from mechanics import calc_dedx, calc_mode


# 停止理由は分類し、flatのときだけ連続終端descriptorを回帰する。
STOP_TO_INDEX = {"flat": 0, "collision": 1, "timeout": 2}


def candidate_input(node, displacement, reference_point, length_scale):
    """摂動前節点座標と物理節点変位を無次元化して一つのvectorにする。"""
    state_descriptor = state_coordinate_descriptor(
        node,
        reference_point,
        length_scale,
    )
    displacement_descriptor = (
        np.asarray(displacement, dtype=float) / length_scale
    ).reshape(-1)
    return np.concatenate((state_descriptor, displacement_descriptor))


def generate_candidates(
    history,
    parent_ids,
    num_perturbations_per_parent,
    member,
    num_node,
    num_member,
    index_free,
    perturbation_scale,
    reference_point,
    length_scale,
    rng,
    mode_cache=None,
):
    """各親StateNodeについて複数の摂動方向を生成する。

    correctorは実行しない。候補生成時に必要なのはmode計算と行列積だけで、
    選択された候補の物理変位はroll.displacement_rollへそのまま渡す。
    """
    candidates = []

    for parent_id in parent_ids:
        parent_id = int(parent_id)
        node = history[parent_id].node

        # StateNodeは生成後に変化しないので、history全体を候補にする場合は
        # state_idごとのmodeを再利用できる。frontier探索でも同じ関数を使う。
        if mode_cache is not None and parent_id in mode_cache:
            mode = mode_cache[parent_id]
        else:
            dedx = calc_dedx(
                node,
                num_node,
                member,
                num_member,
                index_free,
            )
            mode = calc_mode(index_free, dedx, num_node)
            if mode_cache is not None:
                mode_cache[parent_id] = mode
        num_mode = mode.shape[1]
        if num_mode == 0:
            continue

        # +aと-aを組にして、少ない候補数でも両方向を調べる。
        num_base = (num_perturbations_per_parent + 1) // 2
        base_coefficients = rng.normal(size=(num_base, num_mode))
        coefficients = []
        for coefficient in base_coefficients:
            coefficients.append(coefficient)
            if len(coefficients) < num_perturbations_per_parent:
                coefficients.append(-coefficient)

        for coefficient in coefficients[:num_perturbations_per_parent]:
            coefficient_norm = np.linalg.norm(coefficient)
            if coefficient_norm < 1.0e-12:
                continue

            scaled_coefficient = (
                perturbation_scale * coefficient / coefficient_norm
            )
            displacement = (mode @ scaled_coefficient).reshape(num_node, 3)

            candidates.append(
                {
                    "parent_id": parent_id,
                    "coefficient": coefficient.copy(),
                    "displacement": displacement,
                    "input": candidate_input(
                        node,
                        displacement,
                        reference_point,
                        length_scale,
                    ),
                }
            )

    return candidates


def sample_frontier_parent_ids(frontier_archive, num_parents, rng):
    """frontier prototypeから重複なしで親StateNodeを選ぶ。"""
    if len(frontier_archive) == 0:
        return []

    state_ids = np.asarray(
        [cell["state_id"] for cell in frontier_archive],
        dtype=int,
    )
    sample_size = min(int(num_parents), len(state_ids))
    selected = rng.choice(state_ids, size=sample_size, replace=False)
    return selected.tolist()


def select_parent_ids(
    history,
    frontier_archive,
    parent_source,
    num_parents,
    rng,
):
    """摂動候補を作る親StateNodeを選ぶ。

    parent_source
    -------------
    frontier:
        frontier archiveのprototypeから最大num_parents個を一様抽出する。
    history_random:
        history内の全ての非終端状態を母集団として、最大num_parents個を
        一様抽出する。候補生成費用をfrontier条件と揃えたい比較に使う。
    history_all:
        history内の全ての非終端状態を返す。num_parentsは使用しない。

    root探索は共通bootstrapで行うため、本探索の公平な比較からは除く。
    flatとcollisionも物理的な終端なので除く。一方、最大rollout stepで
    打ち切られただけの有効状態は摂動可能な状態として残す。
    """
    if parent_source == "frontier":
        return sample_frontier_parent_ids(
            frontier_archive,
            num_parents,
            rng,
        )

    eligible_ids = np.asarray(
        [
            state.state_id
            for state in history
            if state.state_id != 0 and state.terminal_reason is None
        ],
        dtype=int,
    )
    if parent_source == "history_all":
        return eligible_ids.tolist()
    if parent_source == "history_random":
        sample_size = min(int(num_parents), len(eligible_ids))
        selected = rng.choice(
            eligible_ids,
            size=sample_size,
            replace=False,
        )
        return selected.tolist()

    raise ValueError(
        "FOLDING_NN_PARENT_SOURCE must be "
        "'frontier', 'history_random', or 'history_all'"
    )


def terminal_regression_target(
    history,
    terminal_state_id,
    member,
    reference_point,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """連続な終端教師vectorを作る。

    前半は終端状態のxy座標、後半は最後の非平坦状態のz座標である。
    後半を予測することで、終端形状で重なる部材間の上下関係を復元できる。
    """
    shape_descriptor, _ = terminal_descriptor(
        history,
        terminal_state_id,
        member,
        length_scale,
        layer_threshold,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )
    observation_state_id = find_last_nonflat_state_id(
        history,
        terminal_state_id,
        layer_threshold,
    )
    observation_z = (
        history[observation_state_id].node[:, 2] - reference_point
    ) / length_scale
    return np.concatenate((shape_descriptor, observation_z))


def append_training_sample(
    training_inputs,
    training_targets,
    training_stop_labels,
    model_input,
    stop_reason,
    history,
    terminal_state_id,
    member,
    reference_point,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """実rollout一回分を教師データへ追加する。"""
    training_inputs.append(np.asarray(model_input, dtype=float).copy())
    training_stop_labels.append(STOP_TO_INDEX[stop_reason])

    if stop_reason == "flat":
        target = terminal_regression_target(
            history,
            terminal_state_id,
            member,
            reference_point,
            length_scale,
            layer_threshold,
            overlap_tolerance,
            separation_tolerance,
            collinearity_tolerance,
            parallel_tolerance,
        )
    else:
        # collisionとtimeoutには有効な平坦終端descriptorがない。
        # loss計算時にmaskするため、ここでは同じ長さの0 vectorを保存する。
        num_node = history[0].node.shape[0]
        target = np.zeros(3 * num_node, dtype=float)

    training_targets.append(target)


def _make_network(input_dimension, continuous_dimension, hidden_dimension):
    """custom classを作らず、小さなMLPをSequentialで構成する。"""
    return torch_nn.Sequential(
        torch_nn.Linear(input_dimension, hidden_dimension),
        torch_nn.SiLU(),
        torch_nn.Linear(hidden_dimension, hidden_dimension),
        torch_nn.SiLU(),
        torch_nn.Linear(hidden_dimension, continuous_dimension + 3),
    )


def _nearest_normalized_rms_distance(
    query_inputs,
    reference_inputs,
    exclude_matching_index=False,
):
    """標準化済み入力について、最近傍までのRMS距離を返す。

    全組合せの差分tensorを作ると候補数が多い場合にメモリを消費するため、
    二乗距離の展開式を使って ``(候補数, 教師数)`` 行列だけを作る。
    """
    query = np.asarray(query_inputs, dtype=np.float32)
    reference = np.asarray(reference_inputs, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("support distance inputs must be two-dimensional")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference input dimensions must match")
    if len(reference) == 0:
        raise ValueError("at least one reference input is required")

    query_norm = np.sum(query * query, axis=1, keepdims=True)
    reference_norm = np.sum(reference * reference, axis=1)[None, :]
    squared_distance = (
        query_norm
        + reference_norm
        - 2.0 * query @ reference.T
    )
    # 浮動小数点誤差でごく小さな負値になる場合がある。
    squared_distance = np.maximum(squared_distance, 0.0)

    if exclude_matching_index:
        if len(query) != len(reference):
            raise ValueError(
                "matching indices can be excluded only for equal-sized arrays"
            )
        np.fill_diagonal(squared_distance, np.inf)

    input_dimension = max(query.shape[1], 1)
    return np.sqrt(np.min(squared_distance, axis=1) / input_dimension)


def _training_support_radius(x_normalized):
    """教師点群の局所間隔から、OOD判定用の正の半径を決める。

    2点以上なら各教師点から別の最近傍教師までのRMS距離の95%点を
    使用する。1点だけ、または全教師が重複する場合にも、後段で0除算
    しないよう小さな正のfallbackを返す。
    """
    normalized = np.asarray(x_normalized, dtype=np.float32)
    minimum_radius = 1.0e-6
    if normalized.ndim != 2:
        raise ValueError("normalized training inputs must be two-dimensional")
    if len(normalized) <= 1:
        return 1.0

    nearest = _nearest_normalized_rms_distance(
        normalized,
        normalized,
        exclude_matching_index=True,
    )
    finite = nearest[np.isfinite(nearest)]
    if len(finite) == 0:
        return 1.0
    radius = float(np.quantile(finite, 0.95))
    return max(radius, minimum_radius)


def train_ensemble(
    training_inputs,
    training_targets,
    training_stop_labels,
    num_models=5,
    hidden_dimension=64,
    epochs=200,
    learning_rate=1.0e-3,
    weight_decay=1.0e-4,
    stop_loss_weight=0.25,
    seed=0,
):
    """bootstrap samplingと初期値の異なる複数NNを学習する。"""
    if len(training_inputs) == 0:
        return None

    x = np.asarray(training_inputs, dtype=np.float32)
    y = np.asarray(training_targets, dtype=np.float32)
    stop = np.asarray(training_stop_labels, dtype=np.int64)
    flat_indices = np.flatnonzero(stop == STOP_TO_INDEX["flat"])
    if len(flat_indices) == 0:
        return None

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1.0e-6] = 1.0

    y_mean = y[flat_indices].mean(axis=0)
    y_std = y[flat_indices].std(axis=0)
    y_std[y_std < 1.0e-4] = 1.0

    x_normalized = (x - x_mean) / x_std
    y_normalized = (y - y_mean) / y_std
    input_dimension = x.shape[1]
    continuous_dimension = y.shape[1]

    # 小規模なCPU学習ではthreadを増やし過ぎない方が安定して速い。
    torch.set_num_threads(1)
    models = []

    for model_index in range(num_models):
        model_seed = int(seed + 1009 * model_index)
        torch.manual_seed(model_seed)
        bootstrap_rng = np.random.default_rng(model_seed)

        resampled_indices = bootstrap_rng.integers(
            0,
            len(x),
            size=len(x),
        )
        # 初期データが少ない段階で特定の終端を丸ごと失わないよう、全データを
        # 一度含めた上でbootstrap標本を追加する。初期値と重複度がmodelごとに異なる。
        bootstrap_indices = np.concatenate(
            (np.arange(len(x), dtype=int), resampled_indices)
        )

        x_tensor = torch.as_tensor(
            x_normalized[bootstrap_indices],
            dtype=torch.float32,
        )
        y_tensor = torch.as_tensor(
            y_normalized[bootstrap_indices],
            dtype=torch.float32,
        )
        stop_tensor = torch.as_tensor(
            stop[bootstrap_indices],
            dtype=torch.long,
        )

        model = _make_network(
            input_dimension,
            continuous_dimension,
            hidden_dimension,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        model.train()
        for _ in range(epochs):
            output = model(x_tensor)
            continuous_output = output[:, :continuous_dimension]
            stop_logits = output[:, continuous_dimension:]
            flat_mask = stop_tensor == STOP_TO_INDEX["flat"]

            if torch.any(flat_mask):
                continuous_loss = torch_nn.functional.mse_loss(
                    continuous_output[flat_mask],
                    y_tensor[flat_mask],
                )
            else:
                continuous_loss = continuous_output.sum() * 0.0

            stop_loss = torch_nn.functional.cross_entropy(
                stop_logits,
                stop_tensor,
            )
            loss = continuous_loss + stop_loss_weight * stop_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        models.append(model)

    return {
        "models": models,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "continuous_dimension": continuous_dimension,
        # 候補が教師分布から離れ過ぎていないかを、学習後に安価に
        # 判定できるよう、元の教師だけを標準化して保存する。
        "training_inputs_normalized": x_normalized.copy(),
        "training_support_radius": _training_support_radius(x_normalized),
    }


def predict_ensemble(ensemble, model_inputs):
    """全NNの連続終端予測と停止理由確率を返す。"""
    x = np.asarray(model_inputs, dtype=np.float32)
    x_normalized = (x - ensemble["x_mean"]) / ensemble["x_std"]
    x_tensor = torch.as_tensor(x_normalized, dtype=torch.float32)

    continuous_predictions = []
    stop_probabilities = []
    continuous_dimension = ensemble["continuous_dimension"]

    with torch.no_grad():
        for model in ensemble["models"]:
            output = model(x_tensor)
            continuous_normalized = output[:, :continuous_dimension].numpy()
            continuous = (
                continuous_normalized * ensemble["y_std"]
                + ensemble["y_mean"]
            )
            stop_probability = torch.softmax(
                output[:, continuous_dimension:],
                dim=1,
            ).numpy()
            continuous_predictions.append(continuous)
            stop_probabilities.append(stop_probability)

    return (
        np.asarray(continuous_predictions),
        np.asarray(stop_probabilities),
    )


def candidate_support_distance(ensemble, candidates):
    """各candidateから最近傍教師入力までの標準化RMS距離を返す。

    戻り値は半径で割っていない生の距離である。通常は
    ``distance <= ensemble["training_support_radius"]`` を教師分布内の
    目安として使用する。
    """
    if ensemble is None:
        raise ValueError("ensemble must not be None")
    if len(candidates) == 0:
        return np.empty(0, dtype=float)
    if "training_inputs_normalized" not in ensemble:
        raise KeyError(
            "ensemble does not contain normalized training inputs"
        )

    model_inputs = np.asarray(
        [candidate["input"] for candidate in candidates],
        dtype=np.float32,
    )
    if model_inputs.ndim != 2:
        raise ValueError("candidate inputs must form a two-dimensional array")
    normalized = (
        model_inputs - ensemble["x_mean"]
    ) / ensemble["x_std"]
    return _nearest_normalized_rms_distance(
        normalized,
        ensemble["training_inputs_normalized"],
    ).astype(float, copy=False)


def _make_cost_network(input_dimension, hidden_dimension):
    """計算step数を回帰する小さなMLPを作る。"""
    return torch_nn.Sequential(
        torch_nn.Linear(input_dimension, hidden_dimension),
        torch_nn.SiLU(),
        torch_nn.Linear(hidden_dimension, hidden_dimension),
        torch_nn.SiLU(),
        torch_nn.Linear(hidden_dimension, 1),
    )


def train_cost_ensemble(
    inputs,
    costs,
    num_models=3,
    hidden=32,
    epochs=200,
    seed=0,
):
    """入力からrollout計算costを予測するbootstrap ensembleを学習する。

    既存の終端予測ensembleとは独立したbundleを返す。空データの場合は
    ``None`` を返し、既存の ``train_ensemble`` と同じ扱いにする。
    """
    if len(inputs) == 0:
        return None

    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(costs, dtype=np.float32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("cost inputs must be a two-dimensional array")
    if len(x) != len(y):
        raise ValueError("inputs and costs must contain the same samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("cost training data must be finite")
    if num_models <= 0 or hidden <= 0 or epochs < 0:
        raise ValueError("invalid cost ensemble hyperparameters")

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1.0e-6] = 1.0
    cost_mean = float(y.mean())
    cost_std = float(y.std())
    if cost_std < 1.0e-6:
        cost_std = 1.0

    x_normalized = (x - x_mean) / x_std
    cost_normalized = (y - cost_mean) / cost_std
    torch.set_num_threads(1)
    models = []

    for model_index in range(int(num_models)):
        model_seed = int(seed + 1009 * model_index)
        torch.manual_seed(model_seed)
        bootstrap_rng = np.random.default_rng(model_seed)
        resampled_indices = bootstrap_rng.integers(
            0,
            len(x),
            size=len(x),
        )
        # 元データを必ず含めた上でbootstrap標本を加え、小標本でも
        # 一部のcost範囲を完全に失わないようにする。
        bootstrap_indices = np.concatenate(
            (np.arange(len(x), dtype=int), resampled_indices)
        )
        x_tensor = torch.as_tensor(
            x_normalized[bootstrap_indices],
            dtype=torch.float32,
        )
        cost_tensor = torch.as_tensor(
            cost_normalized[bootstrap_indices, None],
            dtype=torch.float32,
        )

        model = _make_cost_network(x.shape[1], int(hidden))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-4,
        )
        model.train()
        for _ in range(int(epochs)):
            prediction = model(x_tensor)
            loss = torch_nn.functional.mse_loss(
                prediction,
                cost_tensor,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        models.append(model)

    return {
        "models": models,
        "x_mean": x_mean,
        "x_std": x_std,
        "cost_mean": cost_mean,
        "cost_std": cost_std,
    }


def predict_cost_ensemble(ensemble, model_inputs):
    """cost ensembleの予測平均とmodel間標準偏差を返す。"""
    if ensemble is None:
        raise ValueError("ensemble must not be None")
    x = np.asarray(model_inputs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("cost prediction inputs must be two-dimensional")
    normalized = (x - ensemble["x_mean"]) / ensemble["x_std"]
    x_tensor = torch.as_tensor(normalized, dtype=torch.float32)

    predictions = []
    with torch.no_grad():
        for model in ensemble["models"]:
            normalized_cost = model(x_tensor)[:, 0].numpy()
            cost = (
                normalized_cost * ensemble["cost_std"]
                + ensemble["cost_mean"]
            )
            predictions.append(cost)

    member_predictions = np.asarray(predictions, dtype=float)
    return (
        np.mean(member_predictions, axis=0),
        np.std(member_predictions, axis=0),
    )


def decode_terminal_prediction(
    continuous_prediction,
    member,
    num_node,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """連続NN出力を既存archiveと同じ(shape_descriptor, layer_key)へ戻す。"""
    prediction = np.asarray(continuous_prediction, dtype=float)
    shape_dimension = 2 * num_node
    shape_descriptor = prediction[:shape_dimension]
    observation_z = prediction[shape_dimension:shape_dimension + num_node]

    if not np.all(np.isfinite(prediction)):
        return shape_descriptor, ((-1, -1, -1),)

    # shape_descriptorは既に無次元なのでlength_scale=1として重複判定する。
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


def _member_pairs(member):
    """layer codeの列に対応する全ての部材対を一定順序で返す。"""
    return [
        (member_i, member_j)
        for member_i in range(len(member))
        for member_j in range(member_i + 1, len(member))
    ]


def _decode_terminal_predictions_vectorized(
    continuous_predictions,
    member,
    num_node,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
):
    """多数の連続NN出力を、形状と固定長layer codeへ一括変換する。

    layer codeは各部材対について、0:非重複、1:若い番号が上、
    2:大きい番号が上、3:判定不能、とする。部材対を明示する既存の
    layer_keyと一対一に対応するが、固定長なので候補方向にvectorizeできる。
    """
    predictions = np.asarray(continuous_predictions, dtype=float)
    if collinearity_tolerance is None:
        collinearity_tolerance = overlap_tolerance
    shape_dimension = 2 * num_node
    shape_descriptors = predictions[:, :shape_dimension]
    terminal_xy = shape_descriptors.reshape(-1, num_node, 2)
    observation_z = predictions[
        :,
        shape_dimension:shape_dimension + num_node,
    ]

    pairs = _member_pairs(member)
    layer_codes = np.zeros((len(predictions), len(pairs)), dtype=np.int8)
    finite = np.all(np.isfinite(predictions), axis=1)
    degenerate = np.zeros(len(predictions), dtype=bool)

    for pair_index, (member_i, member_j) in enumerate(pairs):
        i0, i1 = member[member_i]
        j0, j1 = member[member_j]
        p0 = terminal_xy[:, i0]
        p1 = terminal_xy[:, i1]
        q0 = terminal_xy[:, j0]
        q1 = terminal_xy[:, j1]
        vector_i = p1 - p0
        vector_j = q1 - q0
        length_i = np.linalg.norm(vector_i, axis=1)
        length_j = np.linalg.norm(vector_j, axis=1)
        nondegenerate = (length_i > 1.0e-12) & (length_j > 1.0e-12)
        degenerate |= ~nondegenerate

        with np.errstate(divide="ignore", invalid="ignore"):
            cross_ij = (
                vector_i[:, 0] * vector_j[:, 1]
                - vector_i[:, 1] * vector_j[:, 0]
            )
            parallel_error = np.abs(cross_ij) / (length_i * length_j)

            q0_relative = q0 - p0
            q1_relative = q1 - p0
            distance_q0 = np.abs(
                vector_i[:, 0] * q0_relative[:, 1]
                - vector_i[:, 1] * q0_relative[:, 0]
            ) / length_i
            distance_q1 = np.abs(
                vector_i[:, 0] * q1_relative[:, 1]
                - vector_i[:, 1] * q1_relative[:, 0]
            ) / length_i

            axis = vector_i / length_i[:, None]
            projection_p0 = np.sum(p0 * axis, axis=1)
            projection_p1 = np.sum(p1 * axis, axis=1)
            projection_q0 = np.sum(q0 * axis, axis=1)
            projection_q1 = np.sum(q1 * axis, axis=1)
            interval_i_min = np.minimum(projection_p0, projection_p1)
            interval_i_max = np.maximum(projection_p0, projection_p1)
            interval_j_min = np.minimum(projection_q0, projection_q1)
            interval_j_max = np.maximum(projection_q0, projection_q1)
            overlap_start = np.maximum(interval_i_min, interval_j_min)
            overlap_end = np.minimum(interval_i_max, interval_j_max)
            overlap_length = overlap_end - overlap_start

        overlaps = (
            finite
            & nondegenerate
            & (parallel_error <= parallel_tolerance)
            & (np.maximum(distance_q0, distance_q1) <= collinearity_tolerance)
            & (overlap_length > overlap_tolerance)
        )
        if not np.any(overlaps):
            continue

        sample_position = 0.5 * (overlap_start + overlap_end)
        sample_node = p0 + (
            sample_position - projection_p0
        )[:, None] * axis
        with np.errstate(divide="ignore", invalid="ignore"):
            parameter_i = (
                np.sum((sample_node - p0) * vector_i, axis=1)
                / length_i**2
            )
            parameter_j = (
                np.sum((sample_node - q0) * vector_j, axis=1)
                / length_j**2
            )

        z_i = (
            (1.0 - parameter_i) * observation_z[:, i0]
            + parameter_i * observation_z[:, i1]
        )
        z_j = (
            (1.0 - parameter_j) * observation_z[:, j0]
            + parameter_j * observation_z[:, j1]
        )
        separation = z_i - z_j

        pair_code = layer_codes[:, pair_index]
        pair_code[overlaps & (separation > separation_tolerance)] = 1
        pair_code[overlaps & (separation < -separation_tolerance)] = 2
        pair_code[
            overlaps & (np.abs(separation) <= separation_tolerance)
        ] = 3

    # 非有限な予測や縮退部材は、有効な新規cellとは数えない。
    valid = finite & ~degenerate & ~np.any(layer_codes == 3, axis=1)
    return shape_descriptors, layer_codes, valid


def _layer_key_to_codes(layer_key, member):
    """既存archiveの可変長layer_keyを固定長codeへ変換する。"""
    pairs = _member_pairs(member)
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    codes = np.zeros(len(pairs), dtype=np.int8)
    for member_i, member_j, upper_member in layer_key:
        pair_index = pair_to_index[(member_i, member_j)]
        if upper_member == member_i:
            codes[pair_index] = 1
        elif upper_member == member_j:
            codes[pair_index] = 2
        else:
            codes[pair_index] = 3
    return codes


def score_candidates(
    ensemble,
    candidates,
    elite_archive,
    member,
    num_node,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
    return_flat_probability=False,
):
    """ensembleを経験分布としてP_newをMonte Carlo近似する。

    p(y|x,D) を M 個のNN予測の経験分布で近似すると、実装式は

        P_new(x) ~= (1/M) sum_m p_m(flat|x)
                    * I[archive_check(A_t, y_m(x)) == new]

    となる。したがって、この値は有限ensembleによる近似値であり、
    厳密に校正された確率ではない。
    """
    model_inputs = np.asarray([candidate["input"] for candidate in candidates])
    continuous_predictions, stop_probabilities = predict_ensemble(
        ensemble,
        model_inputs,
    )

    num_models, num_candidates, continuous_dimension = (
        continuous_predictions.shape
    )
    flattened_predictions = continuous_predictions.reshape(
        num_models * num_candidates,
        continuous_dimension,
    )
    shapes, layer_codes, valid = _decode_terminal_predictions_vectorized(
        flattened_predictions,
        member,
        num_node,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )

    belongs_to_existing_cell = np.zeros(len(shapes), dtype=bool)
    for cell in elite_archive:
        cell_codes = _layer_key_to_codes(cell["layer_key"], member)
        same_layer = np.all(layer_codes == cell_codes, axis=1)
        difference = (
            shapes.reshape(-1, num_node, 2)
            - np.asarray(cell["shape_descriptor"]).reshape(1, num_node, 2)
        )
        node_distance = np.linalg.norm(difference, axis=2)
        rms_distance = np.sqrt(np.mean(node_distance**2, axis=1))
        max_distance = np.max(node_distance, axis=1)
        same_shape = (
            (rms_distance <= rms_tolerance)
            & (max_distance <= max_tolerance)
        )
        belongs_to_existing_cell |= same_layer & same_shape

    predicted_new = (
        valid & ~belongs_to_existing_cell
    ).reshape(num_models, num_candidates)
    flat_probability = stop_probabilities[:, :, STOP_TO_INDEX["flat"]]
    new_probability = np.mean(
        flat_probability * predicted_new,
        axis=0,
    )

    # NN間のdescriptor予測の分散を、未学習度の補助指標にする。
    uncertainty = np.mean(
        np.var(continuous_predictions, axis=0),
        axis=1,
    )

    # ensemble平均のarchiveからの距離は、P_newが同点の候補を並べるために使う。
    mean_predictions = np.mean(continuous_predictions, axis=0)
    mean_shapes, mean_layer_codes, _ = (
        _decode_terminal_predictions_vectorized(
            mean_predictions,
            member,
            num_node,
            overlap_tolerance,
            separation_tolerance,
            collinearity_tolerance,
            parallel_tolerance,
        )
    )
    if len(elite_archive) == 0:
        predicted_distance = np.ones(num_candidates, dtype=float)
    else:
        predicted_distance = np.full(num_candidates, np.inf, dtype=float)
        has_same_layer = np.zeros(num_candidates, dtype=bool)
        for cell in elite_archive:
            cell_codes = _layer_key_to_codes(cell["layer_key"], member)
            same_layer = np.all(mean_layer_codes == cell_codes, axis=1)
            difference = (
                mean_shapes.reshape(-1, num_node, 2)
                - np.asarray(cell["shape_descriptor"]).reshape(
                    1,
                    num_node,
                    2,
                )
            )
            node_distance = np.linalg.norm(difference, axis=2)
            rms_distance = np.sqrt(np.mean(node_distance**2, axis=1))
            max_distance = np.max(node_distance, axis=1)
            normalized_distance = np.maximum(
                rms_distance / max(rms_tolerance, 1.0e-12),
                max_distance / max(max_tolerance, 1.0e-12),
            )
            predicted_distance[same_layer] = np.minimum(
                predicted_distance[same_layer],
                normalized_distance[same_layer],
            )
            has_same_layer |= same_layer
        predicted_distance[~has_same_layer] = 2.0

    if return_flat_probability:
        return (
            new_probability,
            uncertainty,
            predicted_distance,
            np.mean(flat_probability, axis=0),
        )
    return new_probability, uncertainty, predicted_distance


def select_candidates(
    candidates,
    new_probability,
    uncertainty,
    predicted_distance,
    num_guided,
    num_random,
    rng,
):
    """P_new上位候補と完全ランダム候補を合わせて選ぶ。"""
    num_candidates = len(candidates)
    if num_candidates == 0:
        return []

    order = sorted(
        range(num_candidates),
        key=lambda index: (
            new_probability[index],
            predicted_distance[index],
            uncertainty[index],
        ),
        reverse=True,
    )

    selected_indices = []
    selected_parents = set()

    if num_guided > 0:
        # 最初は同じ親を重複させず、摂動位置の多様性を確保する。
        for index in order:
            parent_id = candidates[index]["parent_id"]
            if parent_id in selected_parents:
                continue
            selected_indices.append(index)
            selected_parents.add(parent_id)
            if len(selected_indices) >= num_guided:
                break

        # 親数が不足した場合だけ、同じ親の別係数で補う。
        for index in order:
            if len(selected_indices) >= num_guided:
                break
            if index not in selected_indices:
                selected_indices.append(index)

    remaining_indices = np.asarray(
        [
            index
            for index in range(num_candidates)
            if index not in selected_indices
        ],
        dtype=int,
    )
    random_count = min(num_random, len(remaining_indices))
    if random_count > 0:
        random_indices = rng.choice(
            remaining_indices,
            size=random_count,
            replace=False,
        ).tolist()
    else:
        random_indices = []

    selected = []
    for rank, index in enumerate(selected_indices + random_indices):
        candidate = candidates[index].copy()
        candidate["new_probability"] = float(new_probability[index])
        candidate["uncertainty"] = float(uncertainty[index])
        candidate["predicted_distance"] = float(predicted_distance[index])
        candidate["selection_type"] = (
            "guided" if rank < len(selected_indices) else "random"
        )
        selected.append(candidate)

    return selected


def select_random_candidates(candidates, num_selected, rng):
    """NN scoreを使わず、親の重複を避けながら候補を一様選択する。

    guided選択も最初は異なる親を選ぶため、random対照にも同じ制約を
    課して「親の多様化」ではなくNNによる順位付けだけを比較する。
    """
    if len(candidates) == 0 or num_selected <= 0:
        return []

    random_order = rng.permutation(len(candidates)).tolist()
    selected_indices = []
    selected_parents = set()

    for index in random_order:
        parent_id = candidates[index]["parent_id"]
        if parent_id in selected_parents:
            continue
        selected_indices.append(index)
        selected_parents.add(parent_id)
        if len(selected_indices) >= num_selected:
            break

    # 親状態数が不足した場合だけ、同じ親の別摂動で残りを補う。
    for index in random_order:
        if len(selected_indices) >= num_selected:
            break
        if index not in selected_indices:
            selected_indices.append(index)

    selected = []
    for index in selected_indices:
        candidate = candidates[index].copy()
        candidate["new_probability"] = np.nan
        candidate["uncertainty"] = np.nan
        candidate["predicted_distance"] = np.nan
        candidate["selection_type"] = "random"
        selected.append(candidate)
    return selected
