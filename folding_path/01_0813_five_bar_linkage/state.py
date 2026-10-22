"""StateNodeと、その親子関係を持つhistory。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StateNode:
    state_id: int                    # history全体で一意な状態番号
    node: np.ndarray                 # その状態の節点座標、shapeは(num_node, 3)
    parent_id: int | None            # 一つ前の状態番号。rootだけNone
    children_ids: list[int]          # この状態から生成された直接の子状態
    depth: int                       # rootからの成功step数
    terminal_reason: str | None      # None、"flat"、"collision"
    potential: float                 # 重力ポテンシャルのキャッシュ
    # 親状態に加えたcorrector前の物理変位。rootだけNone。
    # child.node-parent.nodeはcorrector後の変位なので、GP入力には使わない。
    incoming_displacement: np.ndarray | None = None
    incoming_step_type: str | None = None  # None、"perturbation"、"deterministic"

# root state
def create_history(initial_node,initial_potential,):
    
    root = StateNode(
        state_id=0,
        node=np.asarray(initial_node, dtype=float).copy(),
        parent_id=None,
        children_ids=[],
        depth=0,
        terminal_reason=None,
        potential=float(initial_potential),
        incoming_displacement=None,
        incoming_step_type=None,
    )
    return [root]

# state_idからStateNodeを取り出す
def get_state(history, state_id):
    return history[state_id]

# 終端から親をたどり、rootから終端までのstate_id列を返す。
def get_path_state_ids(
    history: list[StateNode],
    terminal_state_id: int,
) -> list[int]:

    path: list[int] = []
    state_id: int | None = terminal_state_id
    while state_id is not None:
        path.append(state_id)
        state_id = get_state(history, state_id).parent_id
    return list(reversed(path))
