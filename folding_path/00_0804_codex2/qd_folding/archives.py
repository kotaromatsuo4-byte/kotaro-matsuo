"""The valid terminal archive and complete trajectory history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .descriptors import terminal_cell
from .mechanics import Model


@dataclass(slots=True)
class ArchiveUpdate:
    status: str
    key: tuple[str, str] | None
    relative_improvement: float = 0.0
    old_steps: int | None = None
    new_steps: int | None = None

    @property
    def is_new(self) -> bool:
        return self.status == "new_cell"

    @property
    def is_improvement(self) -> bool:
        return self.status == "elite_improved"


@dataclass
class OutcomeArchive:
    """One shortest complete path for each shape-layer pair."""

    cells: dict[tuple[str, str], dict] = field(default_factory=dict)
    update_count: int = 0

    def update(
        self,
        trajectory: dict,
        model: Model,
        config,
        *,
        evaluation_index: int,
        method: str,
        candidate_metadata: dict[str, Any] | None = None,
    ) -> ArchiveUpdate:
        if (
            not trajectory["is_flat"]
            or trajectory["collided"]
            or trajectory["failed"]
            or trajectory["terminal_reason"] == "budget_exhausted"
        ):
            return ArchiveUpdate("rejected", None)

        key, metadata = terminal_cell(trajectory, model, config)
        logical_steps = int(trajectory["path"].shape[1] - 1)
        previous = self.cells.get(key)
        if previous is None:
            status = "new_cell"
            relative = 0.0
            old_steps = None
        else:
            old_steps = int(previous["logical_steps"])
            if logical_steps >= old_steps:
                return ArchiveUpdate(
                    "duplicate",
                    key,
                    old_steps=old_steps,
                    new_steps=logical_steps,
                )
            status = "elite_improved"
            relative = (old_steps - logical_steps) / max(old_steps, 1)

        self.cells[key] = {
            "key": key,
            "shape": metadata["shape"],
            "layer_order": metadata["layer_order"],
            "layer_observation_step": metadata["layer_observation_step"],
            "terminal_descriptor": metadata["terminal_descriptor"],
            "trajectory": trajectory,
            "logical_steps": logical_steps,
            "evaluation_index": evaluation_index,
            "method": method,
            "candidate_metadata": candidate_metadata or {},
        }
        self.update_count += 1
        return ArchiveUpdate(
            status,
            key,
            relative_improvement=relative,
            old_steps=old_steps,
            new_steps=logical_steps,
        )

    def copy(self) -> "OutcomeArchive":
        copied = OutcomeArchive()
        copied.cells = dict(self.cells)
        copied.update_count = self.update_count
        return copied

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def shape_count(self) -> int:
        return len({key[0] for key in self.cells})

    @property
    def layer_count(self) -> int:
        return len({key[1] for key in self.cells})


@dataclass
class HistoryStore:
    """All generated trajectories; active search repositories remain separate."""

    trajectories: list[dict] = field(default_factory=list)

    def append(self, trajectory: dict) -> int:
        history_id = len(self.trajectories)
        trajectory["history_id"] = history_id
        self.trajectories.append(trajectory)
        return history_id

    @property
    def state_count(self) -> int:
        return sum(value["path"].shape[1] for value in self.trajectories)
