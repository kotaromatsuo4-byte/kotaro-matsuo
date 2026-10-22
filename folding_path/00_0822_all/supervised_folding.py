"""Goal-conditioned direction regression with iterative dataset aggregation.

Teacher: MM.T @ (target - current), as in mechanics_origin.predictor.
Student rollout never receives target coordinates. See SUPERVISED_FOLDING.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import mechanics

BASE = Path(__file__).resolve().parent
GOALS = (2.0, 3.0)


class Geometry:
    def __init__(self, directory, step_width=0.1):
        self.directory = Path(directory)
        self.initial = np.loadtxt(self.directory / "node_initial.dat", delimiter=",")
        self.member = np.loadtxt(self.directory / "member.dat", delimiter=",", dtype=int).reshape(-1, 2)
        self.fix = np.loadtxt(self.directory / "fix.dat", delimiter=",", dtype=int).reshape(-1, 2)
        if self.initial.ndim != 2 or self.initial.shape[1] != 3 or np.ptp(self.initial[:, 1]) > 1e-10:
            raise ValueError("This implementation requires planar x-z coordinates")
        self.coordinate_size = self.initial.size
        self.free = mechanics.index_free(len(self.initial), self.fix)
        self.lengths = np.linalg.norm(self.initial[self.member[:, 0]] - self.initial[self.member[:, 1]], axis=1)
        self.scale = float(self.lengths.max())
        self.initial_potential = mechanics.compute_flatness_potential(self.initial, 1)
        if self.initial_potential <= 1e-12:
            raise ValueError("A nonflat reference shape is required")
        self.step_width = float(step_width)
        self.pairs = mechanics.nonadjacent_bar_pairs(self.member)

    def modes(self, node):
        return mechanics.canonical_modes(node, self.member, self.free)

    def metrics(self, node, goal):
        flatness = mechanics.compute_flatness_potential(node, 1) / self.initial_potential
        spread = mechanics.projected_spread(node, 1, self.scale)
        strain = mechanics.current_strain(node, self.member, self.lengths)
        return dict(flatness=float(flatness), spread=float(spread),
                    spread_error=float(abs(spread - goal)), strain=float(strain),
                    success=int(flatness <= 1e-3 and abs(spread - goal) <= 0.05 and strain <= 1e-8))

    def advance(self, node, direction, width=None):
        """Target-free constraint correction and crossing rejection."""
        width = self.step_width if width is None else float(width)
        corrected, accepted, failed, *_ = mechanics.project_candidate(
            node, direction, width, self.member, self.lengths, self.free,
            max_step_width=self.step_width,
        )
        if failed or abs(accepted) == 0:
            return node.copy(), "projection_rejected"
        if mechanics.has_proper_self_crossing_planar(node, corrected, self.member, self.pairs):
            return node.copy(), "crossing_rejected"
        return corrected, "accepted"


def load_targets(geometry):
    targets = {}
    for goal in GOALS:
        target = np.loadtxt(geometry.directory / f"node_target_d{int(goal)}.dat", delimiter=",")
        if target.shape != geometry.initial.shape or not np.isfinite(target).all():
            raise ValueError(f"Invalid target D{goal}")
        fixed = (geometry.fix[:, 0], geometry.fix[:, 1])
        if not np.allclose(target[fixed], geometry.initial[fixed], atol=1e-8, rtol=0):
            raise ValueError("Target changes fixed coordinates")
        # Input reference coordinates are rounded: tolerate small length mismatch.
        lengths = np.linalg.norm(target[geometry.member[:, 0]] - target[geometry.member[:, 1]], axis=1)
        if np.max(abs(lengths - geometry.lengths)) > 1e-3 * geometry.scale:
            raise ValueError("Target has incompatible bar lengths")
        if abs(mechanics.projected_spread(target, 1, geometry.scale) - goal) > 0.05:
            raise ValueError("Target spread disagrees with its D label")
        targets[goal] = target
    return targets


def teacher_direction(node, target, modes):
    """The signed unit predictor direction; no corrector displacement labels."""
    gradient = (target - node).reshape(-1)
    projected = modes @ (modes.T @ gradient)
    norm = float(np.linalg.norm(projected))
    return projected / max(norm, 1e-12), norm


def teacher_advance(geometry, node, target, direction):
    # Check descent AFTER constraint correction as well as feasibility.
    error = np.linalg.norm(node - target)
    width = geometry.step_width
    while width >= 1e-4:
        candidate, status = geometry.advance(node, direction, width)
        if status == "accepted" and np.linalg.norm(candidate - target) < error - 1e-12:
            return candidate, status
        width *= 0.5
    return node.copy(), "teacher_stalled"


class DirectionNet(nn.Module):
    def __init__(self, hidden=96, coordinate_size=15):
        super().__init__()
        self.coordinate_size = int(coordinate_size)
        self.body = nn.Sequential(nn.Linear(self.coordinate_size + 1, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, self.coordinate_size))

    def forward(self, inputs, projectors):
        raw = self.body(inputs)
        projected = torch.bmm(projectors, raw.unsqueeze(-1)).squeeze(-1)
        return F.normalize(projected, dim=-1, eps=1e-8)


@torch.no_grad()
def student_direction(model, geometry, node, goal):
    modes = geometry.modes(node)
    inputs = np.append(node.reshape(-1) / geometry.scale, goal).astype(np.float32)
    projector = (modes @ modes.T).astype(np.float32)
    return model(torch.from_numpy(inputs[None]), torch.from_numpy(projector[None]))[0].numpy()


def generate_starts(geometry, count, rng, min_walk, max_walk, excluded=()):
    """Independent feasible random walks from the reference, never test rollouts."""
    result = []
    for _ in range(max(100, count * 100)):
        node = geometry.initial.copy()
        for _ in range(int(rng.integers(min_walk, max_walk + 1))):
            modes = geometry.modes(node)
            if modes.shape[1] == 0:
                break
            direction = modes @ rng.normal(size=modes.shape[1])
            direction /= max(np.linalg.norm(direction), 1e-12)
            node, _ = geometry.advance(node, direction)
        if geometry.metrics(node, 2)["flatness"] < 0.05:
            continue
        if any(np.linalg.norm(node - other) / geometry.scale < 0.03 for other in [*excluded, *result]):
            continue
        result.append(node)
        if len(result) == count:
            return np.stack(result)
    raise RuntimeError("Could not generate enough distinct feasible starts; increase walk range")


class Dataset:
    def __init__(self, geometry, targets, spacing=0.03):
        self.geometry, self.targets, self.spacing = geometry, targets, spacing
        self.rows = []
        self.skipped_zero = 0

    def add_path(self, path, round_index, rollout_id):
        last = None
        for index, node in enumerate(path):
            if last is not None and index != len(path) - 1 and np.linalg.norm(node - last) < self.spacing * self.geometry.scale:
                continue
            if last is not None and np.linalg.norm(node - last) < 1e-10:
                continue
            last = node
            modes = self.geometry.modes(node)
            directions, norms = zip(*(teacher_direction(node, self.targets[g], modes) for g in GOALS))
            self.skipped_zero += sum(n <= 1e-8 for n in norms)
            self.rows.append((node.copy(), modes @ modes.T, np.stack(directions),
                              np.array(norms), round_index, rollout_id))

    def arrays(self):
        if not self.rows:
            raise ValueError("No teacher states collected")
        nodes, projectors, directions, norms, rounds, ids = zip(*self.rows)
        return dict(nodes=np.array(nodes), projectors=np.array(projectors, dtype=np.float32),
                    directions=np.array(directions, dtype=np.float32), gradient_norms=np.array(norms),
                    source_round=np.array(rounds), rollout_id=np.array(ids))

    def tensors(self):
        data = self.arrays()
        valid = data["gradient_norms"] > 1e-8
        state_index, goal_index = np.where(valid)
        inputs = np.column_stack((data["nodes"][state_index].reshape(-1, self.geometry.coordinate_size) / self.geometry.scale,
                                  np.array(GOALS)[goal_index])).astype(np.float32)
        return (torch.from_numpy(inputs), torch.from_numpy(data["projectors"][state_index]),
                torch.from_numpy(data["directions"][state_index, goal_index]),
                torch.from_numpy(goal_index))


def train(model, optimizer, dataset, epochs, batch_size, generator,
          history=None, round_index=0, update_offset=0):
    inputs, projectors, labels, goals = dataset.tensors()
    groups = [torch.where(goals == i)[0] for i in range(2)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Both goals need nonzero teacher labels")
    model.train()
    total_loss, updates = 0.0, 0
    for epoch in range(epochs):
        epoch_loss, epoch_updates = 0.0, 0
        for _ in range(max(1, (len(inputs) + batch_size - 1) // batch_size)):
            # Equal goal weight, regardless of path length or zero-gradient exclusions.
            indices = torch.cat([group[torch.randint(len(group), (batch_size // 2,), generator=generator)] for group in groups])
            prediction = model(inputs[indices], projectors[indices])
            loss = ((prediction - labels[indices]) ** 2).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            updates += 1
            epoch_loss += float(loss.detach())
            epoch_updates += 1
        if history is not None:
            history.append(dict(round=round_index, epoch=epoch + 1,
                                cumulative_updates=update_offset + updates,
                                loss=epoch_loss / epoch_updates,
                                states=len(dataset.rows), labels=len(inputs)))
    model.eval()
    return dict(loss=total_loss / updates, updates=updates, states=len(dataset.rows), labels=len(inputs),
                skipped_zero_labels=dataset.skipped_zero)


def rollout(geometry, initial, goal, max_steps, model=None, target=None):
    if (model is None) == (target is None):
        raise ValueError("Choose exactly one controller: model or teacher target")
    node = initial.copy()
    path = [node.copy()]
    rejected, reason = 0, "timeout"
    for _ in range(max_steps):
        if geometry.metrics(node, goal)["success"]:
            reason = "success"
            break
        if target is not None:
            direction, norm = teacher_direction(node, target, geometry.modes(node))
            if norm <= 1e-8:
                reason = "zero_teacher_gradient"
                break
            candidate, status = teacher_advance(geometry, node, target, direction)
        else:
            direction = student_direction(model, geometry, node, goal)
            candidate, status = geometry.advance(node, direction)
        if status != "accepted" or np.linalg.norm(candidate - node) < 1e-12:
            rejected += 1
            reason = status if status != "accepted" else "zero_student_direction"
            # Deterministic policy will repeat the same action from an unchanged state.
            break
        node = candidate
        path.append(node.copy())
    metrics = geometry.metrics(node, goal)
    if metrics["success"]:
        reason = "success"
    return np.array(path), dict(**metrics, steps=len(path) - 1, rejected=rejected, reason=reason)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, geometry, starts, goals, max_steps, directory, round_index):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for start_id, initial in enumerate(starts):
        for goal in goals:
            path, metrics = rollout(geometry, initial, goal, max_steps, model=model)
            rows.append(dict(round=round_index, start_id=start_id, goal=goal, **metrics))
            np.savez_compressed(directory / f"start{start_id:03d}_d{goal:g}.npz", nodes=path)
    write_csv(directory / "rollouts.csv", rows)
    return rows


def save_checkpoint(path, model, optimizer, generator, args, round_index):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    hidden=args.hidden, coordinate_size=model.coordinate_size,
                    round=round_index, torch_rng=torch.get_rng_state(),
                    batch_rng=generator.get_state()), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, default=BASE / "model/five_bar_linkage")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--train-starts", type=int, default=16)
    parser.add_argument("--test-starts", type=int, default=8)
    parser.add_argument("--test-starts-from", type=Path,
                        help="Reuse the test array of an earlier starts.npz; exclude it from new training starts")
    parser.add_argument("--walk-min", type=int, default=20)
    parser.add_argument("--walk-max", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--step-width", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-spacing", type=float, default=0.03)
    parser.add_argument("--eval-goals", type=float, nargs="+", default=list(GOALS))
    parser.add_argument("--evaluate-only", type=Path, help="Existing run; optional --eval-goals 2 2.5 3")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evaluate_only:
        run = args.evaluate_only.resolve()
        design = json.loads((run / "design.json").read_text(encoding="utf-8"))
        geometry = Geometry(run / "model", design["step_width"])
        checkpoint = torch.load(run / "checkpoint_final.pt", weights_only=True)
        model = DirectionNet(checkpoint["hidden"], checkpoint.get("coordinate_size", 15))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        starts = np.load(run / "starts.npz")["test"]
        output = run / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        rows = evaluate(model, geometry, starts, args.eval_goals, design["max_steps"], output, checkpoint["round"])
        print(f"Evaluation saved: {output}", flush=True)
        print(json.dumps(rows), flush=True)
        return
    for key in ("train_starts", "test_starts", "walk_min", "max_steps", "epochs", "hidden"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if (args.walk_max < args.walk_min or args.rounds < 0 or args.batch_size < 2
            or args.batch_size % 2 or args.step_width <= 0 or args.lr <= 0 or args.sample_spacing <= 0):
        parser.error("Invalid walk range, rounds, even batch size, width, learning rate or spacing")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    geometry = Geometry(args.model_directory, args.step_width)
    targets = load_targets(geometry)
    fixed_test = None
    if args.test_starts_from:
        with np.load(args.test_starts_from) as archive:
            fixed_test = archive["test"].copy()
        if fixed_test.shape != (args.test_starts, *geometry.initial.shape) or not np.isfinite(fixed_test).all():
            parser.error("Reused test starts have incompatible count, shape or nonfinite coordinates")
        fixed = (geometry.fix[:, 0], geometry.fix[:, 1])
        for node in fixed_test:
            if (mechanics.current_strain(node, geometry.member, geometry.lengths) > 1e-8
                    or not np.allclose(node[fixed], geometry.initial[fixed], atol=1e-10, rtol=0)):
                parser.error("Reused test starts violate length or fixed-coordinate constraints")
    output = args.output or BASE / "result/experiments" / ("supervised_" + args.model_directory.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "model").mkdir()
    import shutil
    for name in ("node_initial.dat", "member.dat", "fix.dat", "node_target_d2.dat", "node_target_d3.dat"):
        shutil.copyfile(args.model_directory / name, output / "model" / name)
    design = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    design.update(training_goals=GOALS, coordinate_size=geometry.coordinate_size,
                  scale=geometry.scale, flatness_reference=geometry.initial_potential,
                  test_seed=None if fixed_test is not None else args.seed + 10000,
                  test_starts_sha256=(hashlib.sha256(args.test_starts_from.read_bytes()).hexdigest()
                                      if args.test_starts_from else None),
                  teacher="normalized MM.T(target-current); corrected descent",
                  source_sha256={name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                                 for name in ("supervised_folding.py", "plot_supervised_learning.py",
                                              "mechanics.py", "mechanics_origin.py")})
    (output / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(f"Run: {output}", flush=True)
    train_starts = generate_starts(geometry, args.train_starts, np.random.default_rng(args.seed),
                                   args.walk_min, args.walk_max, () if fixed_test is None else fixed_test)
    test_starts = (fixed_test if fixed_test is not None else generate_starts(
        geometry, args.test_starts, np.random.default_rng(args.seed + 10000), args.walk_min, args.walk_max, train_starts))
    np.savez_compressed(output / "starts.npz", train=train_starts, test=test_starts)
    dataset = Dataset(geometry, targets, args.sample_spacing)
    collection_rows, evaluation_rows, training_rows, epoch_rows = [], [], [], []
    teacher_dir = output / "teacher_paths"
    teacher_dir.mkdir()
    for start_id, initial in enumerate(train_starts):
        for goal in GOALS:
            path, metrics = rollout(geometry, initial, goal, args.max_steps, target=targets[goal])
            dataset.add_path(path, 0, len(collection_rows))
            collection_rows.append(dict(round=0, start_id=start_id, goal=goal, **metrics))
            np.savez_compressed(teacher_dir / f"start{start_id:03d}_d{goal:g}.npz", nodes=path)
        print(f"Teacher starts: {start_id + 1}/{len(train_starts)}, states={len(dataset.rows)}", flush=True)
    model = DirectionNet(args.hidden, geometry.coordinate_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    evaluation_rows.extend(evaluate(model, geometry, test_starts, args.eval_goals, args.max_steps, output / "eval_initial", -1))
    for round_index in range(args.rounds + 1):
        if round_index:
            paths_dir = output / f"student_paths_round{round_index:03d}"
            paths_dir.mkdir()
            for start_id, initial in enumerate(train_starts):
                for goal in GOALS:
                    path, metrics = rollout(geometry, initial, goal, args.max_steps, model=model)
                    dataset.add_path(path, round_index, len(collection_rows))
                    collection_rows.append(dict(round=round_index, start_id=start_id, goal=goal, **metrics))
                    np.savez_compressed(paths_dir / f"start{start_id:03d}_d{goal:g}.npz", nodes=path)
                print(f"Round {round_index} collection: {start_id + 1}/{len(train_starts)}", flush=True)
        stats = train(model, optimizer, dataset, args.epochs, args.batch_size, generator,
                      history=epoch_rows, round_index=round_index,
                      update_offset=sum(row["updates"] for row in training_rows))
        training_rows.append(dict(round=round_index, **stats))
        save_checkpoint(output / f"checkpoint_round{round_index:03d}.pt", model, optimizer, generator, args, round_index)
        save_checkpoint(output / "checkpoint_final.pt", model, optimizer, generator, args, round_index)
        np.savez_compressed(output / "dataset.npz", **dataset.arrays())
        evaluation_rows.extend(evaluate(model, geometry, test_starts, args.eval_goals, args.max_steps,
                                        output / f"eval_round{round_index:03d}", round_index))
        write_csv(output / "collection.csv", collection_rows)
        write_csv(output / "training.csv", training_rows)
        write_csv(output / "training_epochs.csv", epoch_rows)
        write_csv(output / "evaluation.csv", evaluation_rows)
        from plot_supervised_learning import plot_run
        plot_run(output)
        latest = [r for r in evaluation_rows if r["round"] == round_index]
        rates = {g: np.mean([r["success"] for r in latest if r["goal"] == g]) for g in args.eval_goals}
        print(f"Round {round_index}: {stats}, held-out success={rates}", flush=True)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
