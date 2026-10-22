"""Controlled supervised transfer experiments (D2/D3 -> D4).

Training/validation/test starts are separated before teacher trajectories are
generated. Label budgets are nested prefixes of the same support trajectories.
No target coordinates are ever passed to student policies or feasibility checks.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mechanics
from supervised_folding import BASE, Geometry, DirectionNet, generate_starts, load_targets, teacher_direction, write_csv


def target_d4():
    target = np.zeros((7, 3))
    target[:, 0] = [-1, 0, 1, 2, 3, 2, 1]
    return target


def propose(g, node, direction, width, backtrack=True):
    """Only geometry is used. No target-dependent acceptance criterion."""
    attempts = 0
    reason = "zero_direction"
    if np.linalg.norm(direction) < 1e-10:
        return node.copy(), 0., attempts, reason
    while width >= 1e-4:
        attempts += 1
        candidate, accepted, failed, *_ = mechanics.project_candidate(
            node, direction, width, g.member, g.lengths, g.free, max_step_width=g.step_width)
        reason = "projection_rejected"
        if not failed and accepted:
            crossing = (g.collision_checker(node, candidate) if hasattr(g, "collision_checker") else
                        mechanics.has_proper_self_crossing_planar(node, candidate, g.member, g.pairs))
            reason = "crossing_rejected" if crossing else "accepted"
            if not crossing:
                return candidate, abs(float(accepted)), attempts, reason
        if not backtrack:
            break
        width *= .5
    return node.copy(), 0., attempts, reason


def oracle(g, node, target, modes=None):
    modes = g.modes(node) if modes is None else modes
    direction, norm = teacher_direction(node, target, modes)
    if norm <= 1e-8:
        return direction, 0., node.copy(), "zero_gradient", 0
    width, calls = g.step_width, 0
    error = np.linalg.norm(node - target)
    while width >= 1e-4:
        candidate, accepted, attempts, reason = propose(g, node, direction, width, True)
        calls += attempts
        if accepted and np.linalg.norm(candidate - target) < error - 1e-12:
            return direction, accepted, candidate, "accepted", calls
        width *= .5
    return direction, 0., node.copy(), "teacher_stalled", calls


class ModalStudent(nn.Module):
    """Shared encoder of (coordinates, D, mode); outputs mode coefficients and width."""
    def __init__(self, coordinate_size=21, hidden=96):
        super().__init__()
        self.coordinate_size, self.hidden = coordinate_size, hidden
        self.predicts_width = True
        self.encoder = nn.Sequential(nn.Linear(2 * coordinate_size + 1, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.coefficient = nn.Linear(hidden, 1)
        self.width = nn.Sequential(nn.Linear(2 * coordinate_size + 1 + hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, inputs, modes, teacher_direction=None):
        mask = modes.square().sum(-1) > 1e-10
        expanded = inputs[:, None, :].expand(-1, modes.shape[1], -1)
        features = self.encoder(torch.cat([expanded, modes], -1))
        coefficients = torch.tanh(self.coefficient(features).squeeze(-1)) * mask
        direction = F.normalize(torch.bmm(coefficients[:, None, :], modes).squeeze(1), dim=-1, eps=1e-8)
        pooled = (features * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        conditioning = direction if teacher_direction is None else teacher_direction
        width = torch.sigmoid(self.width(torch.cat([inputs, conditioning, pooled], -1)).squeeze(-1))
        return direction, width


def pad_modes(g, node):
    modes = g.modes(node)
    # Generous fixed pad within this structure; zero columns are masked by the NN.
    padded = np.zeros((g.coordinate_size, g.coordinate_size), dtype=np.float32)
    padded[:modes.shape[1]] = modes.T
    return padded


def build_records(g, nodes, target, goal, path_id, spacing=.03):
    records, last = [], None
    for i, node in enumerate(nodes):
        if last is not None and np.linalg.norm(node - last) < spacing * g.scale and i != len(nodes) - 1:
            continue
        if last is not None and np.linalg.norm(node - last) < 1e-10:
            continue
        last = node
        modes = g.modes(node)
        direction, norm = teacher_direction(node, target, modes)
        if norm <= 1e-8:
            continue
        _, width, _, status, calls = oracle(g, node, target, modes)
        records.append(dict(node=node.copy(), goal=goal, modes=pad_modes(g, node),
                            direction=direction, width=width / g.step_width,
                            width_valid=float(status == "accepted"), path_id=path_id,
                            teacher_calls=calls))
    return records


def pack(records):
    if not records:
        raise ValueError("No training records")
    return {key: np.asarray([r[key] for r in records]) for key in records[0]}


def tensors(data, scale):
    nodes = data["node"]
    inputs = np.column_stack([nodes.reshape(len(nodes), -1) / scale, data["goal"]]).astype(np.float32)
    return dict(inputs=torch.from_numpy(inputs), modes=torch.tensor(data["modes"], dtype=torch.float32),
                direction=torch.tensor(data["direction"], dtype=torch.float32),
                width=torch.tensor(data["width"], dtype=torch.float32),
                width_valid=torch.tensor(data["width_valid"], dtype=torch.float32))


def train_fixed(model, data, scale, steps, seed, lr=1e-3, batch=64, width_weight=.2, epochs=None):
    """Fresh Adam for each adaptation arm; identical sampled indices by seed."""
    t = tensors(data, scale)
    if epochs is not None:
        steps = epochs * max(1, int(np.ceil(len(t["inputs"]) / batch)))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    logs, losses, start = [], [], time.monotonic()
    for update in range(1, steps + 1):
        indices = torch.randint(len(t["inputs"]), (batch,), generator=generator)
        x, modes, labels = t["inputs"][indices], t["modes"][indices], t["direction"][indices]
        if getattr(model, "predicts_width", False):
            prediction, width = model(x, modes, labels)
            valid = t["width_valid"][indices]
            width_loss = (((width - t["width"][indices]) ** 2) * valid).sum() / valid.sum().clamp(min=1)
        else:
            projector = torch.bmm(modes.transpose(1, 2), modes)
            prediction = model(x, projector)
            width_loss = prediction.new_zeros(())
        direction_loss = (prediction - labels).square().sum(-1).mean()
        loss = direction_loss + width_weight * width_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append((float(direction_loss.detach()), float(width_loss.detach())))
        if update % 100 == 0 or update == steps:
            avg = np.mean(losses, axis=0)
            logs.append(dict(update=update, direction_loss=avg[0], width_loss=avg[1], elapsed=time.monotonic() - start))
            losses = []
    model.eval()
    return logs


@torch.no_grad()
def action(model, g, node, goal, fixed_width=None):
    x = torch.tensor(np.append(node.reshape(-1) / g.scale, goal)[None], dtype=torch.float32)
    modes = torch.from_numpy(pad_modes(g, node))[None]
    if getattr(model, "predicts_width", False):
        direction, fraction = model(x, modes)
        width = float(fraction) * g.step_width
    else:
        direction = model(x, torch.bmm(modes.transpose(1, 2), modes))
        width = g.step_width
    return direction[0].numpy(), width if fixed_width is None else fixed_width


def path_rollout(g, initial, goal, model=None, target=None, max_steps=400, backtrack=True, fixed_width=None):
    if (model is None) == (target is None):
        raise ValueError("Specify exactly one controller")
    node, nodes, reason, calls, smallest = initial.copy(), [initial.copy()], "timeout", 0, g.step_width
    for _ in range(max_steps):
        if g.metrics(node, goal)["success"]:
            reason = "success"
            break
        if target is not None:
            _, width, candidate, reason, attempts = oracle(g, node, target)
        else:
            direction, requested = action(model, g, node, goal, fixed_width)
            candidate, width, attempts, reason = propose(g, node, direction, requested, backtrack)
        calls += attempts
        if not width or np.linalg.norm(candidate - node) < 1e-12:
            break
        smallest = min(smallest, width)
        node = candidate
        nodes.append(node.copy())
        if len(nodes) > 32 and np.linalg.norm(node - nodes[-25]) < 1e-6:
            reason = "cycle"
            break
    metrics = g.metrics(node, goal)
    if metrics["success"]:
        reason = "success"
    elif reason == "accepted":
        reason = "timeout"
    return np.array(nodes), dict(**metrics, steps=len(nodes) - 1, reason=reason,
                                projection_calls=calls, min_width=smallest)


def prepare(root, support=32, validation=12, test=32):
    root.mkdir(parents=True, exist_ok=False)
    g = Geometry(BASE / "model/seven_bar_linkage")
    source = generate_starts(g, 32, np.random.default_rng(44001), 20, 160)
    support_nodes = generate_starts(g, support, np.random.default_rng(44002), 20, 160, source)
    val = generate_starts(g, validation, np.random.default_rng(44003), 20, 160, [*source, *support_nodes])
    test_nodes = generate_starts(g, test, np.random.default_rng(44004), 20, 160, [*source, *support_nodes, *val])
    np.savez_compressed(root / "splits.npz", source=source, support=support_nodes, validation=val, test=test_nodes)
    targets = load_targets(g)
    targets[4.] = target_d4()
    records, rows = [], []
    (root / "teacher_paths").mkdir()
    for split, starts, goals in (("source", source, (2., 3.)), ("support", support_nodes, (4.,))):
        split_records = []
        for i, initial in enumerate(starts):
            for goal in goals:
                path, metrics = path_rollout(g, initial, goal, target=targets[goal])
                np.savez_compressed(root / "teacher_paths" / f"{split}_{i:03d}_d{goal:g}.npz", nodes=path)
                # Both source goals at source states; support states only receive D4.
                for label_goal in goals:
                    split_records.extend(build_records(g, path, targets[label_goal], label_goal, i))
                rows.append(dict(split=split, path_id=i, goal=goal, **metrics))
            print(f"prepare {split} {i+1}/{len(starts)} records={len(split_records)}", flush=True)
        np.savez_compressed(root / f"{split}_data.npz", **pack(split_records))
    write_csv(root / "teacher_rollouts.csv", rows)
    (root / "design.json").write_text(json.dumps(dict(
        support=support, validation=validation, test=test, source=32,
        source_goals=[2,3], target_goal=4, seeds=[44001,44002,44003,44004],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sampling="both D2/D3 labels at every source trajectory state",
        support_sampling="nested random start prefixes, includes failed teacher paths",
        collision_policy="backtrack until feasible; independent of target coordinates"), indent=2), encoding="utf-8")


def evaluate_model(g, model, starts, goal, max_steps=400, **kwargs):
    rows = []
    for i, initial in enumerate(starts):
        _, result = path_rollout(g, initial, goal, model=model, max_steps=max_steps, **kwargs)
        rows.append(dict(start_id=i, goal=goal, **result))
    return rows


def study(root, architecture, seeds, budgets, updates, source_updates, evaluation_split="test"):
    g = Geometry(BASE / "model/seven_bar_linkage")
    splits = np.load(root / "splits.npz")
    source = dict(np.load(root / "source_data.npz"))
    support = dict(np.load(root / "support_data.npz"))
    output = root / f"{architecture}_u{updates}_src{source_updates}_{evaluation_split}"
    output.mkdir(exist_ok=True)
    constructors = {"modal": lambda: ModalStudent(21, 96), "direct": lambda: DirectionNet(96,21)}
    for seed in seeds:
        torch.manual_seed(seed)
        model = constructors[architecture]()
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        pretrain_file = output / f"source_seed{seed}.pt"
        if pretrain_file.exists():
            model.load_state_dict(torch.load(pretrain_file, weights_only=True)["model"])
        else:
            history = train_fixed(model, source, g.scale, source_updates, seed + 100)
            torch.save(dict(model=model.state_dict(), architecture=architecture, seed=seed), pretrain_file)
            write_csv(output / f"source_seed{seed}_learning.csv", history)
        pretrained = {k:v.clone() for k,v in model.state_dict().items()}
        for budget in budgets:
            mask = support["path_id"] < budget
            data = {k:v[mask] for k,v in support.items()}
            for arm, weights in (("scratch", initial), ("pretrained", pretrained)):
                prefix = f"seed{seed}_{arm}_n{budget:02d}"
                if (output / f"{prefix}_evaluation.csv").exists():
                    continue
                model.load_state_dict(weights)
                if budget:
                    history = train_fixed(model, data, g.scale, updates, seed + 200, lr=3e-4)
                    write_csv(output / f"{prefix}_learning.csv", history)
                torch.save(dict(model=model.state_dict(), architecture=architecture, seed=seed,
                                budget=budget, arm=arm, updates=updates if budget else 0), output / f"{prefix}.pt")
                # Also check retention of source goals after D4 adaptation.
                rows = []
                for goal in (2., 3., 4.):
                    rows.extend(evaluate_model(g, model, splits[evaluation_split], goal))
                rows = [dict(seed=seed, arm=arm, budget=budget, labels=int(mask.sum()), updates=updates if budget else 0, **r) for r in rows]
                write_csv(output / f"{prefix}_evaluation.csv", rows)
                rates = {goal:np.mean([r['success'] for r in rows if r['goal']==goal]) for goal in (2.,3.,4.)}
                print(f"{architecture} {prefix} success={rates}", flush=True)


def diagnostic(root):
    g = Geometry(BASE / "model/seven_bar_linkage")
    starts = np.load(root / "splits.npz")["validation"]
    saved = torch.load(BASE / "result/experiments/supervised_seven_d23_20260914_extended_v1/checkpoint_final.pt", weights_only=True)
    model = DirectionNet(saved["hidden"],21)
    model.load_state_dict(saved["model"])
    rows = []
    for width in (.1,.05,.02):
        for backtrack in (False,True):
            for goal in (2.,3.):
                results = evaluate_model(g, model, starts, goal, fixed_width=width, backtrack=backtrack)
                rows.extend([dict(width=width, backtrack=backtrack, **r) for r in results])
                print("diagnostic",width,backtrack,goal,np.mean([r['success'] for r in results]),flush=True)
            write_csv(root / "integrator_validation.csv", rows)


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","study","diagnostic"))
    parser.add_argument("root",type=Path)
    parser.add_argument("--architecture",choices=("direct","modal"),default="modal")
    parser.add_argument("--seeds",type=int,nargs="+",default=[5501,5502,5503,5504,5505])
    parser.add_argument("--budgets",type=int,nargs="+",default=[0,1,2,4,8,16,32])
    parser.add_argument("--updates",type=int,default=2000)
    parser.add_argument("--source-updates",type=int,default=10000)
    parser.add_argument("--evaluation-split",choices=("validation","test"),default="test")
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.root)
    elif args.command=="diagnostic": diagnostic(args.root)
    else: study(args.root,args.architecture,args.seeds,args.budgets,args.updates,args.source_updates,args.evaluation_split)


if __name__=="__main__": main()
