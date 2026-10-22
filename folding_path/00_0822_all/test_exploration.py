"""Small mathematical checks for the optional exploration controller."""
import numpy as np
import torch
from torch.distributions import Normal
from RL_mixed_goals import exploration_controls
from agent import ActorCriticAgent
from policy import VariableModeActorCritic, canonical_physical_direction


def main():
    torch.set_num_threads(1)
    design = {"adaptive_exploration": True, "tasks": ["a", "b"]}
    rows = [{"task": "a", "strict_success": int(i < 8)} for i in range(32)]
    assert exploration_controls(design, rows) == {"a": (0.25, 0.0), "b": (0.0, 1.0)}
    assert exploration_controls(design, rows + [{"task": "a", "strict_success": 0}] * 32)["a"] == (0.0, 1.0)
    torch.manual_seed(54)
    model = VariableModeActorCritic(15, goal_size=1)
    agent = ActorCriticAgent(model, .1)
    state = np.zeros(16, dtype=np.float32)
    state[-1] = 2
    modes = np.eye(15, dtype=np.float32)[:, :2]
    st, mt = torch.tensor(state[None]), torch.tensor(modes.T)
    for strength in (0.0, 0.21875, 1.0):
        agent.exploration_strength = strength
        torch.manual_seed(123)
        action = agent.get_action(state, modes)
        torch.manual_seed(123)
        mean, variance, features = model.modal_distribution_parameters(st, mt)
        if strength:
            mean, variance = mean / (1 + strength), variance + strength
        dist = Normal(mean, variance.sqrt())
        sample = dist.sample()
        _, direction = canonical_physical_direction(sample / sample.norm(), mt)
        wm, wv = model.width_distribution_parameters(st, direction, features)
        if strength:
            wm, wv = wm / (1 + strength), wv + strength
        wd = Normal(wm, wv.sqrt())
        width = wd.sample()
        assert torch.equal(action.log_probability, dist.log_prob(sample).sum() + wd.log_prob(width).sum())
        assert np.array_equal(action.physical_direction, direction.detach().numpy())
        assert action.requested_width == float((.1 * torch.tanh(width)).detach())
        assert abs(action.requested_width) <= .1
        model.zero_grad()
        (-action.log_probability + action.value).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS: controller history, exact sampling/log-density consistency, physical step bound, finite gradients")


if __name__ == "__main__":
    main()
