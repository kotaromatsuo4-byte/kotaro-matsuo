"""Additional supervised critic minibatches from the current on-policy batch only."""
import numpy as np
import torch


def prepare(agent):
    observations = agent.replay_observations
    returns = []
    for episode in agent.finished_episodes:
        g = 0.
        values = []
        for tr in reversed(episode):
            g = tr.reward + agent.gamma*g
            values.append(g)
        returns.extend(reversed(values))
    assert len(observations) == len(returns)
    states = torch.tensor(np.stack([x[0] for x in observations]), dtype=torch.float32)
    n = max(len(x[1]) for x in observations)
    modes = torch.zeros(len(returns), n, agent.model.coordinate_size)
    mask = torch.zeros(len(returns), n, 1)
    for i, (_, mo) in enumerate(observations):
        modes[i,:len(mo)] = torch.tensor(mo, dtype=torch.float32)
        mask[i,:len(mo)] = 1
    agent.replay_observations.clear()
    return states, modes, mask, torch.tensor(returns,dtype=torch.float32)


def fit(agent, data, count, generator):
    states, modes, mask, targets = data
    model = agent.model
    for _ in range(count):
        idx = torch.randint(len(targets), (64,), generator=generator)
        st = states[idx]
        features = model.critic_mode(torch.cat((st[:,None,:].expand(-1,modes.shape[1],-1),modes[idx]),dim=-1))
        pooled = (features*mask[idx]).sum(1)/mask[idx].sum(1).clamp_min(1)
        pred = model.critic(torch.cat((model.critic_state(st),pooled),dim=1)).squeeze(-1)
        loss = agent.value_coefficient*((pred-targets[idx])**2).mean()
        agent.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_parameter_list,agent.critic_gradient_clip)
        agent.critic_optimizer.step()
