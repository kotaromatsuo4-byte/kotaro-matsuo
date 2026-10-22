"""Frozen-data critic fitting diagnostic; never modifies a training checkpoint."""
from pathlib import Path
import json
import argparse
from main import task_config
import numpy as np
import torch
import RL_mixed_goals as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--update", type=int, default=128)
    parser.add_argument("--fit-updates", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    design = json.loads((args.run / "design.json").read_text())
    model, agent = m.make_model_agent(design["configs"]["five_d2"])
    ck = torch.load(args.run / "checkpoints" / f"update{args.update:05d}.pt", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    rng = np.random.default_rng()
    rng.bit_generator.state = ck["schedule_rng"]
    goals = [t for t in m.TASKS for _ in range(4)]
    rng.shuffle(goals)
    torch.set_rng_state(ck["torch_rng"])
    original_action = agent.get_action
    records = []
    def capture(state, modes, **kwargs):
        records.append((state.copy(), modes.T.copy()))
        return original_action(state, modes, **kwargs)
    agent.get_action = capture
    samples = []
    count = dict.fromkeys(m.TASKS, 0)
    for task in goals:
        count[task] += 1
        env = m.GoalConditionedEnv(design["configs"][task])
        records.clear()
        with torch.no_grad():
            summary, _ = m.rollout(env, agent, training=True)
        transitions = agent.finished_episodes[-1]
        assert len(records) == len(transitions)
        returns = []
        g = 0.
        for tr in reversed(transitions):
            g = tr.reward + agent.gamma*g
            returns.append(g)
        returns.reverse()
        for i in range(0, len(records), 4):
            state, modes = records[i]
            samples.append((state, modes, returns[i], task, count[task] == 4))
    states = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
    max_modes = max(len(s[1]) for s in samples)
    modes = torch.zeros(len(samples), max_modes, model.coordinate_size)
    mask = torch.zeros(len(samples), max_modes, 1)
    for i, s in enumerate(samples):
        modes[i, :len(s[1])] = torch.tensor(s[1], dtype=torch.float32)
        mask[i, :len(s[1])] = 1
    targets = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    train = torch.tensor([i for i,s in enumerate(samples) if not s[4]])
    def predict(indices):
        st = states[indices]
        features = model.critic_mode(torch.cat((st[:,None,:].expand(-1,max_modes,-1), modes[indices]),dim=-1))
        pooled = (features*mask[indices]).sum(1)/mask[indices].sum(1).clamp_min(1)
        return model.critic(torch.cat((model.critic_state(st),pooled),dim=1)).squeeze(-1)
    # Check batched implementation against the existing single-state method.
    with torch.no_grad():
        for i in range(min(20,len(samples))):
            ref=model.value(states[i:i+1],torch.tensor(samples[i][1],dtype=torch.float32))
            assert torch.allclose(predict(torch.tensor([i]))[0],ref,atol=2e-6)
    torch.manual_seed(761234)
    optimizer = torch.optim.Adam(agent.critic_parameter_list, lr=design["critic_lr"])
    output=args.run / "advantage_diagnostic" / "critic_fit"
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for step in range(args.fit_updates+1):
        if step % 100 == 0 or step == args.fit_updates:
            with torch.no_grad():
                for task in m.TASKS:
                    for heldout in (False,True):
                        idx=torch.tensor([i for i,s in enumerate(samples) if s[3]==task and s[4]==heldout])
                        pred=predict(idx); truth=targets[idx]
                        rows.append(dict(fit_update=step,task=task,split="heldout_episode" if heldout else "train",
                            samples=len(idx),mse=float(((pred-truth)**2).mean()),target_mean=float(truth.mean()),value_mean=float(pred.mean())))
            m.write_csv(output/"metrics.csv",rows)
            print(f"Critic fit {step}/{args.fit_updates}",flush=True)
        if step==args.fit_updates:break
        idx=train[torch.randint(len(train),(64,))]
        loss=.5*((predict(idx)-targets[idx])**2).mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_parameter_list,1.)
        optimizer.step()
    (output/"design.json").write_text(json.dumps(dict(checkpoint_update=args.update,
        fit_updates=args.fit_updates,learning_rate=design["critic_lr"],train_episodes_per_goal=3,
        heldout_episodes_per_goal=1,state_stride=4,optimizer="fresh Adam",interpretation="Frozen-data supervised fit, not RL improvement or generalization proof"),indent=2))
    print(json.dumps(rows[-4:],indent=2),flush=True)


if __name__=="__main__":main()
