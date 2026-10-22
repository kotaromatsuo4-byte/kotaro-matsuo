# -*- coding: cp932 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os

#incompatibility
def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i]**2)
    return g

#mode
def index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])

def dedx(node, num_node, member, num_member, index_free):
    dedx_all = np.zeros((num_member, num_node * 3))
    node_diff = node[member[:, 0]] - node[member[:, 1]]

    for i in range(num_member):
        i0, i1 = member[i]
        dedx_all[i, i0*3:(i0+1)*3] = node_diff[i]
        dedx_all[i, i1*3:(i1+1)*3] = -node_diff[i]

    return dedx_all[:, index_free]

def mode(index_free, dedx, num_node, tol=1e-8):
    _, sigma, vh = np.linalg.svd(dedx)
    rank = np.count_nonzero(sigma > tol)
    dof = dedx.shape[1] - rank

    displace_mode = np.zeros((num_node * 3, dof))
    displace_mode[index_free, :] = vh[rank:].T
    return displace_mode

#corrector
def corrector(node, num_node, member,num_member, L0, index_free, beta0=1.0, n_iter=20, tau=0.5):
    g = incompatibility(node, member, L0)
    gammma = dedx(node, num_node, member, num_member, index_free)
    dd_free = -np.linalg.pinv(gammma) @ g
    dd = np.zeros(num_node * 3)
    dd[index_free] = dd_free

    # line search
    beta = beta0
    for _ in range(n_iter):
        node_new = node + beta * dd.reshape(-1,3)

        g_new = incompatibility(node_new, member, L0)

        if np.linalg.norm(g_new) < (1 - 1e-4 * beta) * np.linalg.norm(g):
            break
        beta *= tau

    return beta * dd.reshape(-1,3)

#flatness
def compute_flatness_z(node):
    z_values = node[:, 2]
    z_max = np.max(z_values)
    z_min = np.min(z_values)
    flatness_z= z_max - z_min
    return flatness_z

# Env
class OrigamiEnv:
    def __init__(self, node, member, fix, max_steps=30):
        self.node0 = node.copy()    # initial shape (reference)
        self.node = node.copy()     # current shape
        self.member = member
        self.fix = fix
        self.num_node = node.shape[0]
        self.num_member = member.shape[0]
        self.max_steps = max_steps

        #initial length of each member
        self.L0=np.array([
        np.linalg.norm(node[i0] - node[i1])
        for i0, i1 in member
        ])
        
        self.free = index_free(self.num_node, self.fix)

        # initial modes
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        self.modes = mode(self.free, d, self.num_node)
        self.num_mode = self.modes.shape[1]
        
        self.initial_num_mode = self.num_mode

        self.step_count = 0

        self.prev_flatness = None

    def reset(self):
        self.node = self.node0.copy()
        self.step_count = 0

        self.prev_flatness = compute_flatness_z(self.node)

        return self.get_state()

    def step(self, action):

        # mode update
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        modes = mode(self.free, d, self.num_node)
        self.num_mode = modes.shape[1]

        # action
        k = modes.shape[1]
        action_used = action[:k]
        disp = modes @ action_used
        disp = disp.reshape(self.num_node, 3)

        # trial
        node_trial = self.node + disp
        
        # corrector
        node_tmp = node_trial.copy()

        for _ in range(10):
            dc = corrector(node_tmp,self.num_node,self.member,self.num_member,self.L0,self.free)

            node_tmp += dc

            if np.linalg.norm(dc) < 1e-6:
                break
        
        # node update
        self.node = node_tmp

        # 評価
        flatness = compute_flatness_z(self.node)

        if self.prev_flatness > 1e-8:
            delta = self.prev_flatness - flatness
            rel_improve = delta / self.prev_flatness
        else:
            rel_improve = 0.0

        b = 1.0
        c = 10.0
        reward =  - b * flatness + c * rel_improve

        self.prev_flatness = flatness

        self.step_count += 1
        done = False

        # 平坦化�E劁E
        if flatness < 1e-3:
            reward += 100
            done = True
            

        # strain constraint 
        strain = self.compute_strain()
        if strain > 10:
            reward += -100.0
            done = True

        # step limit
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state(), reward, done, {}

    def get_state(self):
        return self.node.flatten()
    
    def get_modes(self):
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        modes = mode(self.free, d, self.num_node)
        return modes

    def compute_strain(self):
        strain = 0.0
        for i in range(self.num_member):
            n0, n1 = self.member[i]
            l0 = np.linalg.norm(self.node0[n0] - self.node0[n1])
            l = np.linalg.norm(self.node[n0] - self.node[n1])
            strain += (l - l0) ** 2
        return strain
    

# Policy NN

class Policy(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.state_size = state_size

        # One row represents one current mode. This lets the same policy handle
        # a changing number of modes along the folding path.
        self.fc1 = nn.Linear(state_size * 2, 128)
        self.fc2 = nn.Linear(128, 128)

        self.fc_mu = nn.Linear(128, 1)
        self.fc_logstd = nn.Linear(128, 1)

    def forward(self, state,modes):
        """
        state: (1,3n)
        modes: (k,3n)
        """

        k = modes.shape[0]
        if k == 0:
            empty = state.new_empty((0,))
            return empty, empty

        # state is repeated for each mode, so the policy can score each mode independently.
        state_expand = state.repeat(k, 1)  # (k,3n)

        x = torch.cat([state_expand, modes], dim=1)  # (k,6n)
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))

        mu = self.fc_mu(h).squeeze(-1)         # (k,)
        log_std = torch.clamp(self.fc_logstd(h).squeeze(-1), -5, 1)
        std = torch.exp(log_std)

        return mu, std


# Agent

class Agent:
    def __init__(self, state_size):
        self.gamma = 0.99
        self.lr = 1e-3

        self.memory = []
        self.pi = Policy(state_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def sample_action(self, state, modes):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)

        # (3n,k) -> (k,3n). The policy scores each current mode independently.
        modes = torch.tensor(modes.T, dtype=torch.float32)

        mu, std = self.pi(state, modes)

        if mu.numel() == 0:
            return np.zeros(0, dtype=np.float32), torch.tensor(0.0)

        dist = Normal(mu, std)
        raw_action = dist.sample()           # (k,)
        log_prob = dist.log_prob(raw_action).sum()

        # Bound each modal coefficient while keeping it tied to raw_action.
        action = 0.5 * torch.tanh(raw_action)

        return action.detach().numpy(), log_prob

    def get_action(self, state, modes):
        return self.sample_action(state, modes)

    def add(self, reward, log_prob):
        self.memory.append((reward, log_prob))

    def update(self):
        G = 0
        returns = []

        for reward, _ in reversed(self.memory):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for (_, log_prob), G in zip(self.memory, returns):
            loss += -log_prob * G

        loss = loss / len(self.memory)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []

# input
node=np.loadtxt("node_initial.dat",delimiter=',',dtype=float)
member=np.loadtxt("member.dat",delimiter=',',dtype=int)
fix=np.loadtxt("fix.dat",delimiter=',',dtype=int)

# agent and env

env = OrigamiEnv(node, member, fix)

state_size = env.num_node * 3
action_size = env.num_mode

agent = Agent(state_size)

reward_history = []

num_episodes = 5000
batch_size = 100

reward_history = []

# save history
final_dir = "history_final_5000"
best_dir = "history_best_5000"

os.makedirs(final_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

# best reward tracking
best_reward = -np.inf
best_history = None
best_episode = -1

reward_history = []

# training loop

try:
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        node_history = []

        while not done:
            modes = env.get_modes()
            action, log_prob = agent.get_action(state, modes)
            next_state, reward, done, _ = env.step(action)

            agent.add(reward, log_prob)

            state = next_state
            total_reward += reward

            node_history.append(env.node.copy())

        reward_history.append(total_reward)

        # batch update
        if (episode + 1) % batch_size == 0:
            agent.update()

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward {total_reward:.3f}")

        # best reward tracking and history saving
        if total_reward > best_reward:
            best_reward = total_reward
            best_history = [n.copy() for n in node_history]
            best_episode = episode

            episode_dir = os.path.join(best_dir, f"episode_{episode}")
            os.makedirs(episode_dir, exist_ok=True)

            for i, node_step in enumerate(best_history):
                filepath = os.path.join(episode_dir, f"{i+1}.dat")
                np.savetxt(filepath, node_step, fmt="%.6f")

except KeyboardInterrupt:
    print("\n=== Training Interrupted (Ctrl+C) ===")
    
finally:
    print("Saving results...")

    # final episode history
    if 'node_history' in locals():
        final_episode_dir = os.path.join(final_dir, f"episode_{episode}")
        os.makedirs(final_episode_dir, exist_ok=True)

        for i, node_step in enumerate(node_history):
            filepath = os.path.join(final_episode_dir, f"{i+1}.dat")
            np.savetxt(filepath, node_step, fmt="%.6f")

    # best episode history
    if best_history is not None:
        best_summary_dir = os.path.join(best_dir, "best_overall")
        os.makedirs(best_summary_dir, exist_ok=True)

        for i, node_step in enumerate(best_history):
            filepath = os.path.join(best_summary_dir, f"{i+1}.dat")
            np.savetxt(filepath, node_step, fmt="%.6f")

        print("===================================")
        print(f"Best Episode : {best_episode}")
        print(f"Best Reward  : {best_reward:.3f}")
        print("===================================")


    # learning curve
    if len(reward_history) > 0:
        plt.figure()

        plt.plot(reward_history, color='black',alpha=0.3, label="Raw Reward")

        # moving average
        def moving_average(data, window):
            if len(data) < window:
                return np.array([])
            return np.convolve(data, np.ones(window)/window, mode='valid')
        learning_crv = moving_average(reward_history, batch_size)

        if len(learning_crv) > 0:
            x = np.arange(len(learning_crv)) + batch_size - 1
            plt.plot(x, learning_crv, color='black', linewidth=2, label="Moving Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve")
        plt.grid()
        plt.legend()
        np.savetxt("reward_history_5000.csv", np.array(reward_history), delimiter=",")
        plt.savefig("learning_curve_5000.png", dpi=200, bbox_inches="tight")
        plt.close()

