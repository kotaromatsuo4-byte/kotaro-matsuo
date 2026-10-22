import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os
import shutil

#非適合量
def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i]**2)
    return g

#モードの計算
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

#修正子計算
def corrector(node, num_node, member,num_member, L0, index_free, beta0=1.0, n_iter=20, tau=0.5):
    # 非適合
    g = incompatibility(node, member, L0)

    # ヤコビアン
    gammma = dedx(node, num_node, member, num_member, index_free)

    # Moore-Penrose
    dd_free = -np.linalg.pinv(gammma) @ g

    # full DOFに戻す
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

#平坦性評価
def compute_flatness_plane(node):
    # 重心
    centroid = np.mean(node, axis=0)

    # 中心化
    X = node - centroid

    # SVD
    _, _, vh = np.linalg.svd(X)

    # 最小特異値方向 = 法線
    normal = vh[-1]

    # 各点の平面からの距離
    distances = np.abs(X @ normal)

    return np.mean(distances)   # or np.max(distances)

# Env
class OrigamiEnv:
    def __init__(self, node, member, fix, max_steps=20):
        self.node0 = node.copy()    # 初期形状
        self.node = node.copy()     # 現在形状
        self.member = member
        self.fix = fix
        self.num_node = node.shape[0]
        self.num_member = member.shape[0]
        self.max_steps = max_steps

        #初期長さ
        self.L0=np.array([
        np.linalg.norm(node[i0] - node[i1])
        for i0, i1 in member
        ])
        
        self.free = index_free(self.num_node, self.fix)

        # 初期モード（actionサイズ決定用）
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        self.modes = mode(self.free, d, self.num_node)
        self.num_mode = self.modes.shape[1]
        
        self.initial_num_mode = self.num_mode

        self.step_count = 0

        self.prev_flatness = None

    def reset(self):
        self.node = self.node0.copy()
        self.step_count = 0

        self.prev_flatness = compute_flatness_plane(self.node)

        return self.get_state()

    def step(self, action):

        # モードを毎step更新
        d = dedx(self.node, self.num_node, self.member, self.num_member, self.free)
        modes = mode(self.free, d, self.num_node)
        self.num_mode = modes.shape[1]
        # 特異点検出 検出されたときの報酬をpenaltyで-10として、episodeを終了する。
        if self.num_mode != self.initial_num_mode:
            reward = -100.0
            done = True
            return self.get_state(), reward, done, {}
        
        # action正規化
        norm = np.linalg.norm(action)
        if norm > 1e-8:
            action = 0.1 * action / norm
        else:
            action = np.zeros_like(action)

        # 変位
        k = modes.shape[1]
        action_used = action[:k]
        disp = modes @ action_used
        disp = disp.reshape(self.num_node, 3)

        # trial
        node_trial = self.node + disp

        """"
        # おおざっぱに平坦性評価→方向制御（振動防止）
        flatness = compute_flatness_plane(node_trial)
        if self.prev_flatness > flatness:
            pass    # 改善している場合は同じ方向に進む
        else:  
            node_trial -= 2 * disp  # 改善していない場合は逆方向に進む  
        """
        
        # corrector
        node_tmp = node_trial.copy()

        for _ in range(10):
            dc = corrector(
                node_tmp,
                self.num_node,
                self.member,
                self.num_member,
                self.L0,
                self.free
            )

            node_tmp += dc

            if np.linalg.norm(dc) < 1e-6:
                break
        
        # 節点更新
        self.node = node_tmp

        # 評価
        strain = self.compute_strain()
        flatness = compute_flatness_plane(self.node)

        if self.prev_flatness > 1e-8:
            delta = self.prev_flatness - flatness
            rel_improve = delta / self.prev_flatness
        else:
            rel_improve = 0.0

        a = 1.0
        b = 1.0
        c = 1000.0
        #reward = -a * strain - b * flatness + c * rel_improve
        #reward = - b * flatness + c * rel_improve
        #reward = c * rel_improve
        reward = - b * flatness

        self.prev_flatness = flatness

        # 終了判定と報酬補正
        self.step_count += 1
        done = False

        # 平坦化成功
        if flatness < 1e-3:
            reward += 100
            done = True
            

        # ひずみ破綻
        if strain > 10:
            reward += -100.0
            done = True

        # 最大ステップ
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state(), reward, done, {}

    def get_state(self):
        return self.node.flatten()

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
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc_mu = nn.Linear(128, action_size)
        self.fc_logstd = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_std = torch.clamp(self.fc_logstd(x), -5, 2)
        std = torch.exp(log_std)
        return mu, std


# Agent

class Agent:
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.lr = 1e-3

        self.memory = []
        self.pi = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)

        mu, std = self.pi(state)
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        action = torch.tanh(action) * 1.0

        return action[0].detach().numpy(), log_prob

    def add(self, reward, log_prob):
        self.memory.append((reward, log_prob))

    def update(self):
        G = 0
        loss = 0

        for reward, log_prob in reversed(self.memory):
            G = reward + self.gamma * G
        for reward, log_prob in self.memory:
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []


# モデル
node=np.loadtxt("node_initial.dat",delimiter=',',dtype=float)
member=np.loadtxt("member.dat",delimiter=',',dtype=int)
fix=np.loadtxt("fix.dat",delimiter=',',dtype=int)
"""
node = np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0],
    [0.5,0.5,0.7071]
])
member = np.array([
    [0,1],[1,2],[2,3],[3,0],[2,0]
])

fix = np.array([
    [0,0],[0,1],[0,2],
    [1,0],[1,1],[1,2],
    [2,0],[2,1],[2,2]
])
"""
# 学習

env = OrigamiEnv(node, member, fix)

state_size = env.num_node * 3
action_size = env.num_mode

agent = Agent(state_size, action_size)

reward_history = []

num_episodes = 10000

# 保存用
final_dir = "history_final"
best_dir = "history_best"

# 既存フォルダ削除
if os.path.exists(final_dir):
    shutil.rmtree(final_dir)
if os.path.exists(best_dir):
    shutil.rmtree(best_dir)

# 再作成
os.makedirs(final_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

# best管理
best_reward = -np.inf
best_history = None
best_episode = -1

reward_history = []

# 学習ループ

try:
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        node_history = []

        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.add(reward, log_prob)

            state = next_state
            total_reward += reward

            node_history.append(env.node.copy())

        agent.update()
        reward_history.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward {total_reward:.3f}")

        # best更新
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

    # 最終エピソード保存（途中でもOK）
    if 'node_history' in locals():
        final_episode_dir = os.path.join(final_dir, f"episode_{episode}")
        os.makedirs(final_episode_dir, exist_ok=True)

        for i, node_step in enumerate(node_history):
            filepath = os.path.join(final_episode_dir, f"{i+1}.dat")
            np.savetxt(filepath, node_step, fmt="%.6f")

    # best保存
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

    # learning curve表示（ここが重要）
    if len(reward_history) > 0:
        plt.plot(reward_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve")
        plt.grid()
        plt.show()