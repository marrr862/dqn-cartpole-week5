"""
Все упражнения из dqn_lab.py (1-7) и dqn_lab_sb3.py (2-4).
Запуск:  python experiments.py      (параллельно на всех ядрах CPU)
Результаты: results.json, графики exp_*.png
"""
import json
import random
import time
from multiprocessing import Pool

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import dqn_lab
from dqn_lab import DQNAgent, evaluate, linear_epsilon

SEEDS = [0, 1, 2]


# ---------- Exercise 6: Double DQN ----------
class DoubleDQNAgent(DQNAgent):
    def learn(self):
        s, a, r, s_next, term = self.buffer.sample(self.batch_size)
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.q_net(s_next).argmax(dim=1, keepdim=True)          # choose: online w
            q_next = self.target_net(s_next).gather(1, a_star).squeeze(1)    # evaluate: target w-
            y = r + self.gamma * (1 - term) * q_next
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1
        if self.n_updates % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


# ---------- general training loop (copy of train_dqn with switches) ----------
def run_own(cfg):
    torch.set_num_threads(1)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dqn_lab.SEED = seed
    env = gym.make("CartPole-v1")
    cls = DoubleDQNAgent if cfg.get("double") else DQNAgent
    agent = cls(4, 2, **cfg.get("agent", {}))
    learning_starts = cfg.get("learning_starts", 1_000)
    decay = cfg.get("decay_steps", 5_000)
    trunc_bug = cfg.get("trunc_bug", False)

    returns, qmeans, step, t0 = [], [], 0, time.perf_counter()
    for ep in range(300):
        s, _ = env.reset(seed=seed + ep)
        ep_ret, done = 0.0, False
        while not done:
            a = agent.act(s, linear_epsilon(step, decay_steps=decay))
            s_next, r, terminated, truncated, _ = env.step(a)
            agent.buffer.push(s, a, r, s_next, (terminated or truncated) if trunc_bug else terminated)
            done = terminated or truncated
            s, ep_ret, step = s_next, ep_ret + r, step + 1
            if len(agent.buffer) >= learning_starts:
                agent.learn()
        returns.append(ep_ret)
        # mean predicted Q(s,a) on a random batch (exercise 6)
        if len(agent.buffer) >= 256 and ep % 10 == 9:
            sb, ab, *_ = agent.buffer.sample(min(256, len(agent.buffer)))
            with torch.no_grad():
                qmeans.append(agent.q_net(sb).gather(1, ab.unsqueeze(1)).mean().item())
    train_time = time.perf_counter() - t0

    mean, std = evaluate(env, lambda s: agent.act(s, 0.0))

    # Q along one greedy episode (exercise 3): max_a Q(s_t, a) as a function of t
    s, _ = env.reset(seed=12_345)
    q_traj, done = [], False
    while not done:
        with torch.no_grad():
            q = agent.q_net(torch.as_tensor(s, dtype=torch.float32))
        q_traj.append(q.max().item())
        s, _, te, tr, _ = env.step(int(q.argmax()))
        done = te or tr
    env.close()
    return dict(cfg=cfg, returns=returns, eval_mean=mean, eval_std=std,
                last25=float(np.mean(returns[-25:])), qmeans=qmeans,
                q_traj=q_traj, time=train_time, steps=step)


# ---------- Stable-Baselines3 ----------
def run_sb3(cfg):
    torch.set_num_threads(1)
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    total = 40_000
    env = Monitor(gym.make("CartPole-v1"))
    eval_env = Monitor(gym.make("CartPole-v1"))
    model = DQN("MlpPolicy", env,
                policy_kwargs=dict(net_arch=cfg.get("net_arch", [128, 128])),
                learning_rate=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
                learning_starts=1_000, train_freq=1, gradient_steps=1,
                target_update_interval=cfg.get("target_update_interval", 200),
                exploration_initial_eps=1.0, exploration_final_eps=0.05,
                exploration_fraction=5_000 / total, seed=cfg["seed"], verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=total)
    t = time.perf_counter() - t0
    returns = env.get_episode_rewards()
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    return dict(cfg=cfg, returns=[float(x) for x in returns], eval_mean=float(mean),
                eval_std=float(std), last25=float(np.mean(returns[-25:])), time=t)


def run(cfg):
    return run_sb3(cfg) if cfg["lib"] == "sb3" else run_own(cfg)


def configs():
    c = []
    for s in SEEDS:
        c += [
            dict(lib="own", name="baseline", seed=s),
            dict(lib="own", name="ex1_no_target", seed=s, agent=dict(target_sync=1)),
            # learning_starts must be <= buffer size, otherwise learning never starts
            dict(lib="own", name="ex2_no_replay", seed=s, agent=dict(buffer_size=64),
                 learning_starts=64),
            dict(lib="own", name="ex3_trunc_bug", seed=s, trunc_bug=True),
            dict(lib="own", name="ex4_decay_1k", seed=s, decay_steps=1_000),
            dict(lib="own", name="ex4_decay_50k", seed=s, decay_steps=50_000),
            dict(lib="own", name="ex6_double", seed=s, double=True),
            dict(lib="sb3", name="sb3_baseline", seed=s),
            dict(lib="sb3", name="sb3_net16", seed=s, net_arch=[16]),
            dict(lib="sb3", name="sb3_net64", seed=s, net_arch=[64, 64]),
            dict(lib="sb3", name="sb3_net256", seed=s, net_arch=[256, 256]),
            dict(lib="sb3", name="sb3_target1", seed=s, target_update_interval=1),
        ]
    return c


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, "valid") if len(x) >= w else x


def plot_group(res, names, title, fname):
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        for r in [r for r in res if r["cfg"]["name"] == n]:
            ax.plot(smooth(r["returns"]), label=f"seed {r['cfg']['seed']}")
        ax.axhline(500, color="k", ls="--", lw=0.8)
        ax.set_title(n); ax.set_xlabel("Episode"); ax.grid(alpha=0.3); ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Return (moving avg 20)")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close(fig)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with Pool() as p:
        res = p.map(run, configs())
    print(f"all runs: {time.perf_counter() - t0:.0f} s")
    json.dump(res, open("results.json", "w"))

    print(f"\n{'experiment':16s} {'seed':>4s} {'greedy eval':>14s} {'last25':>7s} {'time,s':>7s}")
    for r in sorted(res, key=lambda r: (r["cfg"]["name"], r["cfg"]["seed"])):
        print(f"{r['cfg']['name']:16s} {r['cfg']['seed']:4d} {r['eval_mean']:7.1f} +/-{r['eval_std']:5.1f}"
              f" {r['last25']:7.1f} {r['time']:7.1f}")

    print("\nSummary over seeds (greedy eval mean):")
    for n in dict.fromkeys(r["cfg"]["name"] for r in res):
        v = [r["eval_mean"] for r in res if r["cfg"]["name"] == n]
        t = [r["time"] for r in res if r["cfg"]["name"] == n]
        print(f"  {n:16s} {np.mean(v):6.1f}  (seeds: {', '.join(f'{x:.0f}' for x in v)})  time {np.mean(t):.0f}s")

    print("\nEx6: mean predicted Q(s,a) on a replay batch at the end (true max = 100):")
    for n in ["baseline", "ex6_double"]:
        for r in [r for r in res if r["cfg"]["name"] == n]:
            print(f"  {n:12s} seed {r['cfg']['seed']}: final {r['qmeans'][-1]:6.1f}, "
                  f"peak {max(r['qmeans']):6.1f}")

    print("\nEx3: max_a Q along a greedy episode (start / near the end):")
    for n in ["baseline", "ex3_trunc_bug"]:
        for r in [r for r in res if r["cfg"]["name"] == n]:
            q = r["q_traj"]
            print(f"  {n:14s} seed {r['cfg']['seed']}: len {len(q):3d}, Q[0:50]={np.mean(q[:50]):6.1f}, "
                  f"Q[last 50]={np.mean(q[-50:]):6.1f}")

    plot_group(res, ["baseline", "ex1_no_target"], "Ex1: target network", "exp_ex1_target.png")
    plot_group(res, ["baseline", "ex2_no_replay"], "Ex2: replay buffer", "exp_ex2_replay.png")
    plot_group(res, ["baseline", "ex3_trunc_bug"], "Ex3: truncation bug", "exp_ex3_trunc.png")
    plot_group(res, ["ex4_decay_1k", "baseline", "ex4_decay_50k"], "Ex4: epsilon decay 1k / 5k / 50k", "exp_ex4_eps.png")
    plot_group(res, ["sb3_net16", "sb3_net64", "sb3_baseline", "sb3_net256"], "SB3 Ex3: network size", "exp_sb3_net.png")
    plot_group(res, ["sb3_baseline", "sb3_target1"], "SB3 Ex4: target_update_interval", "exp_sb3_target.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n, c in [("baseline", "royalblue"), ("ex6_double", "darkorange")]:
        for r in [r for r in res if r["cfg"]["name"] == n]:
            axes[0].plot(np.arange(len(r["qmeans"])) * 10, r["qmeans"], color=c, alpha=0.7,
                         label=n if r["cfg"]["seed"] == 0 else None)
    axes[0].axhline(100, color="k", ls="--", lw=0.8, label="1/(1-gamma)=100")
    axes[0].set_title("Ex6: mean predicted Q(s,a)"); axes[0].set_xlabel("Episode"); axes[0].legend()
    for n, c in [("baseline", "royalblue"), ("ex3_trunc_bug", "crimson")]:
        for r in [r for r in res if r["cfg"]["name"] == n]:
            axes[1].plot(r["q_traj"], color=c, alpha=0.7, label=n if r["cfg"]["seed"] == 0 else None)
    axes[1].set_title("Ex3: max_a Q(s_t,a) along a greedy episode"); axes[1].set_xlabel("t"); axes[1].legend()
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("exp_ex3_ex6_q.png", dpi=110); plt.close(fig)
    print("plots saved")
