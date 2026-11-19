"""
=====================================================================
 LAB (Week 5, companion): DQN with Stable-Baselines3 on CartPole
=====================================================================
Goal: run the SAME experiment as dqn_lab.py, but with a library.
In dqn_lab.py you built every part by hand. Here each part is one argument:

    dqn_lab.py (by hand)                  Stable-Baselines3 argument
    ------------------------------------  ---------------------------------
    make_q_network(4, 2, hidden=128)      policy_kwargs=dict(net_arch=[128, 128])
    ReplayBuffer(capacity=50_000)         buffer_size=50_000
    batch_size=64                         batch_size=64
    learning_starts=1_000                 learning_starts=1_000
    target_sync=200                       target_update_interval=200
    gamma=0.99, lr=1e-3                   gamma=0.99, learning_rate=1e-3
    one gradient step per env step        train_freq=1, gradient_steps=1
    linear_epsilon(1.0 -> 0.05, 5_000)    exploration_initial_eps=1.0,
                                          exploration_final_eps=0.05,
                                          exploration_fraction=5_000 / TOTAL_STEPS
    agent.act(s, epsilon=0.0)             model.predict(s, deterministic=True)

Two differences to know about:
  * SB3 trains for a number of STEPS (total_timesteps), not episodes.
  * SB3 also handles `terminated` vs `truncated` correctly for you.

Requirements:  pip install "gymnasium>=1.0" numpy matplotlib torch stable-baselines3
Run:           python dqn_lab_sb3.py      (about 1-2 minutes on a CPU)
=====================================================================
"""

import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")          # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

SEED = 0
TOTAL_STEPS = 40_000           # dqn_lab.py's 300 episodes are roughly this many steps


# ---------------------------------------------------------------
# PART 1: Build the agent
# "MlpPolicy" = a fully connected Q-network; net_arch gives the hidden layers.
# ---------------------------------------------------------------
def make_agent(env):
    return DQN(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        learning_starts=1_000,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=200,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=5_000 / TOTAL_STEPS,
        seed=SEED,
        verbose=0,
    )


# ---------------------------------------------------------------
# PART 2: Visualisation
# Monitor is a wrapper that records the return of every training episode.
# ---------------------------------------------------------------
def plot_learning_curve(returns, filename="dqn_sb3_results.png"):
    w = 20
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(returns, alpha=0.3, color="royalblue", label="episode return")
    ax.plot(np.arange(w - 1, len(returns)), np.convolve(returns, np.ones(w) / w, "valid"),
            color="royalblue", linewidth=2, label=f"moving average ({w})")
    ax.axhline(500, color="black", linestyle="--", label="maximum (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning curve (Stable-Baselines3 DQN)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {filename}")


# ---------------------------------------------------------------
# MAIN: run the lab step by step
# ---------------------------------------------------------------
if __name__ == "__main__":
    env = Monitor(gym.make("CartPole-v1"))
    eval_env = Monitor(gym.make("CartPole-v1"))
    model = make_agent(env)

    print("=" * 65)
    print("STEP 0: the Q-network that SB3 built")
    print("=" * 65)
    print(model.q_net)          # compare with make_q_network in dqn_lab.py
    print("The target network is model.q_net_target - same shape, separate weights.")

    print("\n" + "=" * 65)
    print("STEP 1: baseline - the untrained network")
    print("=" * 65)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Untrained policy: return {mean:.1f} +/- {std:.1f}")

    print("\n" + "=" * 65)
    print("STEP 2: train DQN")
    print("=" * 65)
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_STEPS)
    returns = env.get_episode_rewards()
    print(f"Training took {time.perf_counter() - t0:.0f} s, "
          f"{TOTAL_STEPS} steps, {len(returns)} episodes.")
    print(f"Mean return over the last 25 training episodes: {np.mean(returns[-25:]):.1f}")

    print("\n" + "=" * 65)
    print("STEP 3: evaluate the greedy policy (deterministic=True)")
    print("=" * 65)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"DQN greedy policy: return {mean:.1f} +/- {std:.1f}   (maximum 500)")

    plot_learning_curve(returns)
    model.save("dqn_cartpole")  # reload later with DQN.load("dqn_cartpole")
    print("Saved model -> dqn_cartpole.zip")
    env.close()
    eval_env.close()

# ---------------------------------------------------------------
# EXERCISES
# ---------------------------------------------------------------
# 1. Match the parts. For each argument of DQN(...) in make_agent, find the
#    line in dqn_lab.py that does the same job.
# 2. Compare. Run both scripts with SEED = 0, 1, 2. Are the final returns in
#    the same range? The algorithm is the same, so why are the numbers not
#    identical?
# 3. Network size. Try net_arch=[16], [64, 64] and [256, 256]. How do
#    training time and final return change?
# 4. Target network. Set target_update_interval=1 (as in Exercise 1 of
#    dqn_lab.py). Do you see the same effect as in your own implementation?
# 5. Read the source. Open stable_baselines3/dqn/dqn.py and find the train()
#    method. Find the TD target and the loss; compare them line by line with
#    DQNAgent.learn in dqn_lab.py.
# 6. Trade-offs. Name one thing that is easier with the library, and one
#    thing that was easier to see or change in your own implementation.
