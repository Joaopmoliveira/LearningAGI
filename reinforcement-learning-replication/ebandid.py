"""
Replication of the 10-armed testbed experiment from
Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.), Chapter 2.
 
Reproduces Figure 2.2: average reward and % optimal action over time
for the epsilon-greedy action-value method with epsilon = 0, 0.01, 0.1.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
 
def run_bandit_experiment(k=10, epsilon=0.0, steps=3000, runs=2000, seed=None):
    """
    Run the k-armed bandit experiment for a given epsilon.
 
    For each of `runs` independent runs:
      - Draw true action values q*(a) ~ N(0, 1) for a = 1..k
      - At each of `steps` time steps, choose an action epsilon-greedily
        w.r.t. the sample-average action-value estimates Q(a)
      - Reward is drawn from N(q*(a), 1)
      - Update Q(a) incrementally: Q(a) <- Q(a) + (1/N(a)) * (R - Q(a))
 
    Returns
    -------
    avg_reward : array of shape (steps,)
        Reward averaged across all runs, at each time step.
    pct_optimal : array of shape (steps,)
        Percentage of runs in which the optimal action was chosen,
        at each time step.
    """
    rng = np.random.default_rng(seed)
 
    rewards = np.zeros((runs, steps))
    is_optimal = np.zeros((runs, steps))
 
    for run in range(runs):
        q_true = rng.normal(loc=0.0, scale=1.0, size=k)
        optimal_action = np.argmax(q_true)
 
        Q = np.zeros(k)   # action-value estimates
        N = np.zeros(k)   # action counts
 
        for t in range(steps):
            # epsilon-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(k)
            else:
                # break ties randomly among the max-valued actions
                action = rng.choice(np.flatnonzero(Q == Q.max()))
 
            reward = rng.normal(loc=q_true[action], scale=1.0)
 
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
 
            rewards[run, t] = reward
            is_optimal[run, t] = 1.0 if action == optimal_action else 0.0
 
    avg_reward = rewards.mean(axis=0)
    pct_optimal = is_optimal.mean(axis=0) * 100.0
    return avg_reward, pct_optimal
 
 
def main():
    epsilons = [0.0, 0.01, 0.1]
    colors = ["green", "red", "blue"]
    steps = 3000
    runs = 2000
 
    results = {}
    for eps in epsilons:
        print(f"Running epsilon = {eps} ...")
        avg_reward, pct_optimal = run_bandit_experiment(
            k=10, epsilon=eps, steps=steps, runs=runs, seed=42
        )
        results[eps] = (avg_reward, pct_optimal)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))
 
    for eps, color in zip(epsilons, colors):
        avg_reward, _ = results[eps]
        label = f"$\\epsilon$ = {eps}" if eps > 0 else "$\\epsilon$ = 0 (greedy)"
        ax1.plot(avg_reward, color=color, label=label)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Average Reward vs. Steps (10-armed testbed)")
    ax1.legend(loc="lower right")
 
    for eps, color in zip(epsilons, colors):
        _, pct_optimal = results[eps]
        label = f"$\\epsilon$ = {eps}" if eps > 0 else "$\\epsilon$ = 0 (greedy)"
        ax2.plot(pct_optimal, color=color, label=label)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)
    ax2.set_title("% Optimal Action vs. Steps (10-armed testbed)")
    ax2.legend(loc="lower right")
 
    plt.tight_layout()
    plt.savefig("bandit_epsilon_greedy.png", dpi=150)
    print("Saved figure to bandit_epsilon_greedy.png")
 
 
if __name__ == "__main__":
    main()
