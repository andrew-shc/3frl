"""
compare_mc_results.py
---------------------
Produces two combined plots after both notebooks have been run:

  1. Benchmark boxplot  – constant / random / Q-learning (best) / DDQN (best)
  2. Market-making trajectories – mean ± std of Q_t, X_t, V_t over time

Set the three variables in USER SETTINGS, then run from the /code directory:
    conda run -n 3frl python compare_mc_results.py

Reward units: both agents are evaluated with no internal scaling, so rewards
are comparable.  Q-learning uses MonteCarloEnv; DDQN uses MonteCarloEnvDeep.
Benchmarks (constant / random) are evaluated on MonteCarloEnv.
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")  # headless – saves files instead of showing windows
import matplotlib.pyplot as plt
import numpy as np

from mc_model_mm_q_learning import load_Q
from mc_model_evaluation import (
    evaluate_Q_matrix,
    evaluate_constant_strategy,
    evaluate_random_strategy,
)
from mc_model_mm_deep_rl_batch import (
    load_arguments,
    load_file_names,
    load_agent,
    setup_ddqn_agent,
    setup_batch_env,
    get_env_function,
    compute_reward_agent_batch,
    sample_strategies_mean,
)

CENT = 100  # DDQN internally divides rewards by this; we undo it below

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
QL_FOLDER   = "mc_example"                        # folder_name from Q-learning notebook
DDQN_OUTDIR = "results/mc_model_deep/mc_deep_example/"  # outdir from DDQN notebook
N_TEST      = 200    # evaluation episodes (use multiples of num_envs for DDQN)
C           = 1      # constant-strategy depth
GPU         = -1     # -1 = CPU
SAVE_DIR    = "results/comparison"
# ───────────────────────────────────────────────────────────────────────────────


# ── helpers ────────────────────────────────────────────────────────────────────

def load_best_qtable(folder_name):
    """Scan the Q-learning results folder, return the best Q-table by mean reward."""
    folder = f"results/mc_model/{folder_name}"
    pkl_paths = glob.glob(os.path.join(folder, "*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files in {folder}. Run the Q-learning notebook first.")

    best_q, best_args, best_score = None, None, -np.inf
    for path in pkl_paths:
        stem = os.path.basename(path)[:-4]  # strip .pkl
        try:
            q_tab, args, *_ = load_Q(stem, folder_mode=True, folder_name=folder_name)
        except Exception as e:
            print(f"  skipping {stem}: {e}")
            continue
        rewards, _, _ = evaluate_Q_matrix(None, n=30, Q_tab=q_tab, args=args)
        score = np.mean(rewards)
        if score > best_score:
            best_score, best_q, best_args = score, q_tab, args

    if best_q is None:
        raise RuntimeError("Could not load any Q-table from the folder.")
    print(f"  Best Q-table score (quick eval): {best_score:.4f}")
    return best_q, best_args


def eval_qlearning(folder_name, n_test):
    """Load best Q-table, return (rewards, Qs, Xs, Vs)."""
    q_tab, args = load_best_qtable(folder_name)
    rewards, _, _, Qs, Xs, Vs = evaluate_Q_matrix(
        None, n=n_test, Q_tab=q_tab, args=args, return_X_Q_V=True
    )
    # Prepend t=0 column (initial state is zero) so trajectory starts from t=0
    pad = np.zeros((Qs.shape[0], 1))
    Qs = np.hstack([pad, Qs])
    Xs = np.hstack([pad, Xs])
    Vs = np.hstack([pad, Vs])
    return np.array(rewards), Qs, Xs, Vs, args


def eval_ddqn(outdir, n_test, gpu=-1):
    """Load best DDQN agent, return (rewards, Qs, Xs, Vs) with no reward scaling."""
    model_names, _ = load_file_names(outdir)
    args, info = load_arguments(outdir)

    # Evaluate with reward_scale=1 so units match Q-learning rewards
    eval_args = {**args, "reward_scale": 1, "phi": 0}

    env_fn  = get_env_function()
    num_envs = info["num_envs"]
    # Round n_test down to nearest multiple of num_envs
    n_test_adj = (n_test // num_envs) * num_envs

    vec_env = setup_batch_env(eval_args, env_fn, num_envs=num_envs)

    agents = [load_agent(setup_ddqn_agent(vec_env, info, gpu), mn) for mn in model_names]

    # Quick eval to pick the best agent
    num_steps = eval_args["T"] / eval_args["dt"]
    quick_n   = min(num_envs * 2, n_test_adj)
    mean_rs   = [
        float(np.mean(
            compute_reward_agent_batch(a, vec_env, num_episodes=quick_n,
                                       num_steps=num_steps).reshape(-1)
        )) * CENT
        for a in agents
    ]
    best = agents[int(np.argmax(mean_rs))]
    print(f"  Best DDQN agent index: {int(np.argmax(mean_rs))} (score {max(mean_rs):.4f})")

    # Full reward evaluation; multiply by CENT to undo the internal /100
    rewards = (
        compute_reward_agent_batch(best, vec_env, num_episodes=n_test_adj,
                                   num_steps=num_steps).reshape(-1) * CENT
    )

    # Trajectories – use sample_strategies_mean (all agents, batch_act) to match
    # exactly what visualize_strategies does in the DDQN notebook
    n_traj = min(n_test_adj, max(info["num_envs"], 100))
    Qs, Xs, Vs = sample_strategies_mean(agents, env_fn, eval_args, info, n_traj)
    # keep raw tick units (1 tick = 1 cent, mid ≈ 10000 ticks = $100 stock)

    vec_env.close()
    return rewards, np.array(Qs), np.array(Xs), np.array(Vs)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_boxplot(data_dict, save_path):
    labels = list(data_dict.keys())
    data   = [np.array(v) for v in data_dict.values()]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(data, labels=labels)
    ax.set_title("Benchmark comparison – Tabular Q-learning vs DDQN (MC LOB model)")
    ax.set_ylabel("Total episode reward")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_trajectories(ql_trajs, ddqn_trajs, save_path):
    """
    ql_trajs / ddqn_trajs: tuple of (Qs, Xs, Vs), each shape (n_episodes, T+1)
    """
    series    = ["Inventory $Q_t$", "Cash $X_t$", "Value $V_t = X_t + H_t$"]
    ylabels   = ["$Q_t$ (shares)", "$X_t$ (¢)", "$V_t$ (¢)"]
    ql_color  = "steelblue"
    ddqn_color = "darkorange"

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for ax, ql_arr, ddqn_arr, title, ylabel in zip(
        axes, ql_trajs, ddqn_trajs, series, ylabels
    ):
        for arr, label, color in [
            (ql_arr,   "Q-learning", ql_color),
            (ddqn_arr, "DDQN",       ddqn_color),
        ]:
            t  = np.arange(arr.shape[1])
            m  = np.mean(arr, axis=0)
            s  = np.std(arr,  axis=0)
            ax.plot(t, m, color=color, label=label, linewidth=1.5)
            ax.fill_between(t, m - s, m + s, alpha=0.2, color=color, label=f"±σ ({label})")

        ax.set_title(title)
        ax.set_xlabel("Time $t$ (seconds)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)

    plt.suptitle("Market-making trajectories – mean ± std over episodes", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Q-learning
    print("\n[1/4] Loading Q-learning results...")
    ql_rewards, ql_Qs, ql_Xs, ql_Vs, ql_args = eval_qlearning(QL_FOLDER, N_TEST)

    # 2. DDQN
    print("\n[2/4] Loading DDQN results...")
    ddqn_rewards, ddqn_Qs, ddqn_Xs, ddqn_Vs = eval_ddqn(DDQN_OUTDIR, N_TEST, GPU)

    # 3. Benchmarks on Q-learning env (same units as ql_rewards)
    print("\n[3/4] Evaluating benchmarks...")
    r_const  = evaluate_constant_strategy(ql_args, n=N_TEST, c=C)
    r_random = evaluate_random_strategy(ql_args, n=N_TEST)

    # 4. Plots
    print("\n[4/4] Plotting...")

    plot_boxplot(
        {
            f"Constant (d={C})":  r_const,
            "Random":             r_random,
            "Q-learning (best)":  ql_rewards,
            "DDQN (best)":        ddqn_rewards,
        },
        save_path=os.path.join(SAVE_DIR, "boxplot_comparison.png"),
    )

    plot_trajectories(
        ql_trajs   = (ql_Qs,   ql_Xs,   ql_Vs),
        ddqn_trajs = (ddqn_Qs, ddqn_Xs, ddqn_Vs),
        save_path  = os.path.join(SAVE_DIR, "trajectories_comparison.png"),
    )

    # Summary table
    print("\n── Summary ──────────────────────────────────────────────────")
    fmt = "{:<22} {:>10.4f} {:>10.4f}"
    print(f"{'Strategy':<22} {'Mean':>10} {'Std':>10}")
    print("-" * 44)
    for name, arr in [
        (f"Constant (d={C})", r_const),
        ("Random",            r_random),
        ("Q-learning (best)", ql_rewards),
        ("DDQN (best)",       ddqn_rewards),
    ]:
        print(fmt.format(name, np.mean(arr), np.std(arr)))
    print()
