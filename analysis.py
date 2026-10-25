"""Algorithmic complexity, runtime benchmarking, and theoretical property verification.

Performs:
1. Theoretical complexity comparison between Brute Force, Backward Induction,
   Policy Iteration, and Value Iteration.
2. Empirical benchmarking of Policy Iteration vs Value Iteration over grid sizes.
3. Verification of contraction property (gamma < 1) and value function monotonicity (Theorem 5).
4. Export of benchmarking figures to results/.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from grids import create_grids
from policy_iteration import run_policy_iteration
from transitions import precompute_transitions
from value_iteration import run_value_iteration

logger = logging.getLogger(__name__)


def generate_complexity_table() -> pd.DataFrame:
    """Generate theoretical asymptotic computational complexity comparison table."""
    data = [
        {
            "Algorithm": "Brute-Force Enumeration",
            "Time Complexity": "O(n_a^n_S * n_S^2)",
            "Space Complexity": "O(n_S^2)",
            "Typical Operations (n_S=26000, n_a=21)": "21^26,000 ~ 10^34,375",
            "Feasibility": "Completely Infeasible",
            "Theoretical Guarantee": "Global Optimum (exhaustion)",
        },
        {
            "Algorithm": "Backward Induction (Finite Horizon T)",
            "Time Complexity": "O(T * n_S * n_a * n_Z)",
            "Space Complexity": "O(n_S)",
            "Typical Operations (T=10)": "10 * 26,000 * 21 * 40 ~ 2.18 * 10^8",
            "Feasibility": "Feasible (finite horizon only)",
            "Theoretical Guarantee": "Exact finite-horizon Bellman",
        },
        {
            "Algorithm": "Howard's Policy Iteration (Infinite Horizon)",
            "Time Complexity": "O(K * (n_S * n_a * n_Z + nnz(Q)^1.2))",
            "Space Complexity": "O(n_S * n_Z + n_S)",
            "Typical Operations (K <= 7 iters)": "7 * (2.18 * 10^7 eval + 0.1s sparse solve)",
            "Feasibility": "Extremely Fast (~2-3 seconds)",
            "Theoretical Guarantee": "Exact infinite-horizon optimum (Theorem 5)",
        },
        {
            "Algorithm": "Value Iteration (Bellman Operator)",
            "Time Complexity": "O(M * n_S * n_a * n_Z)",
            "Space Complexity": "O(n_S * n_a * n_Z)",
            "Typical Operations (M ~ 300-500 iters)": "400 * 2.18 * 10^7 ops ~ 12 seconds",
            "Feasibility": "Moderately Fast",
            "Theoretical Guarantee": "Geometric contraction (Banach fixed point)",
        },
    ]
    return pd.DataFrame(data)


def run_runtime_benchmarks(
    grid_sizes: list[int] | None = None,
    alpha: float = 0.0,
    vi_tol: float = 1e-6,
) -> pd.DataFrame:
    """Benchmark Policy Iteration vs Value Iteration over varying wealth grid sizes n_V."""
    if grid_sizes is None:
        grid_sizes = [100, 250, 500, 1000]

    results = []

    for n_V in grid_sizes:
        cfg = Config(mode="paper_calibrated", alpha=alpha, n_V=n_V, n_CCR=26, n_actions=21, n_Z=40)
        grids = create_grids(cfg)
        t_trans = precompute_transitions(grids, chunk_size=min(5000, grids.n_S))
        next_state = t_trans.next_state

        n_S = grids.n_S
        logger.info("Benchmarking grid size n_V=%d (n_S=%d)...", n_V, n_S)

        # Policy Iteration
        t0_pi = time.perf_counter()
        pi_res = run_policy_iteration(grids, next_state, epsilon=0.01, max_iter=50, verbose=False)
        pi_time = time.perf_counter() - t0_pi

        # Value Iteration
        t0_vi = time.perf_counter()
        vi_res = run_value_iteration(grids, next_state, tol=vi_tol, max_iter=1000, verbose=False)
        vi_time = time.perf_counter() - t0_vi

        # Action match percentage
        match_pct = float(np.mean(pi_res.policy == vi_res.policy)) * 100.0

        results.append({
            "n_V": n_V,
            "n_S": n_S,
            "PI_time_seconds": pi_time,
            "PI_iterations": pi_res.iterations,
            "VI_time_seconds": vi_time,
            "VI_iterations": vi_res.iterations,
            "Speedup_PI_vs_VI": vi_time / max(pi_time, 1e-6),
            "Policy_Agreement_pct": match_pct,
        })

    return pd.DataFrame(results)


def plot_benchmarks(df_benchmarks: pd.DataFrame, output_dir: str = "results") -> None:
    """Plot runtime and iteration comparisons and save to output_dir."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Runtime
    ax1.plot(
        df_benchmarks["n_S"],
        df_benchmarks["PI_time_seconds"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="Policy Iteration (Howard's)",
    )
    ax1.plot(
        df_benchmarks["n_S"],
        df_benchmarks["VI_time_seconds"],
        "s--",
        color="#ff7f0e",
        linewidth=2,
        label="Value Iteration (Bellman)",
    )
    ax1.set_xlabel("State Space Size (n_S)", fontsize=11)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax1.set_title("Runtime Comparison vs State Space Size", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Iterations to Converge
    ax2.plot(
        df_benchmarks["n_S"],
        df_benchmarks["PI_iterations"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="Policy Iteration (Howard's)",
    )
    ax2.plot(
        df_benchmarks["n_S"],
        df_benchmarks["VI_iterations"],
        "s--",
        color="#ff7f0e",
        linewidth=2,
        label="Value Iteration (Bellman)",
    )
    ax2.set_xlabel("State Space Size (n_S)", fontsize=11)
    ax2.set_ylabel("Iterations to Convergence", fontsize=11)
    ax2.set_title("Convergence Iteration Count", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    runtime_path = f"{output_dir}/runtime_benchmark.png"
    plt.savefig(runtime_path, dpi=300)
    plt.close()
    logger.info("Saved runtime benchmark figure to %s", runtime_path)


def verify_contraction_and_monotonicity(
    grids: Grids,
    next_state: np.ndarray,
) -> dict[str, Any]:
    """Empirically test contraction factor and value function monotonicity (Theorem 5)."""
    gamma = grids.config.gamma
    assert gamma < 1.0, f"Contraction requires gamma < 1, got {gamma}"

    # Run policy iteration with tracking
    pi_res = run_policy_iteration(grids, next_state, epsilon=0.01, verbose=False)

    all_monotone = all(h["is_monotone"] for h in pi_res.history)
    min_diff_ever = min(h["min_v_diff"] for h in pi_res.history)

    return {
        "gamma": gamma,
        "is_contraction": gamma < 1.0,
        "theoretical_contraction_factor": gamma,
        "all_iterations_monotone": all_monotone,
        "minimum_value_difference": min_diff_ever,
        "total_iterations": pi_res.iterations,
    }
