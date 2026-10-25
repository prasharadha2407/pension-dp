"""Main executable driver for the pension-dp project.

Runs:
1. Dynamic Programming Optimization via Howard's Policy Iteration for alphas in {0.0, 0.2, 0.4}.
2. Monte Carlo 10,000-path simulation and empirical metric extraction.
3. Formatted comparison tables against paper Tables 1 and 2 (Lichtenstern & Zagst, 2022).
4. Dual-mode calibration verification: 'as_printed' vs 'paper_calibrated'.
5. Export of publication-quality plots to results/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from analysis import generate_complexity_table, plot_benchmarks, run_runtime_benchmarks
from brute_force_demo import run_brute_force_demo
from config import Config, ConfigMode
from grids import create_grids
from plotting import (
    plot_fig1_policy_vs_ccr,
    plot_fig8_densities,
    plot_scenario_trajectories,
)
from policy_iteration import run_policy_iteration
from simulate import format_comparison_table, run_monte_carlo_simulation
from transitions import precompute_transitions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pension_dp")


def run_pipeline(
    mode: ConfigMode = "paper_calibrated",
    alphas: list[float] | None = None,
    n_paths: int = 10000,
    output_dir: str = "results",
    run_sim: bool = True,
    run_benchmarks: bool = True,
    run_brute_force: bool = True,
    solver: str = "auto",
) -> tuple[dict[float, np.ndarray], list[dict[str, float]], pd.DataFrame]:
    """Execute complete optimization, simulation, and analysis pipeline for a given mode."""
    if alphas is None:
        alphas = [0.0, 0.2, 0.4]

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"RUNNING PENSION-DP OPTIMIZATION PIPELINE (MODE: {mode.upper()})")
    print("=" * 80)

    policies_by_alpha: dict[float, np.ndarray] = {}
    sim_results_list = []
    sim_summaries = []

    for alpha in alphas:
        print(f"\n>>> Solving MDP for Buffer Parameter alpha = {alpha:.1f} ...")
        cfg = Config(mode=mode, alpha=alpha)
        grids = create_grids(cfg)

        P0 = cfg.initial_pension()
        print(f"    Parameters: r={cfg.r}, lambda_x={cfg.lambda_x}, r+lambda_x={cfg.r_plus_lambda:.4f}")
        print(f"    State space: n_S={grids.n_S} (n_V={grids.n_V}, n_CCR={grids.n_CCR}) | actions: {grids.n_actions}")
        print(f"    Initial Pension P0({alpha:.1f}) = {P0:.2f} (Paper reference: 277 for 0.0, 270 for 0.2, 258 for 0.4)")

        # Precompute transitions
        t0_trans = time.perf_counter()
        trans_res = precompute_transitions(grids, chunk_size=2600)
        t_trans = time.perf_counter() - t0_trans
        print(f"    Transitions precomputed in {t_trans:.2f}s "
              f"(clamped min={trans_res.v_min_clamped_count}, max={trans_res.v_max_clamped_count})")

        # Howard's Policy Iteration
        t0_pi = time.perf_counter()
        pi_res = run_policy_iteration(
            grids,
            trans_res.next_state,
            epsilon=0.01,
            max_iter=100,
            solver=solver,
            verbose=True,
        )
        t_pi = time.perf_counter() - t0_pi
        print(f"    Policy Iteration converged in {pi_res.iterations} iterations ({t_pi:.2f}s). "
              f"Mean a*: {np.mean(pi_res.policy_actions):.3f}")

        policies_by_alpha[alpha] = pi_res.policy_actions

        if run_sim:
            print(f"    Running Monte Carlo simulation ({n_paths:,} paths, 10 years)...")
            t0_sim = time.perf_counter()
            sim_res = run_monte_carlo_simulation(
                grids=grids,
                optimal_policy_actions=pi_res.policy_actions,
                n_paths=n_paths,
                n_years=10,
                seed=42,
            )
            t_sim = time.perf_counter() - t0_sim
            print(f"    Simulation completed in {t_sim:.2f}s.")
            sim_results_list.append(sim_res)
            sim_summaries.append(sim_res.summary_metrics)

    # 1. Generate Figures
    print("\n>>> Generating publication-quality figures in results/ ...")
    fig1_path = plot_fig1_policy_vs_ccr(grids, policies_by_alpha, output_dir=output_dir)
    print(f"    Figure 1 saved: {fig1_path}")

    scen_paths = plot_scenario_trajectories(grids, policies_by_alpha, output_dir=output_dir)
    for sp in scen_paths:
        print(f"    Scenario figure saved: {sp}")

    if run_sim and sim_results_list:
        fig8_path = plot_fig8_densities(sim_results_list, output_dir=output_dir)
        print(f"    Figure 8 saved: {fig8_path}")

    # 2. Formatted Comparison Table against Paper Tables 1 and 2
    comp_df = pd.DataFrame()
    if sim_summaries:
        comp_df = format_comparison_table(sim_summaries, mode=mode)
        csv_path = os.path.join(output_dir, f"table_comparison_{mode}.csv")
        comp_df.to_csv(csv_path, index=False)

        print("\n" + "=" * 90)
        print(f"SIMULATION RESULTS vs. LICHTENSTERN & ZAGST (2022) TABLES 1 & 2 (MODE: {mode.upper()})")
        print("=" * 90)
        print(comp_df.to_string(index=False))
        print("=" * 90)
        print(f"Saved comparison table to: {csv_path}\n")

    # 3. Brute Force Correctness Demo
    if run_brute_force:
        run_brute_force_demo(n_V=4, n_CCR=2, n_actions=3, n_Z=10, alpha=0.0, verbose=True)

    # 4. Runtime & Iteration Benchmarks
    if run_benchmarks:
        print("\n>>> Running Algorithm Benchmark (Policy Iteration vs Value Iteration) ...")
        bench_df = run_runtime_benchmarks(grid_sizes=[100, 250, 500, 1000], alpha=0.0)
        bench_csv = os.path.join(output_dir, "runtime_benchmarks.csv")
        bench_df.to_csv(bench_csv, index=False)
        plot_benchmarks(bench_df, output_dir=output_dir)

        print("\n" + "-" * 75)
        print("RUNTIME BENCHMARK SUMMARY:")
        print("-" * 75)
        print(bench_df.to_string(index=False))
        print("-" * 75)

        # Theoretical complexity table
        comp_table = generate_complexity_table()
        comp_csv = os.path.join(output_dir, "complexity_comparison.csv")
        comp_table.to_csv(comp_csv, index=False)
        print("\nTHEORETICAL COMPLEXITY:")
        print(comp_table[["Algorithm", "Time Complexity", "Typical Operations (n_S=26000, n_a=21)", "Feasibility"]].to_string(index=False))

    return policies_by_alpha, sim_summaries, comp_df


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="pension-dp: Policy Iteration for Pension Fund Optimization (Lichtenstern & Zagst, 2022)"
    )
    parser.add_argument(
        "--mode",
        choices=["paper_calibrated", "as_printed", "both"],
        default="both",
        help="Calibration mode (default: both).",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.2,0.4",
        help="Comma-separated buffer alpha values (default: 0.0,0.2,0.4).",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=10000,
        help="Number of Monte Carlo paths (default: 10000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save plots and CSV tables (default: results).",
    )
    parser.add_argument(
        "--solver",
        choices=["auto", "bicgstab", "spsolve"],
        default="auto",
        help="Sparse linear solver for policy evaluation (default: auto).",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="Skip Monte Carlo simulation.",
    )
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="Skip runtime benchmarking.",
    )
    parser.add_argument(
        "--no-brute-force",
        action="store_true",
        help="Skip brute force demo.",
    )

    args = parser.parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    modes_to_run: list[ConfigMode] = []
    if args.mode == "both":
        modes_to_run = ["paper_calibrated", "as_printed"]
    else:
        modes_to_run = [args.mode]  # type: ignore

    print("\n" + "#" * 80)
    print("  PENSION-DP: DYNAMIC PROGRAMMING FOR OPTIMAL PENSION MANAGEMENT")
    print("  Implementing Lichtenstern & Zagst (2022)")
    print("#" * 80)

    for m in modes_to_run:
        run_pipeline(
            mode=m,
            alphas=alphas,
            n_paths=args.n_paths,
            output_dir=args.output_dir,
            run_sim=not args.no_sim,
            run_benchmarks=not args.no_bench and m == modes_to_run[0],
            run_brute_force=not args.no_brute_force and m == modes_to_run[0],
            solver=args.solver,
        )

    print("\n" + "#" * 80)
    print("  ALL RUNS SUCCESSFULLY COMPLETED!")
    print(f"  Artifacts and figures saved to directory: {os.path.abspath(args.output_dir)}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
