"""Algorithmic Runtime and Convergence Benchmarking Suite.

Compares Howard's Policy Iteration (infinite horizon), Value Iteration (infinite horizon),
and Backward Induction (finite horizon T) across varying state space discretizations.

Note:
    Execution runtimes and solver timings are hardware- and machine-dependent.
    All durations are measured using time.perf_counter().
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backward_induction import backward_induction, compare_policies
from pension_core import (
    Grids,
    Params,
    build_next_state,
    policy_iteration,
    value_iteration,
)


def run_benchmark(
    n_vs: Sequence[int] = (100, 250, 500, 1000),
    T: int = 10,
    repeats: int = 1,
    out_dir: str = "results",
    skip_vi_above: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | int]]:
    """Benchmark Policy Iteration, Value Iteration, and Backward Induction.

    Parameters:
        n_vs: Sequence of wealth grid points n_V.
        T: Finite planning horizon for Backward Induction in years (default: 10).
        repeats: Repetitions per algorithm for median timing (default: 1).
        out_dir: Output directory for benchmark artifacts.
        skip_vi_above: If set, skips Value Iteration for n_V > skip_vi_above.
        progress: Optional logging callback.

    Returns:
        List of benchmark metric dictionaries per grid configuration.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "benchmark_runtimes.csv")
    png_path = os.path.join(out_dir, "benchmark_runtime.png")

    results: list[dict[str, float | int]] = []
    total_start = time.perf_counter()

    def log(msg: str) -> None:
        if progress:
            progress(msg)
        print(msg)

    log("=" * 85)
    log(f"RUNTIME BENCHMARK: Policy Iteration vs Value Iteration vs Backward Induction (T={T})")
    log("=" * 85)

    for n_V in n_vs:
        p = Params(n_V=n_V)
        n_S = p.n_V * p.n_ccr
        log(f"\n>>> Benchmarking Grid n_V={n_V} (n_S={n_S:,}, actions={p.n_a}, shocks={p.n_z})")

        # 1. Measure standalone setup time (Grids + build_next_state)
        setup_times = []
        for _ in range(repeats):
            t0_setup = time.perf_counter()
            g = Grids(p)
            _ = build_next_state(g)
            setup_times.append(time.perf_counter() - t0_setup)
        setup_sec = float(np.median(setup_times))
        log(f"    Setup (Grids + transition tensor): {setup_sec:.3f}s")

        # 2. Benchmark Howard's Policy Iteration
        log(f"    Running Policy Iteration ({repeats} run{'s' if repeats > 1 else ''})...")
        pi_times = []
        sol_pi = None
        for _ in range(repeats):
            sol_pi = policy_iteration(p)
            pi_times.append(sol_pi.seconds)
        assert sol_pi is not None
        pi_sec = float(np.median(pi_times))
        pi_iters = sol_pi.iterations
        log(f"    -> PI: {pi_sec:.3f}s, {pi_iters} iterations")

        # 3. Benchmark Backward Induction (T=10)
        log(f"    Running Backward Induction T={T} ({repeats} run{'s' if repeats > 1 else ''})...")
        bi_times = []
        sol_bi = None
        for _ in range(repeats):
            sol_bi = backward_induction(p, T=T)
            bi_times.append(sol_bi.seconds)
        assert sol_bi is not None
        bi_sec = float(np.median(bi_times))
        bi_stages = T
        log(f"    -> BI: {bi_sec:.3f}s, {bi_stages} stages")

        # 4. Benchmark Value Iteration (if not skipped)
        sol_vi = None
        if skip_vi_above is not None and n_V > skip_vi_above:
            log(f"    Skipping Value Iteration (n_V={n_V} > {skip_vi_above})")
            vi_sec = float("nan")
            vi_iters = -1
            pi_vs_vi_pct = float("nan")
            vi_vs_bi_pct = float("nan")
        else:
            log(f"    Running Value Iteration ({repeats} run{'s' if repeats > 1 else ''})...")
            vi_times = []
            for _ in range(repeats):
                sol_vi = value_iteration(p)
                vi_times.append(sol_vi.seconds)
            assert sol_vi is not None
            vi_sec = float(np.median(vi_times))
            vi_iters = sol_vi.iterations
            pi_vs_vi_pct = float(np.mean(sol_pi.policy == sol_vi.policy) * 100.0)
            vi_vs_bi_pct = float(np.mean(sol_vi.policy == sol_bi.policy) * 100.0)
            log(f"    -> VI: {vi_sec:.3f}s, {vi_iters} iterations (PI agreement: {pi_vs_vi_pct:.2f}%)")

        # 5. Evaluate policy agreement
        cmp_pi_bi = compare_policies(sol_bi, sol_pi, ref=sol_pi)
        pi_vs_bi_pct = cmp_pi_bi["exact_agreement_pct"]
        pi_vs_bi_equiv = cmp_pi_bi["equivalent_action_pct"]
        log(f"    -> Policy Agreement: PI vs BI year-0 = {pi_vs_bi_pct:.2f}% (equiv = {pi_vs_bi_equiv:.2f}%)")

        row: dict[str, float | int] = {
            "n_V": n_V,
            "n_S": n_S,
            "setup_seconds": setup_sec,
            "PI_seconds": pi_sec,
            "PI_iterations": pi_iters,
            "VI_seconds": vi_sec,
            "VI_iterations": vi_iters,
            "BI_seconds": bi_sec,
            "BI_stages": bi_stages,
            "PI_vs_VI_agreement_pct": pi_vs_vi_pct,
            "PI_vs_BI_year0_agreement_pct": pi_vs_bi_pct,
            "PI_vs_BI_year0_equivalent_pct": pi_vs_bi_equiv,
            "VI_vs_BI_year0_agreement_pct": vi_vs_bi_pct,
        }
        results.append(row)

    total_wall = time.perf_counter() - total_start
    log(f"\nAll benchmarks finished in {total_wall:.2f}s total wall time.")

    # Save results/benchmark_runtimes.csv
    fieldnames = [
        "n_V",
        "n_S",
        "setup_seconds",
        "PI_seconds",
        "PI_iterations",
        "VI_seconds",
        "VI_iterations",
        "BI_seconds",
        "BI_stages",
        "PI_vs_VI_agreement_pct",
        "PI_vs_BI_year0_agreement_pct",
        "PI_vs_BI_year0_equivalent_pct",
        "VI_vs_BI_year0_agreement_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log(f"Saved benchmark CSV to: {csv_path}")

    # Plot results/benchmark_runtime.png
    plot_benchmark(results, png_path=png_path)
    log(f"Saved benchmark figure to: {png_path}")

    return results


def plot_benchmark(
    results: list[dict[str, float | int]],
    png_path: str = "results/benchmark_runtime.png",
) -> None:
    """Generate a 3-panel publication-quality figure from benchmark results."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    n_s = [int(r["n_S"]) for r in results]
    pi_time = [float(r["PI_seconds"]) for r in results]
    bi_time = [float(r["BI_seconds"]) for r in results]
    vi_time = [float(r["VI_seconds"]) for r in results]
    setup_time = [float(r["setup_seconds"]) for r in results]

    # Panel 1: Runtime vs n_S (log-log)
    ax1.plot(n_s, pi_time, "o-", color="#1f77b4", lw=2, label="Policy Iteration (Howard's)")
    ax1.plot(n_s, bi_time, "^-", color="#2ca02c", lw=2, label="Backward Induction (T=10)")
    # Filter valid VI points
    valid_vi = [(s, t) for s, t in zip(n_s, vi_time) if not np.isnan(t)]
    if valid_vi:
        ax1.plot([s for s, _ in valid_vi], [t for _, t in valid_vi], "s--", color="#d62728", lw=2, label="Value Iteration (Bellman)")
    ax1.plot(n_s, setup_time, ":", color="#7f7f7f", lw=1.5, label="Shared Setup (Grids+Trans)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("State Space Size $n_S$", fontsize=10)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=10)
    ax1.set_title("Runtime Scaling (log-log)", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper left")

    # Panel 2: Iterations / Stages per Algorithm
    pi_iters = [int(r["PI_iterations"]) for r in results]
    bi_stages = [int(r["BI_stages"]) for r in results]
    ax2.plot(n_s, pi_iters, "o-", color="#1f77b4", lw=2, label="PI Iterations")
    ax2.plot(n_s, bi_stages, "^-", color="#2ca02c", lw=2, label="BI Stages ($T=10$)")
    valid_vi_iters = [(s, int(r["VI_iterations"])) for s, r in zip(n_s, results) if not np.isnan(float(r["VI_seconds"]))]
    if valid_vi_iters:
        ax2.plot([s for s, _ in valid_vi_iters], [it for _, it in valid_vi_iters], "s--", color="#d62728", lw=2, label="VI Iterations")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("State Space Size $n_S$", fontsize=10)
    ax2.set_ylabel("Iterations / Stages", fontsize=10)
    ax2.set_title("Convergence Iterations / Stages", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="center right")

    # Panel 3: Policy Agreement
    pi_bi_exact = [float(r["PI_vs_BI_year0_agreement_pct"]) for r in results]
    pi_bi_equiv = [float(r["PI_vs_BI_year0_equivalent_pct"]) for r in results]
    ax3.plot(n_s, pi_bi_exact, "o-", color="#2ca02c", lw=2, label="PI vs BI (Year-0 Exact)")
    ax3.plot(n_s, pi_bi_equiv, "s--", color="#17becf", lw=2, label="PI vs BI (Year-0 Equiv)")

    valid_pi_vi = [(s, float(r["PI_vs_VI_agreement_pct"])) for s, r in zip(n_s, results) if not np.isnan(float(r["PI_vs_VI_agreement_pct"]))]
    if valid_pi_vi:
        ax3.plot([s for s, _ in valid_pi_vi], [ag for _, ag in valid_pi_vi], "D-.", color="#9467bd", lw=1.8, label="PI vs VI (Infinite)")

    valid_vi_bi = [(s, float(r["VI_vs_BI_year0_agreement_pct"])) for s, r in zip(n_s, results) if not np.isnan(float(r["VI_vs_BI_year0_agreement_pct"]))]
    if valid_vi_bi:
        ax3.plot([s for s, _ in valid_vi_bi], [ag for _, ag in valid_vi_bi], "x:", color="#e377c2", lw=1.5, label="VI vs BI (Year-0)")

    ax3.set_xscale("log")
    ax3.set_ylim(-2, 105)
    ax3.set_xlabel("State Space Size $n_S$", fontsize=10)
    ax3.set_ylabel("Policy Agreement (%)", fontsize=10)
    ax3.set_title("Policy Agreement Across Algorithms", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=7.5, loc="lower right")

    plt.suptitle(
        "Algorithm Benchmark: Policy Iteration vs Value Iteration vs Backward Induction\n"
        "Note: Backward Induction solves a finite 10-year problem; Policy Iteration and Value Iteration solve the infinite-horizon MDP.",
        fontsize=10.5,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point for algorithmic runtime benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark Policy Iteration, Value Iteration, and Backward Induction."
    )
    parser.add_argument(
        "--n-vs",
        type=str,
        default="100,250,500,1000",
        help="Comma-separated n_V grid values (default: 100,250,500,1000)",
    )
    parser.add_argument("--T", type=int, default=10, help="Backward induction horizon (default: 10)")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions for median timing (default: 1)")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument(
        "--skip-vi-above",
        type=int,
        default=None,
        help="Skip Value Iteration when n_V exceeds this threshold (e.g. 500)",
    )

    args = parser.parse_args()
    n_vs = [int(v.strip()) for v in args.n_vs.split(",") if v.strip()]

    run_benchmark(
        n_vs=n_vs,
        T=args.T,
        repeats=args.repeats,
        out_dir=args.out_dir,
        skip_vi_above=args.skip_vi_above,
    )


if __name__ == "__main__":
    main()
