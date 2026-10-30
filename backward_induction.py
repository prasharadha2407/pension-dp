"""Finite-horizon Backward Induction solver for the pension fund MDP.

Subclasses pension_core.Solution to provide drop-in compatibility with GUI
and visualization tools, computing optimal stage-dependent investment policies
for a finite planning horizon T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Callable

import numpy as np

from pension_core import (
    Grids,
    Params,
    Solution,
    _q_values,
    build_next_state,
    policy_iteration,
    utility,
)


@dataclass
class BackwardSolution(Solution):
    """Stores the solution of the finite-horizon Backward Induction problem.

    Subclasses pension_core.Solution so it can be passed to all existing
    plotting, metrics, and GUI consumers.

    Inherited fields represent year-0 quantities:
        policy: Year-0 optimal action indices, shape (n_S,).
        value: Year-0 value function V_0(s), shape (n_S,).
        iterations: Number of stages T.
        history: List of (stage_index, max |V_k - V_{k+1}|, elapsed_seconds).
        seconds: Total execution time in seconds (including grid/transition setup).
        method: Solver identifier, set to "backward_induction".

    Additional fields:
        horizon: Planning horizon T in years.
        policy_by_year: Array of optimal action indices across all stages,
            shape (T, n_S) of dtype int64.
        value_by_year: Value function per stage from k=0 to terminal k=T (all 0),
            shape (T + 1, n_S) of dtype float64.
    """

    horizon: int = 10
    policy_by_year: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    value_by_year: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


def backward_induction(
    p: Params,
    T: int = 10,
    progress: Callable[[str], None] | None = None,
    g: Grids | None = None,
    nxt: np.ndarray | None = None,
) -> BackwardSolution:
    """Solve the finite-horizon pension MDP by backward induction.

    Parameters:
        p: Problem parameters.
        T: Finite decumulation horizon in years (default: 10).
        progress: Optional progress callback receiving informative strings.
        g: Optional precomputed Grids instance. If None, built from p.
        nxt: Optional precomputed next_state tensor. If None, built from g.

    Returns:
        BackwardSolution containing year-0 policy, value, stage-by-stage trajectories,
        and execution history.
    """
    p.validate()
    if T <= 0:
        raise ValueError(f"Horizon T must be positive, got {T}")

    t_all = time.perf_counter()
    if g is None:
        g = Grids(p)
    if nxt is None:
        nxt = build_next_state(g)

    reward = p.reward_scale * utility(g.P_s, p.F)
    n_S = g.n_S

    policy_by_year = np.empty((T, n_S), dtype=np.int64)
    value_by_year = np.zeros((T + 1, n_S), dtype=np.float64)

    history = []
    # Dynamic programming recursion backward from k = T - 1 down to 0
    for k in range(T - 1, -1, -1):
        t0 = time.perf_counter()
        q = _q_values(g, nxt, value_by_year[k + 1], reward)
        # Ties break to smallest action index (safest investment asset)
        policy_k = np.argmax(q, axis=1)
        V_k = q[np.arange(n_S), policy_k]

        policy_by_year[k] = policy_k
        value_by_year[k] = V_k

        diff = float(np.abs(V_k - value_by_year[k + 1]).max())
        elapsed = time.perf_counter() - t0
        history.append((k, diff, elapsed))
        if progress:
            progress(
                f"Backward induction stage {k} (T={T}): max |V_{k} - V_{k+1}| = {diff:.4e}"
            )

    # Order history chronologically by year k (0 to T-1)
    history.sort(key=lambda h: h[0])

    total_seconds = time.perf_counter() - t_all
    return BackwardSolution(
        params=p,
        grids=g,
        policy=policy_by_year[0].copy(),
        value=value_by_year[0].copy(),
        iterations=T,
        history=history,
        seconds=total_seconds,
        method="backward_induction",
        horizon=T,
        policy_by_year=policy_by_year,
        value_by_year=value_by_year,
    )


def compare_policies(
    sol_a: Solution,
    sol_b: Solution,
    ref: Solution | None = None,
) -> dict[str, float]:
    """Compare two policies and return detailed agreement metrics.

    Parameters:
        sol_a: First solution (e.g. BackwardSolution or Solution).
        sol_b: Second solution (e.g. infinite-horizon Solution from Policy Iteration).
        ref: Reference solution supplying grids for action-space difference mapping.

    Returns:
        Dictionary with:
            - exact_agreement_pct: Percentage of states where action indices match exactly.
            - equivalent_action_pct: Percentage where actions match within 1 grid interval.
            - mean_abs_action_diff: Mean absolute difference in continuous risky asset shares.
            - max_abs_action_diff: Maximum absolute difference in continuous risky asset shares.
    """
    g = ref.grids if ref is not None else sol_a.grids
    p = ref.params if ref is not None else sol_a.params

    exact = float(np.mean(sol_a.policy == sol_b.policy) * 100.0)
    equiv = float(np.mean(np.abs(sol_a.policy - sol_b.policy) <= 1) * 100.0)

    # Convert action indices to continuous risky asset allocation a in [0, 1]
    a_cont = g.a
    diff_cont = np.abs(a_cont[sol_a.policy] - a_cont[sol_b.policy])

    mean_diff = float(np.mean(diff_cont))
    max_diff = float(np.max(diff_cont))

    return {
        "exact_agreement_pct": exact,
        "equivalent_action_pct": equiv,
        "mean_abs_action_diff": mean_diff,
        "max_abs_action_diff": max_diff,
    }


def run_horizon_sweep(
    p: Params,
    horizons: list[int] | tuple[int, ...] = (5, 10, 20, 40, 55, 80, 160),
    out_dir: str = "results",
) -> list[dict[str, float]]:
    """Run a horizon sweep comparing Backward Induction to stationary Policy Iteration.

    Saves results to:
        - {out_dir}/bi_vs_pi_horizon.csv
        - {out_dir}/bi_vs_pi_horizon.png

    Returns:
        List of result dictionaries per horizon.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bi_vs_pi_horizon.csv")
    png_path = os.path.join(out_dir, "bi_vs_pi_horizon.png")

    print(f"\nSolving reference infinite-horizon Policy Iteration (n_V={p.n_V}, n_S={p.n_V * p.n_ccr})...")
    t0_pi = time.perf_counter()
    pi_sol = policy_iteration(p)
    t_pi = time.perf_counter() - t0_pi
    print(f"Policy Iteration converged in {pi_sol.iterations} iterations ({t_pi:.2f}s).")

    v_max = float(np.max(np.abs(pi_sol.value)))
    g = pi_sol.grids
    nxt = build_next_state(g)

    rows = []
    print(f"\nRunning Backward Induction horizon sweep across T in {list(horizons)}...")
    print(f"{'T':>4} | {'Exact Match':>11} | {'Equiv Match':>11} | {'Mean |delta a|':>15} | {'Value Gap':>11} | {'gamma^T':>11} | {'gamma^T * max|v|':>15}")
    print("-" * 96)

    for T in horizons:
        bi_sol = backward_induction(p, T=T, g=g, nxt=nxt)
        cmp_res = compare_policies(bi_sol, pi_sol, ref=pi_sol)
        val_gap = float(np.max(np.abs(bi_sol.value - pi_sol.value)))
        gamma_pow = float(p.gamma ** T)
        gamma_pow_vmax = float(gamma_pow * v_max)

        row = {
            "T": int(T),
            "exact_agreement_pct": float(cmp_res["exact_agreement_pct"]),
            "equivalent_action_pct": float(cmp_res["equivalent_action_pct"]),
            "mean_abs_action_diff": float(cmp_res["mean_abs_action_diff"]),
            "value_gap_max": val_gap,
            "gamma_pow_T": gamma_pow,
            "gamma_pow_T_times_vmax": gamma_pow_vmax,
        }
        rows.append(row)
        print(
            f"{T:4d} | {row['exact_agreement_pct']:10.2f}% | "
            f"{row['equivalent_action_pct']:10.2f}% | "
            f"{row['mean_abs_action_diff']:15.4f} | "
            f"{row['value_gap_max']:11.4e} | "
            f"{row['gamma_pow_T']:11.4e} | "
            f"{row['gamma_pow_T_times_vmax']:15.4e}"
        )

    # Save CSV
    fieldnames = [
        "T",
        "exact_agreement_pct",
        "equivalent_action_pct",
        "mean_abs_action_diff",
        "value_gap_max",
        "gamma_pow_T",
        "gamma_pow_T_times_vmax",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved horizon comparison table to: {csv_path}")

    # Plot figure
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    ts = [r["T"] for r in rows]
    exacts = [r["exact_agreement_pct"] for r in rows]
    equivs = [r["equivalent_action_pct"] for r in rows]
    gaps = [r["value_gap_max"] for r in rows]
    bounds = [r["gamma_pow_T_times_vmax"] for r in rows]

    color1 = "#1f77b4"
    color2 = "#2ca02c"
    color3 = "#d62728"
    color4 = "#9467bd"

    l1 = ax1.plot(ts, exacts, "o-", color=color1, lw=2, label="Exact Action Agreement (%)")
    l2 = ax1.plot(ts, equivs, "s--", color=color2, lw=2, label="Equivalent Action (±1 step, %)")
    ax1.set_xlabel("Planning Horizon T (years)", fontsize=11)
    ax1.set_ylabel("Year-0 Policy Agreement with PI (%)", color=color1, fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Mark T=55 explicitly
    if 55 in ts:
        exact_55 = next(r["exact_agreement_pct"] for r in rows if r["T"] == 55)
        ax1.axvline(55, color="#7f7f7f", linestyle="--", lw=1.5, alpha=0.8)
        ax1.plot(55, exact_55, "o", color="#d62728", ms=8, zorder=5)
        ax1.annotate(
            f"T=55 (Paper Horizon, 65-120)\nExact Agree: {exact_55:.1f}%",
            xy=(55, exact_55),
            xytext=(62, exact_55 - 18),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff", edgecolor="#d62728", alpha=0.9),
        )

    ax2 = ax1.twinx()
    l3 = ax2.plot(ts, gaps, "^-.", color=color3, lw=1.8, label=r"Max Value Gap $\|V_0 - V^*\|_\infty$")
    l4 = ax2.plot(ts, bounds, "v:", color=color4, lw=1.8, label=r"Contraction Bound $\gamma^T \cdot \max|V^*|$")
    ax2.set_ylabel(r"Value Gap & Contraction Bound", color=color3, fontsize=11)
    ax2.set_yscale("log")

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=8.5)
    plt.title(
        f"Backward Induction Horizon Convergence to Stationary Policy (n_V={p.n_V})\n"
        r"Value gap bounded by $\gamma^T \cdot \max|V^*|$; T=55 represents the paper's 55-year cohort horizon",
        fontsize=10.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved horizon comparison plot to: {png_path}")

    return rows


def main() -> None:
    """CLI entry point for Backward Induction and horizon sweeps."""
    import argparse
    from pension_core import policy_iteration

    parser = argparse.ArgumentParser(
        description="Finite-horizon Backward Induction solver for pension-dp."
    )
    parser.add_argument("--T", type=int, default=10, help="Planning horizon T in years (default: 10)")
    parser.add_argument("--n-v", type=int, default=200, help="Number of wealth grid points (default: 200)")
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,20,40,55,80,160",
        help="Comma-separated horizon values for sweep (default: 5,10,20,40,55,80,160)",
    )
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save artifacts (default: results)")
    parser.add_argument("--no-sweep", action="store_true", help="Skip horizon sweep and only solve specified T")

    args = parser.parse_args()
    p = Params(n_V=args.n_v)

    print("=" * 75)
    print(f"BACKWARD INDUCTION SOLVER (T={args.T}, n_V={args.n_v}, n_S={args.n_v * p.n_ccr})")
    print("=" * 75)

    # (a) Solve for given T and compare with Policy Iteration
    print(f"\n1. Solving Policy Iteration (infinite-horizon baseline)...")
    pi_sol = policy_iteration(p)
    print(f"   Converged in {pi_sol.iterations} iterations ({pi_sol.seconds:.2f}s).")
    v_max = float(np.max(np.abs(pi_sol.value)))

    print(f"\n2. Solving Backward Induction (T={args.T} years)...")
    bi_sol = backward_induction(p, T=args.T, g=pi_sol.grids)
    print(f"   Completed {bi_sol.iterations} stages in {bi_sol.seconds:.2f}s.")

    cmp_res = compare_policies(bi_sol, pi_sol, ref=pi_sol)
    print(f"\n3. Year-0 Policy Comparison (BI T={args.T} vs Stationary PI):")
    print(f"   - Exact action agreement:     {cmp_res['exact_agreement_pct']:.2f}%")
    print(f"   - Equivalent-action agreement: {cmp_res['equivalent_action_pct']:.2f}%")
    print(f"   - Mean |delta action|:         {cmp_res['mean_abs_action_diff']:.4f}")
    print(f"   - Max |V_0^(T) - V_PI|:        {float(np.max(np.abs(bi_sol.value - pi_sol.value))):.4e}")
    print(f"   - Theoretical gamma^T:         {p.gamma ** args.T:.4e}")
    print(f"   - Theoretical gamma^T * max|V|:{p.gamma ** args.T * v_max:.4e}")

    # (b) Horizon sweep
    if not args.no_sweep:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        run_horizon_sweep(p, horizons=horizons, out_dir=args.out_dir)

    print("\n" + "=" * 75)
    print("BACKWARD INDUCTION COMPLETED")


if __name__ == "__main__":
    main()
