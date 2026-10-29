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
        policy=policy_by_year[0],
        value=value_by_year[0],
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
    """Compare two policies and evaluate exact vs near-tie action agreement.

    Parameters:
        sol_a: First solution (e.g., BackwardSolution).
        sol_b: Second solution (e.g., Solution from policy_iteration).
        ref: Optional converged reference solution (with ref.value) used to evaluate
            regret for sol_a actions relative to the reference Q-function:
            regret(s) = max_a Q_ref(s, a) - Q_ref(s, a_chosen).
            An action is equivalent if regret <= 1e-9 * (1 + |V_ref(s)|).

    Returns:
        Dictionary with:
            - exact_agreement_pct: Percentage of states where action indices match exactly.
            - mean_abs_action_diff: Mean absolute difference between risky asset shares.
            - equivalent_action_pct: (If ref provided) Percentage of states where
              sol_a action achieves near-optimal value within the near-tie tolerance.
    """
    g = sol_a.grids
    p = sol_a.params

    exact_pct = float(np.mean(sol_a.policy == sol_b.policy) * 100.0)
    a_a = g.a[sol_a.policy]
    a_b = g.a[sol_b.policy]
    mean_diff = float(np.mean(np.abs(a_a - a_b)))

    out: dict[str, float] = {
        "exact_agreement_pct": exact_pct,
        "mean_abs_action_diff": mean_diff,
    }

    if ref is not None and getattr(ref, "value", None) is not None:
        ref_g = getattr(ref, "grids", g)
        ref_p = getattr(ref, "params", p)
        nxt = getattr(ref, "_nxt", None) or getattr(sol_a, "_nxt", None) or getattr(sol_b, "_nxt", None)
        if nxt is None:
            nxt = build_next_state(ref_g)
        reward = ref_p.reward_scale * utility(ref_g.P_s, ref_p.F)
        q_ref = _q_values(ref_g, nxt, ref.value, reward)

        # Evaluate regret of sol_a's chosen action against ref Q-values
        # (If ref is sol_a, evaluate sol_b's actions)
        a_chosen = sol_b.policy if ref is sol_a else sol_a.policy
        v_ref = ref.value
        regret = q_ref.max(axis=1) - q_ref[np.arange(ref_g.n_S), a_chosen]
        tol = 1e-9 * (1.0 + np.abs(v_ref))
        eq_pct = float(np.mean(regret <= tol) * 100.0)
        out["equivalent_action_pct"] = eq_pct

    return out


def run_horizon_sweep(
    p: Params,
    horizons: list[int] | tuple[int, ...] = (5, 10, 20, 40, 80, 160),
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

    g = pi_sol.grids
    nxt = build_next_state(g)

    rows = []
    print(f"\nRunning Backward Induction horizon sweep across T in {list(horizons)}...")
    print(f"{'T':>4} | {'Exact Match':>11} | {'Equiv Match':>11} | {'Mean |Δa|':>10} | {'Value Gap':>11} | {'gamma^T':>11}")
    print("-" * 72)

    for T in horizons:
        bi_sol = backward_induction(p, T=T, g=g, nxt=nxt)
        cmp_res = compare_policies(bi_sol, pi_sol, ref=pi_sol)
        val_gap = float(np.max(np.abs(bi_sol.value - pi_sol.value)))
        gamma_pow = float(p.gamma ** T)

        row = {
            "T": int(T),
            "exact_agreement_pct": float(cmp_res["exact_agreement_pct"]),
            "equivalent_action_pct": float(cmp_res["equivalent_action_pct"]),
            "mean_abs_action_diff": float(cmp_res["mean_abs_action_diff"]),
            "value_gap_max": val_gap,
            "gamma_pow_T": gamma_pow,
        }
        rows.append(row)
        print(
            f"{T:4d} | {row['exact_agreement_pct']:10.2f}% | "
            f"{row['equivalent_action_pct']:10.2f}% | "
            f"{row['mean_abs_action_diff']:10.4f} | "
            f"{row['value_gap_max']:11.4e} | "
            f"{row['gamma_pow_T']:11.4e}"
        )

    # Save CSV
    fieldnames = [
        "T",
        "exact_agreement_pct",
        "equivalent_action_pct",
        "mean_abs_action_diff",
        "value_gap_max",
        "gamma_pow_T",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved horizon comparison table to: {csv_path}")

    # Plot figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ts = [r["T"] for r in rows]
    exacts = [r["exact_agreement_pct"] for r in rows]
    equivs = [r["equivalent_action_pct"] for r in rows]
    gammas = [r["gamma_pow_T"] for r in rows]

    color1 = "#1f77b4"
    color2 = "#2ca02c"
    color3 = "#d62728"

    l1 = ax1.plot(ts, exacts, "o-", color=color1, lw=2, label="Exact Action Agreement (%)")
    l2 = ax1.plot(ts, equivs, "s--", color=color2, lw=2, label="Equivalent Action Agreement (%)")
    ax1.set_xlabel("Planning Horizon T (years)", fontsize=11)
    ax1.set_ylabel("Year-0 Policy Agreement with PI (%)", color=color1, fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    l3 = ax2.plot(ts, gammas, "^-.", color=color3, lw=1.8, label=r"Contraction Bound $\gamma^T$")
    ax2.set_ylabel(r"Discount Factor $\gamma^T$", color=color3, fontsize=11)
    ax2.set_yscale("log")

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=9)
    plt.title(f"Backward Induction Horizon Convergence to Stationary Policy (n_V={p.n_V})", fontsize=12, fontweight="bold")
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
        default="5,10,20,40,80,160",
        help="Comma-separated horizon values for sweep (default: 5,10,20,40,80,160)",
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

    print(f"\n2. Solving Backward Induction (T={args.T} years)...")
    bi_sol = backward_induction(p, T=args.T, g=pi_sol.grids)
    print(f"   Completed {bi_sol.iterations} stages in {bi_sol.seconds:.2f}s.")

    cmp_res = compare_policies(bi_sol, pi_sol, ref=pi_sol)
    print("\n3. Year-0 Policy Comparison (BI T=10 vs Stationary PI):")
    print(f"   - Exact action agreement:     {cmp_res['exact_agreement_pct']:.2f}%")
    print(f"   - Equivalent-action agreement: {cmp_res['equivalent_action_pct']:.2f}%")
    print(f"   - Mean |delta action|:         {cmp_res['mean_abs_action_diff']:.4f}")
    print(f"   - Max |V_0^(T) - V_PI|:        {float(np.max(np.abs(bi_sol.value - pi_sol.value))):.4e}")
    print(f"   - Theoretical gamma^T:         {p.gamma ** args.T:.4e}")

    # (b) Horizon sweep
    if not args.no_sweep:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        run_horizon_sweep(p, horizons=horizons, out_dir=args.out_dir)

    print("\n" + "=" * 75)
    print("BACKWARD INDUCTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()


