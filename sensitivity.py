"""Parameter and Discretization Sensitivity Analysis.

Investigates the sensitivity of optimal investment strategies and decumulation
performance across key model parameters and numerical grid resolutions:
1. Buffer allocation parameter alpha in [0.0, 0.4]
2. Risky asset volatility sigma in [0.08, 0.25]
3. Risky asset drift mu in [0.02, 0.06]
4. Discretization grid density (n_V, n_z, n_a)

Generates CSV tables and publication-quality figures under results/.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pension_core import Params, metrics, policy_iteration, simulate


def sweep_alpha(
    base_p: Params | None = None,
    alphas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    n_paths: int = 10_000,
    T: int = 10,
    seed: int = 42,
    out_dir: str = "results",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Analyze sensitivity to buffer portfolio weight alpha."""
    if base_p is None:
        base_p = Params(n_V=200)

    rows = []
    sols = {}
    ress = {}

    for alpha in alphas:
        p = replace(base_p, alpha=alpha)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        sol_time = time.perf_counter() - t0
        res = simulate(p, sol, n_paths=n_paths, T=T, seed=seed)
        m = metrics(res, p)

        sols[alpha] = sol
        ress[alpha] = res

        avg_action = float(res.action.mean())
        pol_grid = sol.policy_grid()
        mean_grid_action = float(pol_grid.mean()) * (p.a_max - p.a_min) / (p.n_a - 1)

        row = {
            "alpha": alpha,
            "P0": p.P0(),
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(sol_time, 3),
            "P_cut_prob": m["P(>=1 pension cut)"],
            "P_enh_ge_cuts": m["P(#enh >= #cuts)"],
            "P_avg_gt_P0": m["P(avg pension > P0)"],
            "rel_pension_mean_T": m["Rel. pension mean (T)"],
            "rel_pension_std_T": m["Rel. pension std (T)"],
            "rel_pension_VaR_5pct": m["Rel. pension VaR 5%"],
            "rel_pension_VaR_1pct": m["Rel. pension VaR 1%"],
            "rel_wealth_mean_T": m["Rel. wealth mean (T)"],
            "rel_wealth_std_T": m["Rel. wealth std (T)"],
            "sim_mean_action": avg_action,
            "grid_mean_action": mean_grid_action,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_alpha.csv")
    df.to_csv(csv_path, index=False)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    fig.suptitle(
        f"Sensitivity to Buffer Weight alpha (T={T}yr, {n_paths:,} paths)",
        fontsize=13,
        fontweight="bold",
    )

    # [0, 0] Initial pension vs Alpha
    ax = axs[0, 0]
    ax.plot(df["alpha"], df["P0"], "-o", color="#1f77b4", lw=2, ms=6)
    ax.set_xlabel("Buffer weight alpha")
    ax.set_ylabel("Initial Pension P0")
    ax.set_title("Initial Guaranteed Pension Baseline")
    ax.grid(alpha=0.3)

    # [0, 1] Downside risk
    ax = axs[0, 1]
    ax.plot(df["alpha"], df["P_cut_prob"], "-o", color="#d62728", lw=2, ms=6, label="P(>=1 cut)")
    ax.plot(df["alpha"], df["rel_pension_VaR_5pct"], "-s", color="#ff7f0e", lw=2, ms=6, label="5% VaR (terminal)")
    ax.plot(df["alpha"], df["rel_pension_VaR_1pct"], "-^", color="#8c564b", lw=2, ms=6, label="1% VaR (terminal)")
    ax.set_xlabel("Buffer weight alpha")
    ax.set_ylabel("Probability / Relative Ratio")
    ax.set_title("Downside Pension Protection")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 0] Terminal relative wealth and pension
    ax = axs[1, 0]
    ax.plot(df["alpha"], df["rel_pension_mean_T"], "-o", color="#2ca02c", lw=2, ms=6, label="E[P_T / P_0*]")
    ax.plot(df["alpha"], df["rel_wealth_mean_T"], "-s", color="#17becf", lw=2, ms=6, label="E[V_T / V_0]")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Buffer weight alpha")
    ax.set_ylabel("Relative Ratio")
    ax.set_title("Terminal Mean Expectations (T=10)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 1] Risky investment share
    ax = axs[1, 1]
    ax.plot(df["alpha"], df["sim_mean_action"], "-o", color="#9467bd", lw=2, ms=6, label="Mean simulated a*")
    ax.plot(df["alpha"], df["grid_mean_action"], "--s", color="#8c564b", lw=1.5, ms=5, label="Grid-average a*")
    ax.set_xlabel("Buffer weight alpha")
    ax.set_ylabel("Optimal risky share a*")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Optimal Investment Portfolio Share")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "sensitivity_alpha.png")
    fig.savefig(fig_path)
    return df, fig


def sweep_sigma(
    base_p: Params | None = None,
    sigmas: tuple[float, ...] = (0.08, 0.1175, 0.15, 0.20, 0.25),
    n_paths: int = 10_000,
    T: int = 10,
    seed: int = 42,
    out_dir: str = "results",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Analyze sensitivity to risky asset volatility sigma."""
    if base_p is None:
        base_p = Params(n_V=200)

    rows = []
    for sigma in sigmas:
        p = replace(base_p, sigma=sigma)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        sol_time = time.perf_counter() - t0
        res = simulate(p, sol, n_paths=n_paths, T=T, seed=seed)
        m = metrics(res, p)

        avg_action = float(res.action.mean())
        pol_grid = sol.policy_grid()
        mean_grid_action = float(pol_grid.mean()) * (p.a_max - p.a_min) / (p.n_a - 1)

        row = {
            "sigma": sigma,
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(sol_time, 3),
            "P_cut_prob": m["P(>=1 pension cut)"],
            "P_enh_ge_cuts": m["P(#enh >= #cuts)"],
            "rel_pension_mean_T": m["Rel. pension mean (T)"],
            "rel_pension_std_T": m["Rel. pension std (T)"],
            "rel_pension_VaR_5pct": m["Rel. pension VaR 5%"],
            "rel_pension_VaR_1pct": m["Rel. pension VaR 1%"],
            "rel_wealth_mean_T": m["Rel. wealth mean (T)"],
            "rel_wealth_std_T": m["Rel. wealth std (T)"],
            "sim_mean_action": avg_action,
            "grid_mean_action": mean_grid_action,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_sigma.csv")
    df.to_csv(csv_path, index=False)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    fig.suptitle(
        f"Sensitivity to Market Volatility sigma (T={T}yr, {n_paths:,} paths)",
        fontsize=13,
        fontweight="bold",
    )

    # [0, 0] Risky investment flight-to-safety
    ax = axs[0, 0]
    ax.plot(df["sigma"], df["sim_mean_action"], "-o", color="#1f77b4", lw=2, ms=6, label="Simulated a*")
    ax.plot(df["sigma"], df["grid_mean_action"], "--s", color="#ff7f0e", lw=1.5, ms=5, label="Grid-average a*")
    ax.set_xlabel("Risky asset volatility sigma")
    ax.set_ylabel("Optimal risky allocation a*")
    ax.set_title("Flight to Safety under Volatility")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [0, 1] Downside risk VaR
    ax = axs[0, 1]
    ax.plot(df["sigma"], df["rel_pension_VaR_5pct"], "-s", color="#d62728", lw=2, ms=6, label="5% VaR")
    ax.plot(df["sigma"], df["rel_pension_VaR_1pct"], "-^", color="#8c564b", lw=2, ms=6, label="1% VaR")
    ax.plot(df["sigma"], df["P_cut_prob"], "-o", color="#9467bd", lw=2, ms=6, label="P(>=1 cut)")
    ax.set_xlabel("Risky asset volatility sigma")
    ax.set_ylabel("Ratio / Probability")
    ax.set_title("Downside Risk vs Market Volatility")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 0] Pension dispersion
    ax = axs[1, 0]
    ax.plot(df["sigma"], df["rel_pension_mean_T"], "-o", color="#2ca02c", lw=2, ms=6, label="Mean rel. pension")
    ax.plot(df["sigma"], df["rel_pension_std_T"], "--s", color="#e377c2", lw=2, ms=6, label="Std rel. pension")
    ax.set_xlabel("Risky asset volatility sigma")
    ax.set_ylabel("Relative pension units")
    ax.set_title("Pension Uncertainty at Horizon T")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 1] Wealth trajectory
    ax = axs[1, 1]
    ax.plot(df["sigma"], df["rel_wealth_mean_T"], "-o", color="#17becf", lw=2, ms=6, label="Mean rel. wealth")
    ax.plot(df["sigma"], df["rel_wealth_std_T"], "--s", color="#7f7f7f", lw=2, ms=6, label="Std rel. wealth")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Risky asset volatility sigma")
    ax.set_ylabel("Relative cohort wealth")
    ax.set_title("Cohort Wealth Dispersion (T=10)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "sensitivity_sigma.png")
    fig.savefig(fig_path)
    return df, fig


def sweep_mu(
    base_p: Params | None = None,
    mus: tuple[float, ...] = (0.02, 0.0297, 0.04, 0.05, 0.06),
    n_paths: int = 10_000,
    T: int = 10,
    seed: int = 42,
    out_dir: str = "results",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Analyze sensitivity to risky asset drift mu (expected return)."""
    if base_p is None:
        base_p = Params(n_V=200)

    rows = []
    for mu in mus:
        p = replace(base_p, mu=mu)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        sol_time = time.perf_counter() - t0
        res = simulate(p, sol, n_paths=n_paths, T=T, seed=seed)
        m = metrics(res, p)

        avg_action = float(res.action.mean())
        pol_grid = sol.policy_grid()
        mean_grid_action = float(pol_grid.mean()) * (p.a_max - p.a_min) / (p.n_a - 1)

        row = {
            "mu": mu,
            "risk_premium": mu - p.r,
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(sol_time, 3),
            "P_cut_prob": m["P(>=1 pension cut)"],
            "P_enh_ge_cuts": m["P(#enh >= #cuts)"],
            "P_avg_gt_P0": m["P(avg pension > P0)"],
            "rel_pension_mean_T": m["Rel. pension mean (T)"],
            "rel_pension_std_T": m["Rel. pension std (T)"],
            "rel_pension_VaR_5pct": m["Rel. pension VaR 5%"],
            "rel_wealth_mean_T": m["Rel. wealth mean (T)"],
            "rel_wealth_std_T": m["Rel. wealth std (T)"],
            "sim_mean_action": avg_action,
            "grid_mean_action": mean_grid_action,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_mu.csv")
    df.to_csv(csv_path, index=False)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    fig.suptitle(
        f"Sensitivity to Risky Asset Drift mu (r={base_p.r}, T={T}yr)",
        fontsize=13,
        fontweight="bold",
    )

    # [0, 0] Equity allocation vs risk premium
    ax = axs[0, 0]
    ax.plot(df["mu"], df["sim_mean_action"], "-o", color="#1f77b4", lw=2, ms=6, label="Simulated a*")
    ax.plot(df["mu"], df["grid_mean_action"], "--s", color="#ff7f0e", lw=1.5, ms=5, label="Grid-average a*")
    ax.set_xlabel("Risky asset drift mu")
    ax.set_ylabel("Optimal risky share a*")
    ax.set_title("Equity Demand vs Expected Return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [0, 1] Pension growth and upside
    ax = axs[0, 1]
    ax.plot(df["mu"], df["rel_pension_mean_T"], "-o", color="#2ca02c", lw=2, ms=6, label="Mean rel. pension")
    ax.plot(df["mu"], df["P_enh_ge_cuts"], "-s", color="#17becf", lw=2, ms=6, label="P(#enh >= #cuts)")
    ax.set_xlabel("Risky asset drift mu")
    ax.set_ylabel("Ratio / Probability")
    ax.set_title("Pension Growth and Enhancement Frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 0] Downside cut mitigation
    ax = axs[1, 0]
    ax.plot(df["mu"], df["P_cut_prob"], "-o", color="#d62728", lw=2, ms=6, label="P(>=1 cut)")
    ax.plot(df["mu"], df["rel_pension_VaR_5pct"], "-^", color="#9467bd", lw=2, ms=6, label="5% VaR (terminal)")
    ax.set_xlabel("Risky asset drift mu")
    ax.set_ylabel("Probability / Ratio")
    ax.set_title("Cut Risk Mitigation via Market Growth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 1] Cohort wealth build-up
    ax = axs[1, 1]
    ax.plot(df["mu"], df["rel_wealth_mean_T"], "-o", color="#8c564b", lw=2, ms=6, label="Mean rel. wealth")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Risky asset drift mu")
    ax.set_ylabel("E[V_T / V_0]")
    ax.set_title("Cohort Wealth Generation (T=10)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "sensitivity_mu.png")
    fig.savefig(fig_path)
    return df, fig


def sweep_grid(
    base_p: Params | None = None,
    n_vs: tuple[int, ...] = (50, 100, 200, 400),
    n_zs: tuple[int, ...] = (10, 20, 40, 80),
    n_as: tuple[int, ...] = (11, 21, 41),
    out_dir: str = "results",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Analyze convergence and sensitivity to grid discretization parameters."""
    if base_p is None:
        base_p = Params()

    # Fixed representative evaluation points: (V, ccr)
    eval_points = [
        ("V0_pbar", base_p.V0, base_p.p_bar),
        ("lowV_lowCCR", 5000.0, 1.02),
        ("highV_highCCR", 15000.0, 1.22),
    ]

    rows = []

    # 1. Sweep n_V (with default n_z=40, n_a=21)
    for nv in n_vs:
        p = replace(base_p, n_V=nv, n_z=40, n_a=21)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        elapsed = time.perf_counter() - t0

        row = {
            "parameter_varied": "n_V",
            "n_V": nv,
            "n_z": 40,
            "n_a": 21,
            "n_S": sol.grids.n_S,
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(elapsed, 3),
        }
        for name, V_val, ccr_val in eval_points:
            row[f"action_{name}"] = round(float(sol.action_at(np.array([V_val]), np.array([ccr_val]))[0]), 3)
        rows.append(row)

    # 2. Sweep n_z (with fixed n_V=200, n_a=21)
    for nz in n_zs:
        if nz == 40:
            continue  # already in rows from n_V sweep
        p = replace(base_p, n_V=200, n_z=nz, n_a=21)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        elapsed = time.perf_counter() - t0

        row = {
            "parameter_varied": "n_z",
            "n_V": 200,
            "n_z": nz,
            "n_a": 21,
            "n_S": sol.grids.n_S,
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(elapsed, 3),
        }
        for name, V_val, ccr_val in eval_points:
            row[f"action_{name}"] = round(float(sol.action_at(np.array([V_val]), np.array([ccr_val]))[0]), 3)
        rows.append(row)

    # 3. Sweep n_a (with fixed n_V=200, n_z=40)
    for na in n_as:
        if na == 21:
            continue  # already in rows
        p = replace(base_p, n_V=200, n_z=40, n_a=na)
        t0 = time.perf_counter()
        sol = policy_iteration(p)
        elapsed = time.perf_counter() - t0

        row = {
            "parameter_varied": "n_a",
            "n_V": 200,
            "n_z": 40,
            "n_a": na,
            "n_S": sol.grids.n_S,
            "PI_iterations": sol.iterations,
            "PI_runtime_sec": round(elapsed, 3),
        }
        for name, V_val, ccr_val in eval_points:
            row[f"action_{name}"] = round(float(sol.action_at(np.array([V_val]), np.array([ccr_val]))[0]), 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_grid.csv")
    df.to_csv(csv_path, index=False)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
    fig.suptitle(
        "Numerical Grid Discretization Sensitivity",
        fontsize=13,
        fontweight="bold",
    )

    # [0, 0] Runtime vs n_V
    ax = axs[0, 0]
    df_nv = df[df["parameter_varied"] == "n_V"].sort_values("n_V")
    ax.plot(df_nv["n_V"], df_nv["PI_runtime_sec"], "-o", color="#1f77b4", lw=2, ms=6)
    for _, r in df_nv.iterrows():
        ax.annotate(f"{r['n_S']:,} states", (r["n_V"], r["PI_runtime_sec"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("Wealth grid points n_V")
    ax.set_ylabel("Policy Iteration Runtime (s)")
    ax.set_title("Runtime Scaling with Wealth Discretization")
    ax.grid(alpha=0.3)

    # [0, 1] Action convergence across n_V
    ax = axs[0, 1]
    ax.plot(df_nv["n_V"], df_nv["action_V0_pbar"], "-o", label="Central (V0, p_bar)", color="#2ca02c", lw=2)
    ax.plot(df_nv["n_V"], df_nv["action_lowV_lowCCR"], "-s", label="Low (V=5k, CCR=102%)", color="#d62728", lw=1.5)
    ax.plot(df_nv["n_V"], df_nv["action_highV_highCCR"], "-^", label="High (V=15k, CCR=122%)", color="#9467bd", lw=1.5)
    ax.set_xlabel("Wealth grid points n_V")
    ax.set_ylabel("Optimal action a*")
    ax.set_title("Policy Stability across Wealth Grid Density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # [1, 0] Runtime vs Shock grid points n_z
    ax = axs[1, 0]
    df_nz = df[(df["parameter_varied"] == "n_z") | ((df["parameter_varied"] == "n_V") & (df["n_V"] == 200))].sort_values("n_z")
    ax.plot(df_nz["n_z"], df_nz["PI_runtime_sec"], "-o", color="#ff7f0e", lw=2, ms=6)
    ax.set_xlabel("Shock quadrature points n_z")
    ax.set_ylabel("Policy Iteration Runtime (s)")
    ax.set_title("Runtime Scaling with Shock Discretization")
    ax.grid(alpha=0.3)

    # [1, 1] Action resolution across Action Choices n_a
    ax = axs[1, 1]
    df_na = df[(df["parameter_varied"] == "n_a") | ((df["parameter_varied"] == "n_V") & (df["n_V"] == 200))].sort_values("n_a")
    ax.plot(df_na["n_a"], df_na["action_V0_pbar"], "-o", color="#2ca02c", lw=2, label="Central (V0, p_bar)")
    ax.plot(df_na["n_a"], df_na["action_lowV_lowCCR"], "-s", color="#d62728", lw=1.5, label="Low (V=5k, CCR=102%)")
    ax.set_xlabel("Action grid choices n_a")
    ax.set_ylabel("Optimal action a*")
    ax.set_title("Policy Resolution across Action Grid Choices")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "sensitivity_grid.png")
    fig.savefig(fig_path)
    return df, fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sensitivity experiments for pension fund dynamic programming model."
    )
    parser.add_argument(
        "--sweep",
        choices=["all", "alpha", "sigma", "mu", "grid"],
        default="all",
        help="Which sensitivity experiment to run (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for generated CSVs and figures (default: results)",
    )
    parser.add_argument(
        "--nV",
        type=int,
        default=200,
        help="Wealth grid points for parameter sweeps (default: 200)",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=10_000,
        help="Monte-Carlo paths for simulation evaluation (default: 10,000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Decumulation horizon in years (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Monte-Carlo simulation (default: 42)",
    )

    args = parser.parse_args()
    base_p = Params(n_V=args.nV)

    print("=" * 65)
    print("PENSION DP MODEL: SENSITIVITY EXPERIMENT SUITE")
    print(f"Target: {args.sweep} | Wealth grid n_V: {args.nV} | Paths: {args.paths:,}")
    print("=" * 65)

    if args.sweep in ("all", "alpha"):
        print("\n--> Running alpha buffer sensitivity sweep...")
        df_a, _ = sweep_alpha(base_p, n_paths=args.paths, T=args.horizon, seed=args.seed, out_dir=args.out_dir)
        print(f"Completed alpha sweep ({len(df_a)} values).")

    if args.sweep in ("all", "sigma"):
        print("\n--> Running sigma volatility sensitivity sweep...")
        df_s, _ = sweep_sigma(base_p, n_paths=args.paths, T=args.horizon, seed=args.seed, out_dir=args.out_dir)
        print(f"Completed sigma sweep ({len(df_s)} values).")

    if args.sweep in ("all", "mu"):
        print("\n--> Running mu drift sensitivity sweep...")
        df_m, _ = sweep_mu(base_p, n_paths=args.paths, T=args.horizon, seed=args.seed, out_dir=args.out_dir)
        print(f"Completed mu sweep ({len(df_m)} values).")

    if args.sweep in ("all", "grid"):
        print("\n--> Running numerical grid discretization sweep...")
        df_g, _ = sweep_grid(base_p, out_dir=args.out_dir)
        print(f"Completed grid sweep ({len(df_g)} configurations).")

    print("\nAll requested sensitivity experiments completed successfully.")


if __name__ == "__main__":
    main()
