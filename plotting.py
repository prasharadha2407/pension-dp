"""Visualization module for the pension-dp project.

Generates publication-quality figures reproducing Lichtenstern & Zagst (2022):
- Fig. 1: Average optimal investment strategy a* vs CCR for alpha in {0.0, 0.2, 0.4}.
- Figs. 2-5: Stylized market scenario trajectories (Bull, Bear, Sideways) for Pension, Wealth, CCR.
- Fig. 8: Kernel density estimation of path-average relative pension and relative wealth.
- Runtime and convergence performance plots.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from grids import Grids
from simulate import SimulationResults, simulate_market_scenarios

logger = logging.getLogger(__name__)

# Modern publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
})

COLOR_MAP = {
    0.0: "#1f77b4",  # Blue
    0.2: "#ff7f0e",  # Orange
    0.4: "#2ca02c",  # Green
}


def plot_fig1_policy_vs_ccr(
    grids: Grids,
    policies_by_alpha: dict[float, np.ndarray],
    output_dir: str = "results",
) -> str:
    """Reproduce Figure 1: Average optimal action a* versus CCR for alpha in {0.0, 0.2, 0.4}.

    For each CCR node j in [0, n_CCR-1], computes the average optimal risky allocation
    across all wealth grid levels V_i:
    bar_a*(CCR_j) = (1 / n_V) sum_{i=0}^{n_V-1} a*(V_i, CCR_j).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_V = grids.n_V
    n_CCR = grids.n_CCR
    ccr_vals = grids.CCR_grid

    plt.figure(figsize=(7, 5))

    for alpha, policy_actions in sorted(policies_by_alpha.items()):
        # Reshape policy actions to (n_V, n_CCR)
        # s = i * n_CCR + j
        policy_matrix = policy_actions.reshape((n_V, n_CCR))
        avg_action_by_ccr = np.mean(policy_matrix, axis=0)

        color = COLOR_MAP.get(round(alpha, 1), "#333333")
        plt.plot(
            ccr_vals,
            avg_action_by_ccr,
            "o-",
            label=rf"$\alpha = {alpha:.1f}$",
            color=color,
            markersize=4,
        )

    plt.xlabel("Capital Coverage Ratio (CCR)", fontweight="bold")
    plt.ylabel(r"Average Optimal Risky Asset Share $\bar{a}^*$", fontweight="bold")
    plt.title(r"Optimal Investment Strategy vs. CCR (Paper Fig. 1)", fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(-0.02, 1.02)
    plt.xlim(1.00, 1.25)
    plt.legend(frameon=True, loc="upper left")

    out_path = os.path.join(output_dir, "fig1_optimal_policy_vs_ccr.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 1 to %s", out_path)
    return out_path


def plot_scenario_trajectories(
    grids: Grids,
    policies_by_alpha: dict[float, np.ndarray],
    output_dir: str = "results",
) -> list[str]:
    """Reproduce Figures 2-5: Trajectories of Pension, Wealth, and CCR under Bull, Bear, and Sideways regimes."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    regimes = [
        ("Bull Market (+1.0 standard dev)", 1.0, "fig2_scenario_bull.png"),
        ("Bear Market (-1.0 standard dev)", -1.0, "fig3_scenario_bear.png"),
        ("Sideways Market (0.0 standard dev)", 0.0, "fig4_scenario_sideways.png"),
    ]

    for regime_name, shock_val, filename in regimes:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        ax_p, ax_w, ax_c = axes

        for alpha, policy_actions in sorted(policies_by_alpha.items()):
            color = COLOR_MAP.get(round(alpha, 1), "#333333")
            scenarios = simulate_market_scenarios(grids, policy_actions, n_years=10)

            # Pick matching scenario key
            sc_key = [k for k in scenarios if f"{shock_val:+.1f}" in k or (shock_val == 0.0 and "0.0" in k)][0]
            data = scenarios[sc_key]
            t = data["years"]

            # Panel 1: Relative Individual Pension P_rel(t)
            ax_p.plot(t, data["pension_rel"], "o-", color=color, label=rf"$\alpha = {alpha:.1f}$", markersize=4)

            # Panel 2: Relative Wealth V(t) / V0
            ax_w.plot(t, data["wealth_rel"], "s-", color=color, label=rf"$\alpha = {alpha:.1f}$", markersize=4)

            # Panel 3: CCR(t)
            ax_c.plot(t, data["ccr"], "^-", color=color, label=rf"$\alpha = {alpha:.1f}$", markersize=4)

        # Formatting Panel 1: Pension
        ax_p.set_ylabel("Relative Pension $P_{rel}(t)$", fontweight="bold")
        ax_p.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
        ax_p.grid(True, linestyle="--", alpha=0.5)
        ax_p.legend(loc="best", frameon=True)
        ax_p.set_title(f"Scenario Trajectories: {regime_name}", fontweight="bold", pad=8)

        # Formatting Panel 2: Wealth
        ax_w.set_ylabel("Relative Wealth $V(t) / V_0$", fontweight="bold")
        ax_w.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
        ax_w.grid(True, linestyle="--", alpha=0.5)

        # Formatting Panel 3: CCR
        ax_c.set_ylabel("CCR", fontweight="bold")
        ax_c.set_xlabel("Time (years)", fontweight="bold")
        ax_c.axhline(1.00, color="red", linestyle="--", alpha=0.6, label="Lower Bound (1.00)")
        ax_c.axhline(1.25, color="green", linestyle="--", alpha=0.6, label="Upper Bound (1.25)")
        ax_c.set_ylim(0.95, 1.30)
        ax_c.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        generated_files.append(out_path)
        logger.info("Saved scenario figure to %s", out_path)

    return generated_files


def plot_fig8_densities(
    sim_results_list: Sequence[SimulationResults],
    output_dir: str = "results",
) -> str:
    """Reproduce Figure 8: Kernel density of path-average relative pension and relative wealth."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for res in sim_results_list:
        alpha = res.alpha
        color = COLOR_MAP.get(round(alpha, 1), "#333333")

        # 1. Path-average relative pension across the 10-year path
        path_avg_p_rel = np.mean(res.P_rel_paths, axis=1)
        kde_p = stats.gaussian_kde(path_avg_p_rel)
        x_p = np.linspace(
            np.percentile(path_avg_p_rel, 0.1),
            np.percentile(path_avg_p_rel, 99.9),
            300,
        )
        ax1.plot(x_p, kde_p(x_p), linewidth=2, color=color, label=rf"$\alpha = {alpha:.1f}$")
        ax1.fill_between(x_p, kde_p(x_p), alpha=0.15, color=color)

        # 2. Path-average relative wealth across the 10-year path
        path_avg_v_rel = np.mean(res.V_rel_paths, axis=1)
        kde_v = stats.gaussian_kde(path_avg_v_rel)
        x_v = np.linspace(
            np.percentile(path_avg_v_rel, 0.1),
            np.percentile(path_avg_v_rel, 99.9),
            300,
        )
        ax2.plot(x_v, kde_v(x_v), linewidth=2, color=color, label=rf"$\alpha = {alpha:.1f}$")
        ax2.fill_between(x_v, kde_v(x_v), alpha=0.15, color=color)

    # Panel 1: Relative Pension Density
    ax1.set_xlabel("Path-Average Relative Pension", fontweight="bold")
    ax1.set_ylabel("Kernel Density", fontweight="bold")
    ax1.set_title("Density of Path-Average Relative Pension (Paper Fig. 8 Left)", fontweight="bold")
    ax1.axvline(1.0, color="gray", linestyle=":", alpha=0.8, label="Initial Relative (1.0)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper right")

    # Panel 2: Relative Wealth Density
    ax2.set_xlabel("Path-Average Relative Wealth $V / V_0$", fontweight="bold")
    ax2.set_ylabel("Kernel Density", fontweight="bold")
    ax2.set_title("Density of Path-Average Relative Wealth (Paper Fig. 8 Right)", fontweight="bold")
    ax2.axvline(1.0, color="gray", linestyle=":", alpha=0.8, label="Initial Wealth (1.0)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig8_density_pension_wealth.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 8 to %s", out_path)
    return out_path
