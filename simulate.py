"""Monte Carlo simulation of optimal pension policies over a 10-year horizon.

Simulates 10,000 continuous paths with N(0, 1) shocks and tracks:
- Nominal and relative individual pension trajectories (Paper eq. 5.7).
- Probabilities of pension reductions and enhancements.
- Distributional statistics (Mean, Std, 5% and 1% VaR) for relative pension and wealth.
- Comparison against Tables 1 and 2 in Lichtenstern & Zagst (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from grids import Grids
from transitions import continuous_transition_step

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Stores full trajectory tensors and aggregated summary metrics."""

    alpha: float
    n_paths: int
    n_years: int
    V_paths: np.ndarray  # Shape: (n_paths, n_years + 1)
    P_paths: np.ndarray  # Shape: (n_paths, n_years + 1)
    CCR_paths: np.ndarray  # Shape: (n_paths, n_years + 1)
    P_rel_paths: np.ndarray  # Shape: (n_paths, n_years + 1)
    V_rel_paths: np.ndarray  # Shape: (n_paths, n_years + 1)
    actions_taken: np.ndarray  # Shape: (n_paths, n_years)
    summary_metrics: dict[str, float]


def run_monte_carlo_simulation(
    grids: Grids,
    optimal_policy_actions: np.ndarray,
    n_paths: int = 10000,
    n_years: int = 10,
    seed: int = 42,
) -> SimulationResults:
    """Run Monte Carlo simulation using the learned optimal policy.

    Parameters:
        grids: GridSystem instance.
        optimal_policy_actions: Array of optimal actions a*(s) for each grid node s.
        n_paths: Number of simulation paths (default 10,000).
        n_years: Decumulation horizon in years (default 10).
        seed: Random seed for reproducibility.
    """
    config = grids.config
    rng = np.random.default_rng(seed)

    # Initial state (Paper Section 5)
    V0 = config.V0
    P0 = config.initial_pension()
    E0 = P0 / config.r_plus_lambda
    CCR0 = V0 / E0

    # Edge Case 12: Starting state boundary warning
    v_range = config.V_max - config.V_min
    dist_to_v_min = (V0 - config.V_min) / v_range
    dist_to_v_max = (config.V_max - V0) / v_range
    if dist_to_v_min < 0.05 or dist_to_v_max < 0.05:
        logger.warning(
            "Initial wealth V0=%.1f is within 5%% of grid boundaries! (min dist: %.1f%%)",
            V0,
            min(dist_to_v_min, dist_to_v_max) * 100,
        )

    # Preallocate trajectory arrays: shape (n_paths, n_years + 1)
    V_paths = np.empty((n_paths, n_years + 1), dtype=np.float64)
    P_paths = np.empty((n_paths, n_years + 1), dtype=np.float64)
    CCR_paths = np.empty((n_paths, n_years + 1), dtype=np.float64)
    actions_taken = np.empty((n_paths, n_years), dtype=np.float64)

    V_paths[:, 0] = V0
    P_paths[:, 0] = P0
    CCR_paths[:, 0] = CCR0

    v_min_touch_count = 0
    v_max_touch_count = 0

    # Draw continuous standard normal shocks Z ~ N(0, 1)
    # Shape: (n_paths, n_years)
    shocks = rng.standard_normal((n_paths, n_years))

    for t in range(n_years):
        current_V = V_paths[:, t]
        current_CCR = CCR_paths[:, t]
        current_P = P_paths[:, t]

        # Look up nearest grid node for each path's continuous (V, CCR)
        _, _, s_idx = grids.snap_to_nearest_node(current_V, current_CCR)
        a_t = optimal_policy_actions[s_idx]
        actions_taken[:, t] = a_t

        Z_t = shocks[:, t]

        # Continuous transition step (Paper eq. 5.4, 5.5, 5.12, 5.19)
        V_next, P_next, CCR_next = continuous_transition_step(
            V=current_V,
            P=current_P,
            a=a_t,
            Z=Z_t,
            config=config,
        )

        V_paths[:, t + 1] = V_next
        P_paths[:, t + 1] = P_next
        CCR_paths[:, t + 1] = CCR_next

        # Check boundary encounters
        v_min_touch_count += int(np.sum(np.isclose(V_next, config.V_min)))
        v_max_touch_count += int(np.sum(np.isclose(V_next, config.V_max)))

    if v_min_touch_count > 0 or v_max_touch_count > 0:
        logger.info(
            "Simulation (alpha=%.2f): Boundary encounters - V_min touched %d times, V_max %d times.",
            config.alpha,
            v_min_touch_count,
            v_max_touch_count,
        )

    # Cohort rule & relative individual pension (Paper eq. 5.7):
    # P_rel(t) = P(t) / (exp(-lambda_x * t) * P0)
    time_steps = np.arange(n_years + 1, dtype=np.float64)
    mortality_survival = np.exp(-config.lambda_x * time_steps)  # Shape: (n_years + 1,)
    P_rel_paths = P_paths / (mortality_survival * P0)

    # Relative wealth: V(t) / V0
    V_rel_paths = V_paths / V0

    # Compute Paper Metrics:
    # 1. P(at least one pension reduction)
    # A reduction occurs if P(t+1) < P(t) - 1e-6
    p_changes = P_paths[:, 1:] - P_paths[:, :-1]
    path_has_reduction = np.any(p_changes < -1e-6, axis=1)
    prob_reduction = float(np.mean(path_has_reduction))

    # 2. P(path-average relative pension >= 1.0) and P(path-average pension >= P0)
    path_avg_P_rel = np.mean(P_rel_paths, axis=1)
    prob_avg_pension_ge_P0 = float(np.mean(path_avg_P_rel >= 1.0))

    # 3. P(#enhancements >= #reductions)
    path_num_enhancements = np.sum(p_changes > 1e-6, axis=1)
    path_num_reductions = np.sum(p_changes < -1e-6, axis=1)
    prob_enh_ge_red = float(np.mean(path_num_enhancements >= path_num_reductions))

    # 4. Relative pension at horizon T = 10 years
    final_P_rel = P_rel_paths[:, -1]
    p_rel_mean = float(np.mean(final_P_rel))
    p_rel_std = float(np.std(final_P_rel, ddof=1))
    p_rel_var5 = float(np.percentile(final_P_rel, 5.0))
    p_rel_var1 = float(np.percentile(final_P_rel, 1.0))

    # 5. Relative wealth at horizon T = 10 years
    final_V_rel = V_rel_paths[:, -1]
    v_rel_mean = float(np.mean(final_V_rel))
    v_rel_std = float(np.std(final_V_rel, ddof=1))

    summary = {
        "alpha": config.alpha,
        "P0": P0,
        "prob_at_least_one_reduction": prob_reduction,
        "prob_avg_pension_ge_P0": prob_avg_pension_ge_P0,
        "prob_enhancements_ge_reductions": prob_enh_ge_red,
        "relative_pension_mean_t10": p_rel_mean,
        "relative_pension_std_t10": p_rel_std,
        "relative_pension_var5_t10": p_rel_var5,
        "relative_pension_var1_t10": p_rel_var1,
        "relative_wealth_mean_t10": v_rel_mean,
        "relative_wealth_std_t10": v_rel_std,
        "mean_risky_asset_weight": float(np.mean(actions_taken)),
    }

    return SimulationResults(
        alpha=config.alpha,
        n_paths=n_paths,
        n_years=n_years,
        V_paths=V_paths,
        P_paths=P_paths,
        CCR_paths=CCR_paths,
        P_rel_paths=P_rel_paths,
        V_rel_paths=V_rel_paths,
        actions_taken=actions_taken,
        summary_metrics=summary,
    )


def simulate_market_scenarios(
    grids: Grids,
    optimal_policy_actions: np.ndarray,
    n_years: int = 10,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate deterministic stylized market regimes: Bull, Bear, and Sideways.

    Used to reproduce Figures 2-5 of the paper:
    - Bull: Strong sustained returns (Z = +1.0)
    - Bear: Severe market downturn (Z = -1.0)
    - Sideways: Neutral zero-shock market (Z = 0.0)
    """
    config = grids.config
    scenarios = {
        "Bull (+1.0 sigma)": np.full(n_years, 1.0),
        "Bear (-1.0 sigma)": np.full(n_years, -1.0),
        "Sideways (0.0 sigma)": np.zeros(n_years),
    }

    results: dict[str, dict[str, np.ndarray]] = {}
    time_steps = np.arange(n_years + 1, dtype=np.float64)
    mortality_survival = np.exp(-config.lambda_x * time_steps)
    P0 = config.initial_pension()

    for name, shock_seq in scenarios.items():
        V_seq = [config.V0]
        P_seq = [P0]
        E0 = P0 / config.r_plus_lambda
        CCR_seq = [config.V0 / E0]
        a_seq = []

        for t in range(n_years):
            V_curr = V_seq[-1]
            CCR_curr = CCR_seq[-1]
            P_curr = P_seq[-1]

            _, _, s_idx = grids.snap_to_nearest_node(V_curr, CCR_curr)
            a_t = float(optimal_policy_actions[s_idx])
            a_seq.append(a_t)

            V_next, P_next, CCR_next = continuous_transition_step(
                V=V_curr,
                P=P_curr,
                a=a_t,
                Z=shock_seq[t],
                config=config,
            )
            V_seq.append(V_next)
            P_seq.append(P_next)
            CCR_seq.append(CCR_next)

        P_arr = np.array(P_seq)
        V_arr = np.array(V_seq)
        P_rel_arr = P_arr / (mortality_survival * P0)
        V_rel_arr = V_arr / config.V0

        results[name] = {
            "years": time_steps,
            "wealth": V_arr,
            "pension": P_arr,
            "ccr": np.array(CCR_seq),
            "pension_rel": P_rel_arr,
            "wealth_rel": V_rel_arr,
            "actions": np.array(a_seq),
        }

    return results


def format_comparison_table(
    sim_summaries: list[dict[str, float]],
    mode: str,
) -> pd.DataFrame:
    """Format simulated metrics alongside published values from Lichtenstern & Zagst (2022)."""
    # Paper reference benchmarks from Lichtenstern & Zagst (2022) Tables 1 and 2
    # Note: Paper values reported under r+lambda=0.0312
    paper_reference = {
        0.0: {
            "P0": 277,
            "P(at least 1 reduction)": 0.495,
            "P(enh >= red)": 0.844,
            "P_rel Mean": 1.075,
            "P_rel Std": 0.181,
            "P_rel VaR 5%": 0.833,
            "P_rel VaR 1%": 0.719,
            "V_rel Mean": 0.966,
            "V_rel Std": 0.157,
        },
        0.2: {
            "P0": 270,
            "P(at least 1 reduction)": 0.364,
            "P(enh >= red)": 0.913,
            "P_rel Mean": 1.084,
            "P_rel Std": 0.168,
            "P_rel VaR 5%": 0.848,
            "P_rel VaR 1%": 0.751,
            "V_rel Mean": 0.962,
            "V_rel Std": 0.143,
        },
        0.4: {
            "P0": 258,
            "P(at least 1 reduction)": 0.255,
            "P(enh >= red)": 0.968,
            "P_rel Mean": 1.109,
            "P_rel Std": 0.170,
            "P_rel VaR 5%": 0.865,
            "P_rel VaR 1%": 0.770,
            "V_rel Mean": 0.965,
            "V_rel Std": 0.146,
        },
    }

    rows = []
    for s in sim_summaries:
        a = s["alpha"]
        ref = paper_reference.get(round(a, 1), {})
        rows.append({
            "alpha": a,
            "Sim P0": round(s["P0"], 2),
            "Paper P0": ref.get("P0", np.nan),
            "Sim P(Red >= 1)": f"{s['prob_at_least_one_reduction'] * 100:.1f}%",
            "Paper P(Red >= 1)": f"{ref.get('P(at least 1 reduction)', np.nan) * 100:.1f}%",
            "Sim P(Enh >= Red)": f"{s['prob_enhancements_ge_reductions'] * 100:.1f}%",
            "Paper P(Enh >= Red)": f"{ref.get('P(enh >= red)', np.nan) * 100:.1f}%",
            "Sim P_rel Mean": round(s["relative_pension_mean_t10"], 3),
            "Paper P_rel Mean": ref.get("P_rel Mean", np.nan),
            "Sim P_rel Std": round(s["relative_pension_std_t10"], 3),
            "Paper P_rel Std": ref.get("P_rel Std", np.nan),
            "Sim P_rel VaR 5%": round(s["relative_pension_var5_t10"], 3),
            "Paper P_rel VaR 5%": ref.get("P_rel VaR 5%", np.nan),
            "Sim P_rel VaR 1%": round(s["relative_pension_var1_t10"], 3),
            "Paper P_rel VaR 1%": ref.get("P_rel VaR 1%", np.nan),
            "Sim V_rel Mean": round(s["relative_wealth_mean_t10"], 3),
            "Paper V_rel Mean": ref.get("V_rel Mean", np.nan),
        })

    df = pd.DataFrame(rows)
    return df
