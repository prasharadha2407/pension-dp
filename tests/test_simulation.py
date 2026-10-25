"""Unit tests for simulate.py: Monte Carlo mechanics, cohort rule, and scenario simulations."""

import math
import numpy as np
import pytest

from config import Config
from grids import create_grids
from policy_iteration import run_policy_iteration
from simulate import run_monte_carlo_simulation, simulate_market_scenarios
from transitions import precompute_transitions


def test_simulation_seed_reproducibility() -> None:
    """Edge Case 13: Reproducibility across random seeds."""
    cfg = Config(n_V=30, n_CCR=5, n_actions=5, n_Z=10, alpha=0.0)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids)
    pi_res = run_policy_iteration(grids, trans.next_state, verbose=False)

    sim1 = run_monte_carlo_simulation(grids, pi_res.policy_actions, n_paths=100, n_years=5, seed=42)
    sim2 = run_monte_carlo_simulation(grids, pi_res.policy_actions, n_paths=100, n_years=5, seed=42)

    assert np.allclose(sim1.V_paths, sim2.V_paths)
    assert np.allclose(sim1.P_paths, sim2.P_paths)
    assert np.allclose(sim1.P_rel_paths, sim2.P_rel_paths)


def test_cohort_rule_relative_pension_paper_eq_5_7() -> None:
    """Paper eq. (5.7): Relative individual pension P_rel(t) = P(t) / (exp(-lambda_x * t) * P0).

    When CCR stays inside the corridor and P(t) remains constant at P0,
    the relative individual pension grows at rate exp(lambda_x * t) >= 1.0 due to mortality credit.
    """
    cfg = Config(alpha=0.0)
    grids = create_grids(cfg)
    policy_dummy = np.zeros(grids.n_S)  # Pure safe asset

    # Sideways scenario with zero shocks keeps CCR within corridor
    scenarios = simulate_market_scenarios(grids, policy_dummy, n_years=5)
    sideways = scenarios["Sideways (0.0 sigma)"]

    p_rel = sideways["pension_rel"]
    # Check that initial relative pension is exactly 1.0
    assert math.isclose(p_rel[0], 1.0, rel_tol=1e-9)

    # If no reset occurred, p_rel[t] = exp(lambda_x * t)
    for t in range(len(p_rel)):
        expected = math.exp(cfg.lambda_x * t)
        assert math.isclose(p_rel[t], expected, rel_tol=1e-3)


def test_scenario_trajectories_sanity() -> None:
    """Verify Bull scenario yields higher terminal wealth than Bear scenario."""
    cfg = Config(n_V=30, n_CCR=5, n_actions=5, n_Z=10, alpha=0.2)
    grids = create_grids(cfg)
    policy = np.full(grids.n_S, 0.5)  # 50% in risky asset

    scenarios = simulate_market_scenarios(grids, policy, n_years=5)
    bull = scenarios["Bull (+1.0 sigma)"]
    bear = scenarios["Bear (-1.0 sigma)"]

    # Bull terminal wealth must exceed Bear terminal wealth
    assert bull["wealth"][-1] > bear["wealth"][-1]
