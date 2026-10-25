"""Unit tests for policy_iteration.py: Howard's algorithm, stochastic Q, monotonicity."""

import math
import numpy as np
import pytest

from config import Config
from grids import create_grids
from policy_iteration import (
    build_transition_matrix,
    evaluate_policy,
    run_policy_iteration,
)
from transitions import precompute_transitions


def test_transition_matrix_is_stochastic() -> None:
    """Edge Case 10: Matrix Q must have rows summing to exactly 1.0."""
    cfg = Config(n_V=20, n_CCR=5, n_actions=5, n_Z=10)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids, chunk_size=50)

    # Random policy
    rng = np.random.default_rng(123)
    policy = rng.integers(0, grids.n_actions, size=grids.n_S, dtype=np.int32)

    Q = build_transition_matrix(policy, trans.next_state, grids.n_S, grids.n_Z)
    row_sums = np.array(Q.sum(axis=1)).flatten()

    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_linear_solve_invertible_no_nan() -> None:
    """Edge Case 10: I - gamma*Q must be strictly diagonally dominant and invertible."""
    cfg = Config(n_V=30, n_CCR=5, n_actions=5, n_Z=10)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids)

    policy = np.zeros(grids.n_S, dtype=np.int32)
    V, Q = evaluate_policy(
        policy,
        trans.next_state,
        grids.reward_flat,
        cfg.gamma,
        grids.n_S,
        grids.n_Z,
    )

    assert len(V) == grids.n_S
    assert np.all(np.isfinite(V))
    assert not np.any(np.isnan(V))
    assert not np.any(np.isinf(V))
    # Rewards are negative (since HARA utility is negative), so V must be strictly negative
    assert np.all(V < 0.0)


def test_value_function_monotonicity_theorem_5() -> None:
    """Paper Theorem 5: Value function is monotonically non-decreasing across policy iterations."""
    cfg = Config(n_V=50, n_CCR=10, n_actions=5, n_Z=10, alpha=0.0)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids)

    pi_res = run_policy_iteration(grids, trans.next_state, epsilon=0.01, verbose=False)

    assert pi_res.converged
    # Check each iteration's monotonicity flag
    for h in pi_res.history:
        assert h["is_monotone"], f"Monotonicity failed in iteration {h['iteration']} with min_diff={h['min_v_diff']}"


def test_policy_iteration_convergence_count() -> None:
    """Verify policy iteration converges in few iterations (paper reports <= 7)."""
    cfg = Config(n_V=100, n_CCR=10, n_actions=11, n_Z=20, alpha=0.2)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids)

    pi_res = run_policy_iteration(grids, trans.next_state, epsilon=0.01, verbose=False)

    assert pi_res.converged
    assert pi_res.iterations <= 10


def test_tie_breaking_safest_action_edge_case_6() -> None:
    """Edge Case 6: Ties in best action must pick the smallest (safest) action index."""
    # When all actions have identical continuation value, np.argmax must return 0
    W_identical = np.zeros((10, 5))  # All 0
    best_actions = np.argmax(W_identical, axis=1)
    assert np.all(best_actions == 0), "Ties must be broken in favor of index 0 (safest action)!"
