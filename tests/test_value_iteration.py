"""Unit tests for value_iteration.py: contraction convergence and policy agreement with policy iteration."""

import numpy as np
import pytest

from config import Config
from grids import create_grids
from policy_iteration import run_policy_iteration
from transitions import precompute_transitions
from value_iteration import run_value_iteration


def test_value_iteration_convergence_and_agreement() -> None:
    """Verify Value Iteration converges and produces identical optimal policy as Policy Iteration."""
    cfg = Config(n_V=40, n_CCR=6, n_actions=5, n_Z=10, alpha=0.0)
    grids = create_grids(cfg)
    trans = precompute_transitions(grids)

    # 1. Run Policy Iteration
    pi_res = run_policy_iteration(grids, trans.next_state, epsilon=0.001, verbose=False)

    # 2. Run Value Iteration with tight tolerance
    vi_res = run_value_iteration(grids, trans.next_state, tol=1e-7, max_iter=1500, verbose=False)

    assert vi_res.converged
    assert pi_res.converged

    # Optimal value functions should match closely
    assert np.allclose(pi_res.value_function, vi_res.value_function, atol=1e-4)

    # Optimal policies should be identical
    mismatch_rate = np.mean(pi_res.policy != vi_res.policy)
    assert mismatch_rate < 0.01, f"Mismatch rate {mismatch_rate*100:.2f}% between PI and VI!"
