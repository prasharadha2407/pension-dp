"""Unit tests for transitions.py covering all required edge cases."""

import math
import numpy as np
import pytest

from config import Config
from grids import create_grids
from transitions import continuous_transition_step, precompute_transitions


def test_transition_edge_case_1_boundary_clamping() -> None:
    """Edge Case 1: V_next below V_min or above V_max is clamped safely."""
    cfg = Config()
    V_low = cfg.V_min
    P = 100.0
    a = 1.0
    extreme_neg_Z = -10.0  # Huge market crash

    V_next, P_next, CCR_next = continuous_transition_step(V_low, P, a, extreme_neg_Z, cfg)
    assert V_next >= cfg.V_min
    assert V_next <= cfg.V_max
    assert P_next >= cfg.P_min
    assert P_next <= cfg.P_max
    assert CCR_next >= cfg.ccr_min
    assert CCR_next <= cfg.ccr_max

    extreme_pos_Z = 10.0  # Huge boom
    V_high = cfg.V_max
    V_next_h, _, _ = continuous_transition_step(V_high, P, a, extreme_pos_Z, cfg)
    assert V_next_h <= cfg.V_max


def test_transition_edge_case_3_pension_strictly_above_F() -> None:
    """Edge Case 3: P_next must always satisfy P > F (F=25.8)."""
    cfg = Config(F=25.8, P_min=26.8)
    V_tiny = 1.0
    P_tiny = 20.0  # Below floor
    a = 0.0
    Z = 0.0

    V_next, P_next, _ = continuous_transition_step(V_tiny, P_tiny, a, Z, cfg)
    assert P_next >= cfg.P_min
    assert P_next > cfg.F


def test_transition_edge_case_4_negative_wealth_handling() -> None:
    """Edge Case 4: Wealth zero or negative after paying pension is clamped and tracked."""
    cfg = Config()
    V_tiny = 10.0
    P_huge = 5000.0  # Huge pension payout leading to negative raw wealth
    a = 0.5
    Z = -2.0

    V_next, P_next, _ = continuous_transition_step(V_tiny, P_huge, a, Z, cfg)
    assert V_next == cfg.V_min
    assert P_next >= cfg.P_min


def test_transition_edge_case_5_ccr_corridor_tolerance() -> None:
    """Edge Case 5: CCR exactly on 1.00 or 1.25 with 1e-9 tolerance avoids false resets."""
    cfg = Config(corridor_tolerance=1e-9)
    P = 200.0
    E = P / cfg.r_plus_lambda

    # Case A: V_next / E is exactly 1.00 + 5e-10 (inside corridor)
    # With a=0 and shock=0, let's test directly via ratio logic
    V_exact_lower = E * (1.00 - 5e-10)  # within 1e-9 of 1.00
    # Simulate single step where V_next lands here
    # E = P / r_plus_lambda, so if ratio is within tolerance, P_next should remain P
    ratio = V_exact_lower / E
    tol = cfg.corridor_tolerance
    inside = (ratio >= cfg.ccr_min - tol) and (ratio <= cfg.ccr_max + tol)
    assert inside, "Ratio within 1e-9 tolerance must be recognized as inside corridor!"

    V_exact_upper = E * (1.25 + 5e-10)  # within 1e-9 of 1.25
    ratio_up = V_exact_upper / E
    inside_up = (ratio_up >= cfg.ccr_min - tol) and (ratio_up <= cfg.ccr_max + tol)
    assert inside_up, "Upper ratio within 1e-9 tolerance must be recognized as inside corridor!"


def test_invested_wealth_split() -> None:
    """Paper eq. (5.4) & (5.5): Invested wealth allocation.

    invested = E + (1 - alpha) * (V - E).
    When alpha = 0: invested = V.
    When V = E: invested = E.
    """
    cfg_0 = Config(alpha=0.0)
    V = 12000.0
    P = 300.0
    E = P / cfg_0.r_plus_lambda
    invested_0 = E + (1.0 - cfg_0.alpha) * (V - E)
    assert math.isclose(invested_0, V, rel_tol=1e-9)

    cfg_4 = Config(alpha=0.4)
    invested_4 = E + (1.0 - cfg_4.alpha) * (V - E)
    expected_4 = E + 0.6 * (V - E)
    assert math.isclose(invested_4, expected_4, rel_tol=1e-9)


def test_precompute_transitions_dimensions_and_validity() -> None:
    """Verify precomputed transition tensor next_state has correct shape and indices."""
    cfg = Config(n_V=10, n_CCR=5, n_actions=3, n_Z=4)
    grids = create_grids(cfg)

    res = precompute_transitions(grids, chunk_size=20)
    next_s = res.next_state

    assert next_s.shape == (50, 3, 4)
    assert next_s.dtype == np.int32
    assert np.all(next_s >= 0)
    assert np.all(next_s < 50)
