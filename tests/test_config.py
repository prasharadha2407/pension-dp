"""Unit tests for configuration, parameter validation, and utility functions."""

import math
import pytest
import numpy as np

from config import Config, get_default_config


def test_default_config_paper_calibrated() -> None:
    """Verify default configuration in paper_calibrated mode."""
    cfg = get_default_config(mode="paper_calibrated", alpha=0.0)
    assert cfg.r == 0.01
    assert math.isclose(cfg.lambda_x, 0.0212, rel_tol=1e-5)
    assert math.isclose(cfg.r_plus_lambda, 0.0312, rel_tol=1e-5)
    assert cfg.gamma < 1.0
    assert cfg.gamma > 0.0

    # Initial pensions matching Lichtenstern & Zagst (2022) Table 1
    p0_00 = cfg.initial_pension(alpha=0.0)
    p0_02 = cfg.initial_pension(alpha=0.2)
    p0_04 = cfg.initial_pension(alpha=0.4)

    assert math.isclose(p0_00, 277.33, abs_tol=0.1)
    assert math.isclose(p0_02, 269.84, abs_tol=0.1)
    assert math.isclose(p0_04, 258.21, abs_tol=0.1)


def test_as_printed_mode_discrepancy() -> None:
    """Verify as_printed mode reproduces the mathematical discrepancy documented in Section 8."""
    cfg = Config(mode="as_printed", alpha=0.0)
    assert math.isclose(cfg.lambda_x, 0.0118, rel_tol=1e-5)
    assert math.isclose(cfg.r_plus_lambda, 0.0218, rel_tol=1e-5)

    p0 = cfg.initial_pension(alpha=0.0)
    # 0.0218 / 1.125 * 10000 approx 193.78
    assert math.isclose(p0, 193.78, abs_tol=0.1)


def test_alpha_validation_edge_case_8() -> None:
    """Edge Case 8: alpha must satisfy 0 <= alpha <= (1.25 - p_bar) / (1.25 - 1.00) = 0.5."""
    # Valid alphas
    Config(alpha=0.0)
    Config(alpha=0.2)
    Config(alpha=0.4)
    Config(alpha=0.5)

    # Invalid negative alpha
    with pytest.raises(ValueError, match="Invalid alpha"):
        Config(alpha=-0.01)

    # Invalid alpha exceeding upper bound 0.5
    with pytest.raises(ValueError, match="Invalid alpha"):
        Config(alpha=0.55)


def test_p_bar_validation() -> None:
    """Verify p_bar must be within [ccr_min, ccr_max]."""
    with pytest.raises(ValueError, match="p_bar"):
        Config(p_bar=0.95)

    with pytest.raises(ValueError, match="p_bar"):
        Config(p_bar=1.30)


def test_hara_utility_and_reward() -> None:
    """Verify HARA utility function U(P) = -4 / (P - F) and reward formula."""
    cfg = Config()
    F = cfg.F  # 25.8

    # Scalar test
    P_val = 125.8
    expected_u = -4.0 / (125.8 - 25.8)  # -4 / 100 = -0.04
    assert math.isclose(cfg.utility(P_val), expected_u, rel_tol=1e-7)

    # Array test
    P_arr = np.array([50.8, 125.8, 225.8])
    u_arr = cfg.utility(P_arr)
    assert len(u_arr) == 3
    assert np.all(u_arr < 0.0)  # Utility is negative for all valid P > F

    # Clamping test: P <= F must not cause zero division or NaN
    u_below = cfg.utility(F - 5.0)
    assert np.isfinite(u_below)
    assert u_below < 0.0


def test_pension_bounds_validation() -> None:
    """Edge Case 3: P_min must be strictly greater than F."""
    with pytest.raises(ValueError, match="P_min.*must be strictly greater"):
        Config(P_min=25.0, F=25.8)

    with pytest.raises(ValueError, match="P_max.*must be strictly greater"):
        Config(P_min=100.0, P_max=50.0)
