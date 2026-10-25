"""Unit tests for grid construction, shock discretization, and coordinate mapping."""

import math
import numpy as np
import pytest

from config import Config
from grids import create_grids


def test_grid_dimensions() -> None:
    """Verify state, action, and shock grid dimensions."""
    cfg = Config()
    grids = create_grids(cfg)

    assert grids.n_V == 1000
    assert grids.n_CCR == 26
    assert grids.n_S == 26000
    assert grids.n_actions == 21
    assert grids.n_Z == 40

    assert len(grids.V_flat) == 26000
    assert len(grids.CCR_flat) == 26000
    assert len(grids.P_flat) == 26000
    assert len(grids.reward_flat) == 26000


def test_shock_grid_endpoints_and_symmetry() -> None:
    """Verify shock grid Z construction and endpoints around +/- 2.2414.

    Paper Section 5.3:
    Quantile midpoints (k + 0.5) / 40 for k in 0..39.
    z_0 = norm.ppf(0.5 / 40) = norm.ppf(0.0125) approx -2.241403
    z_39 = norm.ppf(39.5 / 40) = norm.ppf(0.9875) approx +2.241403
    """
    cfg = Config()
    grids = create_grids(cfg)
    Z = grids.Z

    assert len(Z) == 40
    # Endpoints unit test requested in Section 3
    assert math.isclose(Z[0], -2.241403, abs_tol=1e-4)
    assert math.isclose(Z[-1], 2.241403, abs_tol=1e-4)

    # Symmetry around 0
    assert np.allclose(Z, -Z[::-1], atol=1e-12)

    # Weights
    assert np.allclose(grids.Z_weights, 1.0 / 40.0)
    assert math.isclose(float(np.sum(grids.Z_weights)), 1.0, rel_tol=1e-9)


def test_action_grid() -> None:
    """Verify action choices a in [0.0, 1.0] with step 0.05."""
    cfg = Config()
    grids = create_grids(cfg)
    actions = grids.actions

    assert len(actions) == 21
    assert math.isclose(actions[0], 0.0)
    assert math.isclose(actions[-1], 1.0)
    assert np.all(np.diff(actions) > 0)
    assert math.isclose(actions[1] - actions[0], 0.05, abs_tol=1e-6)


def test_state_indexing_bijection() -> None:
    """Verify mapping between (i, j) coordinates and flat index s."""
    cfg = Config(n_V=10, n_CCR=5)
    grids = create_grids(cfg)

    # Test round trip for all states
    for i in range(10):
        for j in range(5):
            s = grids.state_to_index(i, j)
            i_rec, j_rec = grids.index_to_state(s)
            assert i == i_rec
            assert j == j_rec


def test_snap_to_nearest_node() -> None:
    """Edge Case 2: Snapping continuous (V, CCR) to the nearest grid node."""
    cfg = Config()
    grids = create_grids(cfg)

    # Exact node
    v_exact = grids.V_grid[50]
    ccr_exact = grids.CCR_grid[10]
    i_idx, j_idx, s_idx = grids.snap_to_nearest_node(v_exact, ccr_exact)
    assert i_idx == 50
    assert j_idx == 10
    assert s_idx == 50 * grids.n_CCR + 10

    # Slight perturbation within tolerance
    i_pert, j_pert, s_pert = grids.snap_to_nearest_node(v_exact + 0.1, ccr_exact + 0.001)
    assert i_pert == 50
    assert j_pert == 10

    # Outside boundaries: clamps properly
    i_clamp_low, j_clamp_low, _ = grids.snap_to_nearest_node(cfg.V_min - 1000, 0.5)
    assert i_clamp_low == 0
    assert j_clamp_low == 0

    i_clamp_high, j_clamp_high, _ = grids.snap_to_nearest_node(cfg.V_max + 1000, 1.5)
    assert i_clamp_high == grids.n_V - 1
    assert j_clamp_high == grids.n_CCR - 1
