import numpy as np
from pension_core import Params, Grids, build_next_state, policy_iteration, value_iteration, simulate, metrics, step


def small(**kw):
    return Params(n_V=60, **kw)


def test_initial_pension_matches_paper():
    for a, expected in [(0.0, 277.33), (0.2, 269.84), (0.4, 258.21)]:
        assert abs(Params(alpha=a).P0() - expected) < 0.01


def test_next_states_valid_and_rows_stochastic():
    p = small(); g = Grids(p); nxt = build_next_state(g)
    assert nxt.min() >= 0 and nxt.max() < g.n_S
    assert nxt.shape == (g.n_S, p.n_a, p.n_z)
    assert abs(g.w.sum() - 1.0) < 1e-12


def test_policy_iteration_matches_value_iteration():
    p = small(alpha=0.2)
    pi, vi = policy_iteration(p), value_iteration(p)
    assert (pi.policy == vi.policy).mean() > 0.999
    assert pi.iterations <= 10 and pi.iterations < vi.iterations
    assert np.isfinite(pi.value).all()


def test_actions_within_bounds():
    p = small(); s = policy_iteration(p)
    assert s.policy_grid().min() >= 0.0 and s.policy_grid().max() <= 1.0


def test_pension_floor_and_reproducibility():
    p = small(); s = policy_iteration(p)
    r1 = simulate(p, s, n_paths=500, T=5, seed=1); r2 = simulate(p, s, n_paths=500, T=5, seed=1)
    assert np.array_equal(r1.P, r2.P)
    assert r1.P.min() >= p.P_min - 1e-9


def test_higher_buffer_reduces_pension_cuts():
    cuts = []
    for a in (0.0, 0.4):
        p = small(alpha=a); s = policy_iteration(p)
        cuts.append(metrics(simulate(p, s, n_paths=4000, T=10, seed=7), p)["P(>=1 pension cut)"])
    assert cuts[1] < cuts[0]


def test_alpha_validation():
    import pytest
    with pytest.raises(ValueError):
        Params(alpha=0.9).validate()
