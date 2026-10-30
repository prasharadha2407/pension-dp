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


# --------------------------------------------------------------------------
# Backward induction and comparison tests (Task 2)
# --------------------------------------------------------------------------
from backward_induction import backward_induction, compare_policies
from pension_core import _q_values, utility


def test_backward_induction_shapes_and_dtypes():
    p = small()
    sol = backward_induction(p, T=5)
    assert sol.policy_by_year.shape == (5, sol.grids.n_S)
    assert sol.policy_by_year.dtype == np.int64
    assert sol.value_by_year.shape == (6, sol.grids.n_S)
    assert sol.value_by_year.dtype == np.float64
    assert np.array_equal(sol.value_by_year[5], np.zeros(sol.grids.n_S))
    assert sol.iterations == 5
    assert sol.policy_grid().min() >= 0.0 and sol.policy_grid().max() <= 1.0
    assert np.array_equal(sol.policy, sol.policy_by_year[0])
    assert np.array_equal(sol.value, sol.value_by_year[0])
    assert sol.method == "backward_induction"


def test_backward_induction_T1_equals_one_period_reward():
    p = small()
    sol = backward_induction(p, T=1)
    g = sol.grids
    expected_reward = p.reward_scale * utility(g.P_s, p.F)
    assert np.allclose(sol.value, expected_reward)
    assert np.array_equal(sol.policy, np.zeros(g.n_S, dtype=np.int64))


def test_backward_induction_T2_matches_manual_computation():
    p = small()
    g = Grids(p)
    nxt = build_next_state(g)
    sol = backward_induction(p, T=2, g=g, nxt=nxt)

    reward = p.reward_scale * utility(g.P_s, p.F)
    V2 = np.zeros(g.n_S)
    q1 = _q_values(g, nxt, V2, reward)
    pol1 = np.argmax(q1, axis=1)
    V1 = q1[np.arange(g.n_S), pol1]

    q0 = _q_values(g, nxt, V1, reward)
    pol0 = np.argmax(q0, axis=1)
    V0 = q0[np.arange(g.n_S), pol0]

    assert np.array_equal(sol.policy_by_year[1], pol1)
    assert np.allclose(sol.value_by_year[1], V1)
    assert np.array_equal(sol.policy_by_year[0], pol0)
    assert np.allclose(sol.value_by_year[0], V0)


def test_backward_induction_monotonicity_and_convergence():
    p = Params(n_V=30)
    g = Grids(p)
    nxt = build_next_state(g)
    vi = value_iteration(p)
    v_vi = vi.value

    v_prev = None
    for T in [1, 2, 5, 10, 20]:
        bi = backward_induction(p, T=T, g=g, nxt=nxt)
        v0 = bi.value
        if v_prev is not None:
            assert (v0 <= v_prev + 1e-12).all()
        v_prev = v0

        gap = float(np.max(np.abs(v_vi - v0)))
        bound = float((p.gamma ** T) * np.max(np.abs(v_vi)) + 1e-5)
        assert gap <= bound


def test_backward_induction_large_T_matches_policy_iteration():
    p = Params(n_V=30)
    pi = policy_iteration(p)
    bi = backward_induction(p, T=150, g=pi.grids)
    agreement = float((bi.policy == pi.policy).mean())
    assert agreement >= 0.98


def test_compare_policies_identical():
    p = small()
    pi = policy_iteration(p)
    cmp_res = compare_policies(pi, pi, ref=pi)
    assert cmp_res["exact_agreement_pct"] == 100.0
    assert cmp_res["equivalent_action_pct"] == 100.0
    assert cmp_res["mean_abs_action_diff"] == 0.0



def test_benchmark_algorithms_smoke(tmp_path):
    from benchmark_algorithms import run_benchmark
    res = run_benchmark(n_vs=(30,), T=2, repeats=1, out_dir=str(tmp_path))
    assert len(res) == 1
    assert (tmp_path / "benchmark_runtimes.csv").exists()
    assert (tmp_path / "benchmark_runtime.png").exists()
    assert (tmp_path / "benchmark_runtime.png").stat().st_size > 0


def test_complexity_table_smoke():
    from complexity_table import generate_complexity_data
    rows, trans_mb = generate_complexity_data()
    assert len(rows) == 4
    assert trans_mb > 0
    brute = next(r for r in rows if "Brute" in r["Algorithm"])
    val = [v for k, v in brute.items() if "Concrete" in k][0]
    assert "10^34,378" in val


def test_draw_benchmark_headless_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from backward_induction import backward_induction
    import gui_plots as gp

    p = Params(n_V=30)
    pi = policy_iteration(p)
    vi = value_iteration(p, max_iter=15)
    bi = backward_induction(p, T=3)

    fig = Figure(figsize=(9, 6))
    gp.draw_benchmark(fig, pi, vi, bi)
    out_file = tmp_path / "bench_headless.png"
    fig.savefig(out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
