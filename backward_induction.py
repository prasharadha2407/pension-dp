"""Finite-horizon Backward Induction solver for the pension fund MDP.

Subclasses pension_core.Solution to provide drop-in compatibility with GUI
and visualization tools, computing optimal stage-dependent investment policies
for a finite planning horizon T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np

from pension_core import (
    Grids,
    Params,
    Solution,
    _q_values,
    build_next_state,
    utility,
)


@dataclass
class BackwardSolution(Solution):
    """Stores the solution of the finite-horizon Backward Induction problem.

    Subclasses pension_core.Solution so it can be passed to all existing
    plotting, metrics, and GUI consumers.

    Inherited fields represent year-0 quantities:
        policy: Year-0 optimal action indices, shape (n_S,).
        value: Year-0 value function V_0(s), shape (n_S,).
        iterations: Number of stages T.
        history: List of (stage_index, max |V_k - V_{k+1}|, elapsed_seconds).
        seconds: Total execution time in seconds (including grid/transition setup).
        method: Solver identifier, set to "backward_induction".

    Additional fields:
        horizon: Planning horizon T in years.
        policy_by_year: Array of optimal action indices across all stages,
            shape (T, n_S) of dtype int64.
        value_by_year: Value function per stage from k=0 to terminal k=T (all 0),
            shape (T + 1, n_S) of dtype float64.
    """

    horizon: int = 10
    policy_by_year: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    value_by_year: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


def backward_induction(
    p: Params,
    T: int = 10,
    progress: Callable[[str], None] | None = None,
    g: Grids | None = None,
    nxt: np.ndarray | None = None,
) -> BackwardSolution:
    """Solve the finite-horizon pension MDP by backward induction.

    Parameters:
        p: Problem parameters.
        T: Finite decumulation horizon in years (default: 10).
        progress: Optional progress callback receiving informative strings.
        g: Optional precomputed Grids instance. If None, built from p.
        nxt: Optional precomputed next_state tensor. If None, built from g.

    Returns:
        BackwardSolution containing year-0 policy, value, stage-by-stage trajectories,
        and execution history.
    """
    p.validate()
    if T <= 0:
        raise ValueError(f"Horizon T must be positive, got {T}")

    t_all = time.perf_counter()
    if g is None:
        g = Grids(p)
    if nxt is None:
        nxt = build_next_state(g)

    reward = p.reward_scale * utility(g.P_s, p.F)
    n_S = g.n_S

    policy_by_year = np.empty((T, n_S), dtype=np.int64)
    value_by_year = np.zeros((T + 1, n_S), dtype=np.float64)

    history = []
    # Dynamic programming recursion backward from k = T - 1 down to 0
    for k in range(T - 1, -1, -1):
        t0 = time.perf_counter()
        q = _q_values(g, nxt, value_by_year[k + 1], reward)
        # Ties break to smallest action index (safest investment asset)
        policy_k = np.argmax(q, axis=1)
        V_k = q[np.arange(n_S), policy_k]

        policy_by_year[k] = policy_k
        value_by_year[k] = V_k

        diff = float(np.abs(V_k - value_by_year[k + 1]).max())
        elapsed = time.perf_counter() - t0
        history.append((k, diff, elapsed))
        if progress:
            progress(
                f"Backward induction stage {k} (T={T}): max |V_{k} - V_{k+1}| = {diff:.4e}"
            )

    # Order history chronologically by year k (0 to T-1)
    history.sort(key=lambda h: h[0])

    total_seconds = time.perf_counter() - t_all
    return BackwardSolution(
        params=p,
        grids=g,
        policy=policy_by_year[0],
        value=value_by_year[0],
        iterations=T,
        history=history,
        seconds=total_seconds,
        method="backward_induction",
        horizon=T,
        policy_by_year=policy_by_year,
        value_by_year=value_by_year,
    )
