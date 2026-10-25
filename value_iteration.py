"""Value iteration baseline algorithm for the pension MDP.

Implements the standard Bellman optimality operator:
V <- r + gamma * max_a E_Z [ V(next_state[s, a, Z]) ]
Used as a benchmark comparison against Howard's Policy Iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any

import numpy as np

from grids import Grids

logger = logging.getLogger(__name__)


@dataclass
class ValueIterationResult:
    """Stores the output and diagnostic history of Value Iteration."""

    policy: np.ndarray  # Best action indices, shape (n_S,)
    policy_actions: np.ndarray  # Corresponding float action values a in [0, 1]
    value_function: np.ndarray  # Converged value function V(s), shape (n_S,)
    iterations: int  # Total iterations to reach tolerance
    converged: bool  # Whether convergence was achieved within max_iter
    history: list[dict[str, Any]] = field(default_factory=list)


def run_value_iteration(
    grids: Grids,
    next_state: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 2000,
    verbose: bool = False,
) -> ValueIterationResult:
    """Execute Value Iteration until sup-norm change is strictly below tol.

    Bellman Operator:
    T(V)(s) = r(s) + gamma * max_a (1 / n_Z) sum_z V(next_state[s, a, z])
    """
    n_S = grids.n_S
    gamma = grids.config.gamma
    reward_vec = grids.reward_flat
    actions = grids.actions

    # Initialize value function to zero (or reward)
    V = np.zeros(n_S, dtype=np.float64)
    history: list[dict[str, Any]] = []
    converged = False

    t_start = time.perf_counter()

    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()

        # Vectorized expected continuation value W(s, a): shape (n_S, n_actions)
        W = np.mean(V[next_state], axis=2)

        # Maximize over candidate actions: shape (n_S,)
        max_continuation = np.max(W, axis=1)

        # Bellman update
        V_new = reward_vec + gamma * max_continuation

        # Sup-norm difference
        delta_V = float(np.max(np.abs(V_new - V)))
        elapsed = time.perf_counter() - t0

        if it % 50 == 0 or it == 1 or delta_V < tol:
            history.append({
                "iteration": it,
                "delta_V": delta_V,
                "elapsed_cumulative": time.perf_counter() - t_start,
            })
            if verbose:
                logger.info("VI Iter %4d: sup-norm delta = %.3e | step time = %.4fs", it, delta_V, elapsed)

        V = V_new

        if delta_V < tol:
            converged = True
            if verbose:
                logger.info("Value Iteration converged in %d iterations (delta = %.3e)", it, delta_V)
            break
    else:
        logger.warning("Value Iteration reached max iterations (%d) without reaching tol=%.2e", max_iter, tol)

    # Extract greedy policy from the converged value function
    W_final = np.mean(V[next_state], axis=2)
    # np.argmax ensures ties pick smallest action (Edge case 6)
    policy = np.argmax(W_final, axis=1).astype(np.int32)

    return ValueIterationResult(
        policy=policy,
        policy_actions=actions[policy],
        value_function=V,
        iterations=it,
        converged=converged,
        history=history,
    )
