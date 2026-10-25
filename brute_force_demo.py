"""Brute-force policy enumeration on a tiny MDP problem.

Demonstrates exact equivalence between exhaustive brute-force search
and Howard's Policy Iteration on a small state space, and mathematically
illustrates the combinatorial explosion that renders brute force impossible
for the full pension MDP (21^26000 policies).
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from typing import NamedTuple

import numpy as np

from config import Config
from grids import Grids, create_grids
from policy_iteration import evaluate_policy, run_policy_iteration
from transitions import precompute_transitions

logger = logging.getLogger(__name__)


class BruteForceComparison(NamedTuple):
    """Holds comparison results between brute-force search and policy iteration."""

    n_S: int
    n_actions: int
    total_policies: int
    best_policy_brute: np.ndarray
    best_value_brute: np.ndarray
    policy_iter_policy: np.ndarray
    policy_iter_value: np.ndarray
    policy_match: bool
    value_close: bool
    brute_force_seconds: float
    policy_iter_seconds: float
    policy_iter_iterations: int


def run_brute_force_demo(
    n_V: int = 4,
    n_CCR: int = 2,
    n_actions: int = 3,
    n_Z: int = 10,
    alpha: float = 0.0,
    verbose: bool = True,
) -> BruteForceComparison:
    """Enumerate all n_a^n_S policies on a miniature grid and compare with policy iteration.

    Default: n_V=4, n_CCR=2 -> n_S = 8 states, 3 actions -> 3^8 = 6561 policies.
    """
    tiny_config = Config(
        mode="paper_calibrated",
        alpha=alpha,
        n_V=n_V,
        n_CCR=n_CCR,
        n_actions=n_actions,
        n_Z=n_Z,
    )
    grids = create_grids(tiny_config)
    transitions = precompute_transitions(grids, chunk_size=grids.n_S)
    next_state = transitions.next_state

    n_S = grids.n_S
    n_a = grids.n_actions
    gamma = tiny_config.gamma
    reward_vec = grids.reward_flat
    total_policies = n_a**n_S

    if verbose:
        print("\n" + "=" * 70)
        print("BRUTE FORCE vs POLICY ITERATION CORRECTNESS DEMO")
        print("=" * 70)
        print(f"Tiny grid parameters: n_V={n_V}, n_CCR={n_CCR} => n_S={n_S} states")
        print(f"Action choices: {grids.actions.tolist()} (n_a={n_a})")
        print(f"Shock outcomes: n_Z={n_Z}")
        print(f"Total possible policies to enumerate: {n_a}^{n_S} = {total_policies:,}")
        print("-" * 70)

    # 1. Brute-force exhaustive enumeration
    t0_bf = time.perf_counter()
    best_policy_bf: np.ndarray | None = None
    best_V_bf: np.ndarray | None = None
    best_sum_V = -float("inf")

    # Enumerate Cartesian product {0, ..., n_a-1}^n_S
    for p_tuple in itertools.product(range(n_a), repeat=n_S):
        pi = np.array(p_tuple, dtype=np.int32)
        V_pi, _ = evaluate_policy(pi, next_state, reward_vec, gamma, n_S, n_Z)
        sum_V = float(np.sum(V_pi))
        if sum_V > best_sum_V:
            best_sum_V = sum_V
            best_policy_bf = pi
            best_V_bf = V_pi

    elapsed_bf = time.perf_counter() - t0_bf

    # 2. Howard's Policy Iteration on the exact same problem
    t0_pi = time.perf_counter()
    pi_res = run_policy_iteration(grids, next_state, epsilon=1e-5, verbose=False)
    elapsed_pi = time.perf_counter() - t0_pi

    assert best_policy_bf is not None and best_V_bf is not None

    policy_match = bool(np.array_equal(best_policy_bf, pi_res.policy))
    value_close = bool(np.allclose(best_V_bf, pi_res.value_function, atol=1e-7))

    if verbose:
        print(f"1. Exhaustive Brute-Force Search:")
        print(f"   Evaluated: {total_policies:,} policies in {elapsed_bf:.3f} seconds")
        print(f"   Optimal policy: {best_policy_bf.tolist()}")
        print(f"   Optimal value sum: {best_sum_V:.4f}")
        print(f"\n2. Howard's Policy Iteration:")
        print(f"   Converged in: {pi_res.iterations} iterations in {elapsed_pi:.4f} seconds")
        print(f"   Optimal policy: {pi_res.policy.tolist()}")
        print(f"   Optimal value sum: {np.sum(pi_res.value_function):.4f}")
        print(f"\n3. Verification:")
        print(f"   Policies match exactly: {policy_match}")
        print(f"   Value functions match:  {value_close}")
        print("-" * 70)

        # Infeasibility of full MDP calculation
        full_n_S = 26000
        full_n_a = 21
        log10_full = full_n_S * math.log10(full_n_a)
        print("THEORETICAL INFEASIBILITY FOR FULL PROBLEM:")
        print(f"   Full MDP state space: n_S = 26,000 | actions: n_a = 21")
        print(f"   Number of deterministic policies: 21^26,000 = 10^{log10_full:.1f}")
        print(f"   For perspective, atoms in observable universe ~ 10^80.")
        print(f"   At 10^9 evaluations/second, brute force takes ~ 10^{log10_full - 9 - 7.5:.1f} years.")
        print(f"   Howard's Policy Iteration solves this in <= 7 iterations (~2 seconds)!")
        print("=" * 70 + "\n")

    return BruteForceComparison(
        n_S=n_S,
        n_actions=n_a,
        total_policies=total_policies,
        best_policy_brute=best_policy_bf,
        best_value_brute=best_V_bf,
        policy_iter_policy=pi_res.policy,
        policy_iter_value=pi_res.value_function,
        policy_match=policy_match,
        value_close=value_close,
        brute_force_seconds=elapsed_bf,
        policy_iter_seconds=elapsed_pi,
        policy_iter_iterations=pi_res.iterations,
    )


if __name__ == "__main__":
    run_brute_force_demo()
