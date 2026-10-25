"""Unit test for brute_force_demo.py confirming policy iteration finds the exact global optimum."""

import pytest
from brute_force_demo import run_brute_force_demo


def test_brute_force_vs_policy_iteration_equivalence() -> None:
    """Verify that Howard's Policy Iteration finds the identical global optimum as exhaustive enumeration.

    Tests on a miniature grid (n_V=3, n_CCR=2 => n_S=6 states, 3 actions => 3^6 = 729 policies).
    """
    comp = run_brute_force_demo(
        n_V=3,
        n_CCR=2,
        n_actions=3,
        n_Z=8,
        alpha=0.0,
        verbose=False,
    )

    assert comp.policy_match, "Policy Iteration must match the exhaustive brute force optimal policy!"
    assert comp.value_close, "Value function from Policy Iteration must match the brute-force best value!"
    assert comp.policy_iter_iterations <= 4
