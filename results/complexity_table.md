# Algorithmic Complexity Comparison (Lichtenstern & Zagst, 2022)

| Algorithm | Time Complexity | Space Complexity | Concrete Operation Count ($n_S=26,000, n_a=21, n_z=40, T=10$) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Brute-Force Policy Enumeration** | `O(n_a^n_S * cost_eval)` | `O(n_S)` | 21^26,000 approx 10^34,378 | Completely infeasible (exceeds atoms in universe ~10^80). Guarantees global optimum by exhaustive evaluation. |
| **Backward Induction (Finite Horizon T=10)** | `O(T * n_S * n_a * n_z)` | `O(T * n_S) [or O(n_S) for year-0 only]` | 10 x 26,000 x 21 x 40 approx 2.18 x 10^8 ops | Feasible for finite planning horizon T. Exact dynamic programming recursion. Completes in ~3.2s on 26,000 states. |
| **Value Iteration (Bellman Operator)** | `O(M * n_S * n_a * n_z)` | `O(n_S)` | 365 iters (theo M approx 418) x 21,840,000 approx 7.97 x 10^9 ops | Geometric contraction (gamma=0.9501 < 1). Banach fixed point. Converges in 365 iterations (~68s on 26,000 states). |
| **Howard's Policy Iteration (Infinite Horizon)** | `O(K * (solve + n_S * n_a * n_z))` | `O(n_S * n_z + n_S)` | 7 iters x (2.18 x 10^7 improv + BiCGSTAB solve) approx 1.60 x 10^8 ops | Exact infinite-horizon optimum (Theorem 5). Converges in K=7 iterations (~4.3s on 26,000 states). BiCGSTAB solve is <0.2s per iteration. 16x faster than VI. |

> **Shared Preprocessing Memory**: The discrete transition tensor `next_state[s, a, z]` is of shape `(26000, 21, 40)` with dtype `int32`, requiring **83.31 MB** of memory. This represents the dominant working memory allocation shared across all three dynamic programming algorithms (Backward Induction, Value Iteration, and Policy Iteration).
