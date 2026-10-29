"""
pension_core.py - self-contained working model for
Lichtenstern & Zagst, "Optimal investment strategies for pension funds with
regulation-conform dynamic pension payment management in the absence of guarantees".

State  : (V, CCR) on a tensor grid, flattened to s = i * n_ccr + j
Action : a in [0, 1], share of the investment portfolio put in the risky asset
Reward : HARA utility of the current pension  U(P) = -4 / (P - F)   (b=-1, a_hat=1)
Solver : Howard's policy iteration (sparse linear solve + greedy improvement)
"""
from dataclasses import dataclass, field, replace
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.stats import norm


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    # market (paper Sect. 6)
    r: float = 0.01          # risk-free rate
    mu: float = 0.0297       # drift of the risky asset
    sigma: float = 0.1175    # volatility of the risky asset
    # mortality / regulation
    lam: float = 0.0212      # cohort mortality force (0.0212 = paper-calibrated, 0.0118 = as printed)
    p_bar: float = 1.125     # target capital coverage ratio
    ccr_min: float = 1.00
    ccr_max: float = 1.25
    alpha: float = 0.2       # buffer intensity in [0, 0.5]
    # utility
    F: float = 25.8          # subsistence level
    beta: float = 0.03       # subjective utility discount rate
    # MDP
    V0: float = 10_000.0
    V_min: float = 2_000.0
    V_max: float = 50_000.0
    P_min: float = 26.8      # F + 1
    P_max: float = 2_500.0
    # discretisation
    n_V: int = 200
    n_ccr: int = 26
    n_a: int = 21
    n_z: int = 40

    def validate(self):
        if not (0.0 <= self.alpha <= (self.ccr_max - self.p_bar) / (self.ccr_max - self.ccr_min) + 1e-12):
            raise ValueError("alpha must be in [0, (ccr_max - p_bar)/(ccr_max - ccr_min)] (= [0, 0.5])")
        if not (self.ccr_min <= self.p_bar <= self.ccr_max):
            raise ValueError("p_bar must lie inside the CCR corridor")
        if self.lam <= 0 or self.beta < 0:
            raise ValueError("lambda must be > 0 and beta >= 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.n_V < 10 or self.n_ccr < 3 or self.n_a < 2 or self.n_z < 3:
            raise ValueError("grid sizes too small")

    @property
    def rl(self):
        return self.r + self.lam

    @property
    def gamma(self):
        """One-period total discount factor e^{-(lambda+beta)}."""
        return float(np.exp(-(self.lam + self.beta)))

    @property
    def reward_scale(self):
        """(1 - e^{-(lam+beta)}) / (lam+beta), paper eq. (5.12)."""
        return (1.0 - self.gamma) / (self.lam + self.beta)

    def P0(self):
        """Initial equilibrium pension, paper eq. (5.12)."""
        a = self.alpha
        return (1 - a) / (self.p_bar - a) * self.rl * self.V0


def utility(P, F):
    """HARA utility with b = -1, a_hat = 1  ->  -4 / (P - F)."""
    return -4.0 / (np.asarray(P, dtype=float) - F)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------
class Grids:
    def __init__(self, p: Params):
        self.p = p
        self.V = np.linspace(p.V_min, p.V_max, p.n_V)
        self.ccr = np.linspace(p.ccr_min, p.ccr_max, p.n_ccr)
        self.a = np.linspace(0.0, 1.0, p.n_a)
        # quantile-midpoint discretisation of N(0,1)
        q = (np.arange(p.n_z) + 0.5) / p.n_z
        self.z = norm.ppf(q)
        self.w = np.full(p.n_z, 1.0 / p.n_z)
        self.n_S = p.n_V * p.n_ccr
        Vg, Cg = np.meshgrid(self.V, self.ccr, indexing="ij")
        self.V_s = Vg.ravel()
        self.ccr_s = Cg.ravel()
        self.P_s = np.clip(self.V_s * p.rl / self.ccr_s, p.P_min, p.P_max)

    def snap(self, V, ccr):
        """Nearest-node snapping (decoupled in V and CCR) -> flat state index."""
        p = self.p
        dV = (p.V_max - p.V_min) / (p.n_V - 1)
        dC = (p.ccr_max - p.ccr_min) / (p.n_ccr - 1)
        i = np.clip(np.rint((V - p.V_min) / dV), 0, p.n_V - 1).astype(np.int64)
        j = np.clip(np.rint((ccr - p.ccr_min) / dC), 0, p.n_ccr - 1).astype(np.int64)
        return i * p.n_ccr + j


# --------------------------------------------------------------------------
# One-step dynamics (shared by DP and simulation)
# --------------------------------------------------------------------------
TOL = 1e-9


def step(p: Params, V, P, a, z):
    """
    One year (Delta = 1) of the cohort fund, paper eqs. (3.27), (4.24)-(4.25), (5.13).
    P is the *cohort* pension; without a reset it decays with the survivors, e^{-lam} P.
    Returns V_next, P_next, ratio  where ratio = P_next / (e^{-lam} P):
        ratio < 1 individual pension cut, ratio = 1 stable, ratio > 1 enhancement.
    """
    E = P / p.rl                                   # reserve at t
    surplus = np.maximum(V - E, 0.0)
    I = E + (1 - p.alpha) * surplus                # investment portfolio (3.10)
    ret = p.r + a * (p.mu - p.r) + a * p.sigma * z
    V_next = np.clip(V + I * ret - P, p.V_min, p.V_max)

    decay = np.exp(-p.lam)
    E_next = decay * P / p.rl                      # reserve of the surviving cohort at t+1
    ccr = V_next / E_next
    inside = (ccr >= p.ccr_min - TOL) & (ccr <= p.ccr_max + TOL)
    P_reset = (1 - p.alpha) / (p.p_bar - p.alpha) * p.rl * V_next
    P_next = np.where(inside, decay * P, np.clip(P_reset, p.P_min, p.P_max))
    ratio = P_next / (decay * P)
    return V_next, P_next, ratio


# --------------------------------------------------------------------------
# Transition tensor
# --------------------------------------------------------------------------
def build_next_state(g: Grids):
    """next_state[s, a, z] (int32)."""
    p = g.p
    nxt = np.empty((g.n_S, p.n_a, p.n_z), dtype=np.int32)
    chunk = 2000
    for lo in range(0, g.n_S, chunk):
        hi = min(lo + chunk, g.n_S)
        V = g.V_s[lo:hi, None, None]
        P = g.P_s[lo:hi, None, None]
        a = g.a[None, :, None]
        z = g.z[None, None, :]
        Vn, Pn, _ = step(p, V, P, a, z)
        ccr_n = Vn * p.rl / Pn
        nxt[lo:hi] = g.snap(Vn, ccr_n)
    return nxt


# --------------------------------------------------------------------------
# Policy iteration
# --------------------------------------------------------------------------
@dataclass
class Solution:
    params: Params
    grids: Grids
    policy: np.ndarray            # action index per state
    value: np.ndarray
    iterations: int
    history: list = field(default_factory=list)   # (iter, n_changed, seconds)
    seconds: float = 0.0
    method: str = "policy_iteration"

    def action_at(self, V, ccr):
        return self.grids.a[self.policy[self.grids.snap(V, ccr)]]

    def policy_grid(self):
        p = self.params
        return self.grids.a[self.policy].reshape(p.n_V, p.n_ccr)


def _policy_matrix(nxt, policy, w, n_S):
    rows = np.repeat(np.arange(n_S), nxt.shape[2])
    cols = nxt[np.arange(n_S), policy, :].ravel()
    vals = np.tile(w, n_S)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_S, n_S))


def _q_values(g, nxt, v, reward):
    # q[s,a] = r(s) + gamma * sum_z w_z v[next[s,a,z]]   (chunked to keep memory low)
    ev = np.empty(nxt.shape[:2])
    for lo in range(0, nxt.shape[0], 4000):
        hi = lo + 4000
        ev[lo:hi] = (v[nxt[lo:hi]] * g.w[None, None, :]).sum(axis=2)
    return reward[:, None] + g.p.gamma * ev


def policy_iteration(p: Params, progress=None, max_iter=100):
    p.validate()
    t_all = time.perf_counter()
    g = Grids(p)
    nxt = build_next_state(g)
    reward = p.reward_scale * utility(g.P_s, p.F)
    n_S = g.n_S
    policy = np.zeros(n_S, dtype=np.int64)       # start: all in safe asset
    history = []
    v = np.zeros(n_S)
    for k in range(1, max_iter + 1):
        t0 = time.perf_counter()
        Q = _policy_matrix(nxt, policy, g.w, n_S)
        A = (sparse.identity(n_S, format="csr") - p.gamma * Q).tocsc()
        r_pi = reward
        if n_S <= 6000:
            v = spsolve(A, r_pi)
        else:
            v, info = bicgstab(A, r_pi, x0=v, rtol=1e-10, atol=0.0, maxiter=500)
            if info != 0:
                v = spsolve(A, r_pi)
        q = _q_values(g, nxt, v, reward)
        new_policy = np.argmax(q, axis=1)        # ties -> first (safest) action
        # keep current action if it is within tolerance of the best (avoids cycling)
        cur = q[np.arange(n_S), policy]
        keep = cur >= q[np.arange(n_S), new_policy] - 1e-12
        new_policy = np.where(keep, policy, new_policy)
        changed = int((new_policy != policy).sum())
        history.append((k, changed, time.perf_counter() - t0))
        if progress:
            progress(f"Policy iteration {k}: {changed} states changed")
        policy = new_policy
        if changed == 0:
            break
    return Solution(p, g, policy, v, k, history, time.perf_counter() - t_all)


def value_iteration(p: Params, progress=None, tol=1e-8, max_iter=2000):
    """Baseline for the runtime comparison."""
    p.validate()
    t_all = time.perf_counter()
    g = Grids(p)
    nxt = build_next_state(g)
    reward = p.reward_scale * utility(g.P_s, p.F)
    v = np.zeros(g.n_S)
    history = []
    for k in range(1, max_iter + 1):
        t0 = time.perf_counter()
        q = _q_values(g, nxt, v, reward)
        v_new = q.max(axis=1)
        diff = float(np.abs(v_new - v).max())
        v = v_new
        history.append((k, diff, time.perf_counter() - t0))
        if progress and k % 20 == 0:
            progress(f"Value iteration {k}: max change {diff:.2e}")
        if diff < tol * (1 - p.gamma) / p.gamma:
            break
    q = _q_values(g, nxt, v, reward)
    policy = np.argmax(q, axis=1)
    return Solution(p, g, policy, v, k, history, time.perf_counter() - t_all, "value_iteration")


# --------------------------------------------------------------------------
# Monte-Carlo simulation
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    P: np.ndarray            # (n_paths, T+1) cohort pension
    V: np.ndarray            # (n_paths, T+1) cohort wealth
    rel_P: np.ndarray        # individual relative pension  P_t / (e^{-lam t} P0)
    rel_V: np.ndarray        # V_t / V0
    ccr: np.ndarray          # (n_paths, T+1) total CCR
    action: np.ndarray       # (n_paths, T)
    n_cuts: np.ndarray
    n_enh: np.ndarray
    P0: float
    V0: float
    strategy: str
    params: Params = None


def simulate(p: Params, sol=None, n_paths=10_000, T=10, seed=42, const_action=None, z_paths=None):
    """Monte-Carlo simulation. Use `sol` for the optimal policy or const_action for a benchmark.
    z_paths: optional (n_paths, T) array of shocks (used for the bull/bear/sideways scenarios)."""
    rng = np.random.default_rng(seed)
    P0 = p.P0()
    V = np.full(n_paths, p.V0)
    P = np.full(n_paths, P0)
    Ps = np.empty((n_paths, T + 1)); Vs = np.empty((n_paths, T + 1)); Cs = np.empty((n_paths, T + 1))
    acts = np.empty((n_paths, T))
    Ps[:, 0], Vs[:, 0], Cs[:, 0] = P, V, V * p.rl / P
    cuts = np.zeros(n_paths, dtype=int); enh = np.zeros(n_paths, dtype=int)
    for t in range(T):
        ccr = np.clip(V * p.rl / P, p.ccr_min, p.ccr_max)
        if const_action is not None:
            a = np.full(n_paths, float(const_action))
        else:
            a = sol.action_at(V, ccr)
        z = z_paths[:, t] if z_paths is not None else rng.standard_normal(n_paths)
        V, P, ratio = step(p, V, P, a, z)
        cuts += ratio < 1 - 1e-9
        enh += ratio > 1 + 1e-9
        Ps[:, t + 1], Vs[:, t + 1], acts[:, t] = P, V, a
        Cs[:, t + 1] = V * p.rl / P
    tt = np.arange(T + 1)
    rel_P = Ps / (np.exp(-p.lam * tt)[None, :] * P0)
    rel_V = Vs / p.V0
    name = "optimal policy" if const_action is None else f"constant a={const_action:g}"
    return SimResult(Ps, Vs, rel_P, rel_V, Cs, acts, cuts, enh, P0, p.V0, name, p)


def metrics(res: SimResult, p: Params):
    rp = res.rel_P[:, -1]
    rv = res.rel_V[:, -1]
    avg_p = res.rel_P[:, 1:].mean(axis=1)
    return {
        "Strategy": res.strategy,
        "Initial pension P0": res.P0,
        "P(>=1 pension cut)": float((res.n_cuts >= 1).mean()),
        "P(#enh >= #cuts)": float((res.n_enh >= res.n_cuts).mean()),
        "P(avg pension > P0)": float((avg_p > 1.0).mean()),
        "Rel. pension mean (T)": float(rp.mean()),
        "Rel. pension std (T)": float(rp.std()),
        "Rel. pension VaR 5%": float(np.quantile(rp, 0.05)),
        "Rel. pension VaR 1%": float(np.quantile(rp, 0.01)),
        "Rel. wealth mean (T)": float(rv.mean()),
        "Rel. wealth std (T)": float(rv.std()),
    }
