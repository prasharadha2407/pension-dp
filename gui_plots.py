"""Pure-matplotlib drawing functions used by the GUI (no Tk dependency, so they can be tested headless)."""
import numpy as np

from pension_core import simulate

COLORS = {0.0: "#1f77b4", 0.2: "#ff7f0e", 0.4: "#2ca02c"}


def _clear(fig, nrows=1, ncols=1):
    fig.clear()
    return fig.subplots(nrows, ncols, squeeze=False)


def draw_policy(fig, sol):
    """Heat-map of a*(V, CCR) and the average a* / total risky share against CCR (paper Fig. 1)."""
    p, g = sol.params, sol.grids
    axs = _clear(fig, 1, 2)
    pol = sol.policy_grid()                                   # (n_V, n_ccr)
    ax = axs[0, 0]
    im = ax.imshow(pol.T, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                   extent=[p.V_min, p.V_max, p.ccr_min * 100, p.ccr_max * 100])
    ax.set_xlabel("Wealth V"); ax.set_ylabel("CCR (%)")
    ax.set_title("Optimal risky share a*(V, CCR)")
    fig.colorbar(im, ax=ax, label="a*")
    ax2 = axs[0, 1]
    ccr = g.ccr * 100
    a_avg = pol.mean(axis=0)
    ccr_total = g.ccr
    pi_total = ((1 - p.alpha) * ccr_total + p.alpha) / ccr_total * a_avg
    ax2.plot(ccr, a_avg, "-o", ms=3, label="a*  (investment portfolio)")
    ax2.plot(ccr, pi_total, "-s", ms=3, label="pi* (total portfolio)")
    ax2.set_ylim(-0.02, 1.02); ax2.set_xlabel("CCR (%)"); ax2.set_ylabel("average risky share")
    ax2.set_title("Average optimal allocation vs CCR"); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.tight_layout()


def _fan(ax, x, data, color, label):
    q = np.quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(x, q[0], q[4], color=color, alpha=.15, label="5-95%")
    ax.fill_between(x, q[1], q[3], color=color, alpha=.30, label="25-75%")
    ax.plot(x, q[2], color=color, lw=2, label="median")
    ax.plot(x, data.mean(axis=0), "k--", lw=1.2, label="mean")
    ax.set_xlabel("years"); ax.set_ylabel(label); ax.grid(alpha=.3)


def draw_simulation(fig, res):
    axs = _clear(fig, 2, 2)
    T = res.rel_P.shape[1] - 1
    x = np.arange(T + 1)
    _fan(axs[0, 0], x, res.rel_P, "#1f77b4", "relative pension")
    axs[0, 0].axhline(1, color="grey", lw=.8); axs[0, 0].set_title("Relative individual pension"); axs[0, 0].legend(fontsize=7)
    _fan(axs[0, 1], x, res.rel_V, "#2ca02c", "relative wealth")
    axs[0, 1].axhline(1, color="grey", lw=.8); axs[0, 1].set_title("Relative cohort wealth")
    _fan(axs[1, 0], x, res.ccr * 100, "#9467bd", "CCR (%)")
    axs[1, 0].axhline(100, color="r", lw=.8); axs[1, 0].axhline(125, color="r", lw=.8); axs[1, 0].set_title("Capital coverage ratio")
    ax = axs[1, 1]
    ax.hist(res.rel_P[:, -1], bins=60, color="#1f77b4", alpha=.8)
    ax.axvline(1, color="k", lw=.8); ax.set_title(f"Terminal relative pension (t={T})")
    ax.set_xlabel("relative pension"); ax.set_ylabel("paths")
    fig.suptitle(f"{res.strategy}  -  {res.rel_P.shape[0]:,} paths", fontsize=10)
    fig.tight_layout()


def scenario_shocks(T):
    """Illustrative bullish / bearish / sideways shock sequences (cf. paper Figs 2-5)."""
    t = np.arange(T)
    bull = np.full(T, 0.9) + 0.4 * np.sin(1.7 * t)
    bear = np.full(T, -0.9) + 0.4 * np.sin(1.7 * t)
    side = 1.2 * np.where(t % 2 == 0, 1.0, -1.0)
    return {"Bullish": bull, "Bearish": bear, "Sideways": side}


def draw_scenarios(fig, p, solutions, T=10):
    """solutions: {alpha: Solution}. Rows: price index, relative pension, CCR, a*."""
    shocks = scenario_shocks(T)
    axs = _clear(fig, 4, 3)
    for c, (name, z) in enumerate(shocks.items()):
        px = np.concatenate([[1.0], np.cumprod(1 + p.mu + p.sigma * z)])
        axs[0, c].plot(px, "k-o", ms=3); axs[0, c].set_title(name)
        if c == 0: axs[0, c].set_ylabel("risky asset index")
        for alpha, sol in solutions.items():
            pa = sol.params
            res = simulate(pa, sol, n_paths=1, T=T, z_paths=z[None, :])
            col = COLORS.get(round(alpha, 1), None)
            lab = f"alpha={alpha:g}"
            axs[1, c].step(range(T + 1), res.rel_P[0], where="post", color=col, label=lab)
            axs[2, c].plot(res.ccr[0] * 100, color=col, label=lab)
            axs[3, c].step(range(T), res.action[0], where="post", color=col, label=lab)
        if c == 0:
            axs[1, c].set_ylabel("rel. pension"); axs[2, c].set_ylabel("CCR (%)"); axs[3, c].set_ylabel("a*")
            axs[1, c].legend(fontsize=7)
        axs[2, c].axhline(100, color="r", lw=.6); axs[2, c].axhline(125, color="r", lw=.6)
        axs[3, c].set_ylim(-0.05, 1.05); axs[3, c].set_xlabel("years")
        for r in range(4): axs[r, c].grid(alpha=.3)
    fig.tight_layout()


def draw_benchmark(fig, pi_sol, vi_sol):
    axs = _clear(fig, 1, 3)
    agree = float((pi_sol.policy == vi_sol.policy).mean())
    ax = axs[0, 0]
    ax.bar(["Policy iteration", "Value iteration"], [pi_sol.seconds, vi_sol.seconds], color=["#1f77b4", "#d62728"])
    ax.set_ylabel("seconds"); ax.set_title(f"Runtime  (speed-up {vi_sol.seconds / max(pi_sol.seconds, 1e-9):.1f}x)")
    for i, (s, it) in enumerate([(pi_sol, pi_sol.iterations), (vi_sol, vi_sol.iterations)]):
        ax.text(i, s.seconds, f"{it} iters", ha="center", va="bottom", fontsize=9)
    ax = axs[0, 1]
    ks = [h[0] for h in pi_sol.history]; ch = [max(h[1], 0.5) for h in pi_sol.history]
    ax.semilogy(ks, ch, "-o"); ax.set_xlabel("PI iteration"); ax.set_ylabel("states whose action changed")
    ax.set_title("Policy iteration convergence"); ax.grid(alpha=.3, which="both")
    ax = axs[0, 2]
    ks = [h[0] for h in vi_sol.history]; d = [h[1] for h in vi_sol.history]
    ax.semilogy(ks, d); ax.set_xlabel("VI iteration"); ax.set_ylabel("max |V_new - V|")
    ax.set_title(f"Value iteration (contraction)  |  policy agreement {agree * 100:.2f}%"); ax.grid(alpha=.3, which="both")
    fig.tight_layout()


def draw_alpha_comparison(fig, results):
    """results: {alpha: SimResult} - density of path-wise average pension / wealth (paper Fig. 8)."""
    axs = _clear(fig, 1, 3)
    for alpha, res in results.items():
        col = COLORS.get(round(alpha, 1), None)
        avg_p = res.rel_P[:, 1:].mean(axis=1)
        avg_v = res.rel_V[:, 1:].mean(axis=1)
        axs[0, 0].hist(avg_p, bins=70, density=True, histtype="step", lw=1.6, color=col, label=f"alpha={alpha:g}")
        axs[0, 1].hist(avg_v, bins=70, density=True, histtype="step", lw=1.6, color=col, label=f"alpha={alpha:g}")
        axs[0, 2].plot(np.quantile(res.rel_P, 0.05, axis=0), color=col, label=f"alpha={alpha:g}")
    axs[0, 0].set_title("Path-wise average relative pension"); axs[0, 1].set_title("Path-wise average relative wealth")
    axs[0, 2].set_title("5% quantile of relative pension"); axs[0, 2].set_xlabel("years")
    for ax in axs[0]:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
