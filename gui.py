"""
gui.py - desktop GUI for the pension-fund optimisation model
(Lichtenstern & Zagst, Eur. Actuar. J. 2022).

Run:  python gui.py
Needs: numpy, scipy, matplotlib (Tkinter ships with the standard Python installers).
"""
import os
import queue
import threading
import traceback
import tkinter as tk
from dataclasses import replace
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from pension_core import Params, policy_iteration, value_iteration, simulate, metrics
from backward_induction import backward_induction
import gui_plots as gp

# Reference numbers quoted in the project README for paper Tables 1/2 (10,000 paths, 10 years).
PAPER_REF = {
    0.0: {
        "Initial pension P0": 277,
        "P(>=1 pension cut)": 0.495,
        "P(#enh >= #cuts)": 0.844,
        "P(avg pension > P0)": 0.741,
        "Rel. pension mean (T)": 1.075,
        "Rel. pension std (T)": 0.181,
        "Rel. pension VaR 5%": 0.833,
        "Rel. pension VaR 1%": 0.719,
        "Rel. wealth mean (T)": 0.966,
        "Rel. wealth std (T)": 0.157,
    },
    0.2: {
        "Initial pension P0": 270,
        "P(>=1 pension cut)": 0.364,
        "P(#enh >= #cuts)": 0.913,
        "P(avg pension > P0)": 0.808,
        "Rel. pension mean (T)": 1.084,
        "Rel. pension std (T)": 0.168,
        "Rel. pension VaR 5%": 0.848,
        "Rel. pension VaR 1%": 0.751,
        "Rel. wealth mean (T)": 0.962,
        "Rel. wealth std (T)": 0.143,
    },
    0.4: {
        "Initial pension P0": 258,
        "P(>=1 pension cut)": 0.255,
        "P(#enh >= #cuts)": 0.968,
        "P(avg pension > P0)": 0.861,
        "Rel. pension mean (T)": 1.109,
        "Rel. pension std (T)": 0.170,
        "Rel. pension VaR 5%": 0.865,
        "Rel. pension VaR 1%": 0.770,
        "Rel. wealth mean (T)": 0.965,
        "Rel. wealth std (T)": 0.146,
    },
}

FIELDS = [  # key, label, default
    ("r", "Risk-free rate r", "0.01"),
    ("mu", "Risky drift mu", "0.0297"),
    ("sigma", "Risky volatility sigma", "0.1175"),
    ("lam", "Mortality force lambda", "0.0212"),
    ("beta", "Utility discount beta", "0.03"),
    ("p_bar", "Target CCR p_bar", "1.125"),
    ("alpha", "Buffer alpha (0 - 0.5)", "0.2"),
    ("V0", "Initial wealth V0", "10000"),
    ("F", "Subsistence level F", "25.8"),
]
PRESETS = {"Paper-calibrated (lambda=0.0212)": 0.0212, "As printed (lambda=0.0118)": 0.0118}
GRID_CHOICES = ["100", "200", "500", "1000"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Pension Fund Optimiser - Dynamic Programming & Policy Iteration")
        root.geometry("1360x860")
        self.q = queue.Queue()
        self.busy = False
        self.sol = None
        self.solutions = {}
        self.vars = {}
        self._build_left()
        self._build_tabs()
        self.log("Ready. Click 'Solve' to compute the optimal policy.")
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Model parameters", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.preset = tk.StringVar(value=list(PRESETS)[0])
        cb = ttk.Combobox(left, textvariable=self.preset, values=list(PRESETS), state="readonly", width=32)
        cb.pack(fill="x", pady=(2, 6))
        cb.bind("<<ComboboxSelected>>", self._apply_preset)
        grid = ttk.Frame(left); grid.pack(fill="x")
        for i, (k, label, default) in enumerate(FIELDS):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", pady=1)
            v = tk.StringVar(value=default); self.vars[k] = v
            ttk.Entry(grid, textvariable=v, width=10).grid(row=i, column=1, padx=4)
        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="Numerics / simulation", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        g2 = ttk.Frame(left); g2.pack(fill="x")
        self.nV = tk.StringVar(value="200"); self.paths = tk.StringVar(value="10000")
        self.horizon = tk.StringVar(value="10"); self.seed = tk.StringVar(value="42")
        for i, (lab, var, widget) in enumerate([
                ("Wealth grid points n_V", self.nV, "combo"), ("Monte-Carlo paths", self.paths, "entry"),
                ("Horizon (years)", self.horizon, "entry"), ("Random seed", self.seed, "entry")]):
            ttk.Label(g2, text=lab).grid(row=i, column=0, sticky="w", pady=1)
            if widget == "combo":
                ttk.Combobox(g2, textvariable=var, values=GRID_CHOICES, width=8).grid(row=i, column=1, padx=4)
            else:
                ttk.Entry(g2, textvariable=var, width=10).grid(row=i, column=1, padx=4)
        ttk.Separator(left).pack(fill="x", pady=6)
        self.buttons = []
        for text, cmd in [("1  Solve (policy iteration)", self.on_solve),
                          ("2  Simulate fund (optimal vs constant mixes)", self.on_simulate),
                          ("3  Compare buffer alpha = 0 / 0.2 / 0.4", self.on_compare),
                          ("4  Benchmark: PI vs VI vs Backward Induction", self.on_benchmark),
                          ("Run everything", self.on_all),
                          ("Export results...", self.on_export)]:
            b = ttk.Button(left, text=text, command=cmd); b.pack(fill="x", pady=2); self.buttons.append(b)
        self.status = ttk.Label(left, text="idle", foreground="#555", wraplength=250)
        self.status.pack(anchor="w", pady=(8, 0))
        self.pbar = ttk.Progressbar(left, mode="indeterminate"); self.pbar.pack(fill="x", pady=4)

    def _make_tab(self, title):
        frame = ttk.Frame(self.nb); self.nb.add(frame, text=title)
        fig = Figure(figsize=(9, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        NavigationToolbar2Tk(canvas, frame).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def _build_tabs(self):
        self.nb = ttk.Notebook(self.root); self.nb.pack(side="right", fill="both", expand=True)
        self.t_policy = self._make_tab("Optimal policy")
        self.t_sim = self._make_tab("Simulation")
        self.t_scen = self._make_tab("Scenarios")
        self.t_alpha = self._make_tab("Alpha comparison")
        self.t_bench = self._make_tab("Algorithm benchmark")
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Metrics table")
        self.tree = ttk.Treeview(tab, show="headings"); self.tree.pack(fill="both", expand=True)
        note = ttk.Label(tab, foreground="#555", wraplength=900, text=(
            "'paper' columns are the reference values quoted in the project README for Tables 1/2. "
            "Differences are expected: see the README's note on the lambda discrepancy; "
            "paper values may pool over years; ours are terminal year t=10 (hypothesis, not verified)."))
        note.pack(anchor="w", padx=6, pady=4)
        logf = ttk.Frame(self.nb); self.nb.add(logf, text="Log")
        self.logbox = tk.Text(logf, height=10, state="disabled", wrap="word"); self.logbox.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ helpers
    def _apply_preset(self, _evt=None):
        self.vars["lam"].set(str(PRESETS[self.preset.get()]))

    def params(self, **override):
        try:
            kw = {k: float(v.get()) for k, v in self.vars.items()}
            kw["n_V"] = int(self.nV.get())
        except ValueError:
            raise ValueError("All parameter fields must be numbers.")
        kw.update(override)
        p = Params(**kw)
        p.validate()
        return p

    def sim_settings(self):
        return int(self.paths.get()), int(self.horizon.get()), int(self.seed.get())

    def log(self, msg):
        self.logbox.config(state="normal"); self.logbox.insert("end", msg + "\n")
        self.logbox.see("end"); self.logbox.config(state="disabled")

    def _set_busy(self, flag, text=None):
        self.busy = flag
        for b in self.buttons:
            b.state(["disabled"] if flag else ["!disabled"])
        if flag: self.pbar.start(12)
        else: self.pbar.stop()
        if text is not None: self.status.config(text=text)

    def run_async(self, work, done, label):
        """Run work() in a thread (keeps the GUI responsive); call done(result) on the Tk thread."""
        if self.busy:
            return
        self._set_busy(True, label)
        self.log(f"-- {label}")

        def target():
            try:
                self.q.put(("done", done, work()))
            except Exception as e:                                   # noqa
                self.q.put(("err", traceback.format_exc(), e))
        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, *rest = self.q.get_nowait()
                if kind == "log":
                    self.log(rest[0])
                elif kind == "done":
                    done, result = rest
                    self._set_busy(False, "done")
                    try:
                        done(result)
                    except Exception as e:                           # noqa
                        self.log(traceback.format_exc()); messagebox.showerror("Error", str(e))
                elif kind == "err":
                    tb, e = rest
                    self._set_busy(False, "error"); self.log(tb); messagebox.showerror("Error", str(e))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _progress(self, msg):
        self.q.put(("log", msg))

    def _show(self, tab):
        fig, canvas = tab
        canvas.draw_idle()

    def _table(self, columns, rows):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = columns
        for i, c in enumerate(columns):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=230 if i == 0 else 130, anchor="w" if i == 0 else "center")
        for r in rows:
            self.tree.insert("", "end", values=r)

    @staticmethod
    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    def _need_solution(self, p):
        if self.sol is not None and self.sol.params == p:
            return self.sol
        return policy_iteration(p, progress=self._progress)

    # ------------------------------------------------------------------ actions
    def on_solve(self, then=None):
        try:
            p = self.params()
        except Exception as e:
            return messagebox.showerror("Invalid input", str(e))

        def done(sol):
            self.sol = sol
            self.log(f"Policy iteration converged in {sol.iterations} iterations ({sol.seconds:.2f}s) "
                     f"on {sol.grids.n_S:,} states. Initial pension P0 = {p.P0():.2f}")
            gp.draw_policy(self.t_policy[0], sol); self._show(self.t_policy)
            self.nb.select(0)
            if callable(then): then()
        self.run_async(lambda: policy_iteration(p, progress=self._progress), done, "Solving MDP by policy iteration...")

    def on_simulate(self, then=None):
        try:
            p = self.params(); n, T, seed = self.sim_settings()
        except Exception as e:
            return messagebox.showerror("Invalid input", str(e))

        def work():
            sol = self._need_solution(p)
            res = [simulate(p, sol, n, T, seed)] + [simulate(p, None, n, T, seed, const_action=a) for a in (0.0, 0.5, 1.0)]
            return sol, res

        def done(out):
            sol, res = out
            self.sol = sol
            gp.draw_policy(self.t_policy[0], sol); self._show(self.t_policy)
            gp.draw_simulation(self.t_sim[0], res[0]); self._show(self.t_sim)
            ms = [metrics(r, p) for r in res]
            keys = [k for k in ms[0] if k != "Strategy"]
            self._table(["Metric"] + [m["Strategy"] for m in ms],
                        [[k] + [self._fmt(m[k]) for m in ms] for k in keys])
            self.log("Simulation finished.")
            self.nb.select(1)
            if callable(then): then()
        self.run_async(work, done, "Simulating the fund...")

    def on_compare(self, then=None):
        try:
            base = self.params(); n, T, seed = self.sim_settings()
        except Exception as e:
            return messagebox.showerror("Invalid input", str(e))
        alphas = (0.0, 0.2, 0.4)

        def work():
            sols, ress = {}, {}
            for a in alphas:
                pa = replace(base, alpha=a)
                self._progress(f"alpha = {a:g}")
                sols[a] = policy_iteration(pa, progress=self._progress)
                ress[a] = simulate(pa, sols[a], n, T, seed)
            return sols, ress

        def done(out):
            sols, ress = out
            self.solutions = sols
            gp.draw_scenarios(self.t_scen[0], base, sols, T); self._show(self.t_scen)
            gp.draw_alpha_comparison(self.t_alpha[0], ress); self._show(self.t_alpha)
            ms = {a: metrics(ress[a], replace(base, alpha=a)) for a in alphas}
            keys = [k for k in ms[0.0] if k not in ("Strategy",)]
            cols = ["Metric"]
            for a in alphas: cols += [f"alpha={a:g} (ours)", f"alpha={a:g} (paper)"]
            rows = []
            for k in keys:
                row = [k]
                for a in alphas:
                    row += [self._fmt(ms[a][k]), self._fmt(PAPER_REF[a].get(k, "-"))]
                rows.append(row)
            self._table(cols, rows)
            self.log("alpha comparison finished (iterations: " + ", ".join(f"a={a:g}: {sols[a].iterations}" for a in alphas) + ")")
            self.nb.select(3)
            if callable(then): then()
        self.run_async(work, done, "Comparing alpha values...")

    def on_benchmark(self, then=None):
        try:
            p = self.params()
            try:
                T = int(self.horizon.get())
            except (ValueError, AttributeError):
                T = 10
            if T <= 0:
                T = 10
        except Exception as e:
            return messagebox.showerror("Invalid input", str(e))

        def work():
            pi = policy_iteration(p, progress=self._progress)
            vi = value_iteration(p, progress=self._progress)
            bi = backward_induction(p, T=T, progress=self._progress)
            return pi, vi, bi

        def done(out):
            pi, vi, bi = out
            gp.draw_benchmark(self.t_bench[0], pi, vi, bi); self._show(self.t_bench)
            speed_vi_pi = vi.seconds / max(pi.seconds, 1e-9)
            speed_vi_bi = vi.seconds / max(bi.seconds, 1e-9)
            agree_vi = (pi.policy == vi.policy).mean() * 100
            agree_bi = (pi.policy == bi.policy).mean() * 100
            self.log(
                f"PI: {pi.iterations} iters, {pi.seconds:.2f}s | "
                f"VI: {vi.iterations} iters, {vi.seconds:.2f}s | "
                f"BI (T={T}): {bi.iterations} stages, {bi.seconds:.2f}s | "
                f"Speed-up VI/PI {speed_vi_pi:.1f}x, VI/BI {speed_vi_bi:.1f}x | "
                f"Policy agreement PI vs VI: {agree_vi:.2f}%, PI vs BI yr-0: {agree_bi:.2f}%"
            )
            self.nb.select(4)
            if callable(then): then()
        self.run_async(work, done, "Benchmarking PI vs VI vs Backward Induction...")

    def on_all(self):
        def step4(): self.on_benchmark()
        def step3(): self.root.after(50, lambda: self.on_compare(then=lambda: self.root.after(50, step4)))
        def step2(): self.root.after(50, lambda: self.on_simulate(then=step3))
        self.on_solve(then=step2)

    def on_export(self):
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        names = [("policy", self.t_policy), ("simulation", self.t_sim), ("scenarios", self.t_scen),
                 ("alpha_comparison", self.t_alpha), ("benchmark", self.t_bench)]
        n = 0
        for name, (fig, _) in names:
            if fig.axes:
                fig.savefig(os.path.join(folder, f"{name}.png"), dpi=150); n += 1
        if self.tree["columns"]:
            import csv
            with open(os.path.join(folder, "metrics.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(self.tree["columns"])
                for iid in self.tree.get_children():
                    w.writerow(self.tree.item(iid)["values"])
        self.log(f"Exported {n} figures (+ metrics.csv if available) to {folder}")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
