"""
web_app.py - local web server for the pension planner.
Run:   python web_app.py      then open http://localhost:8000
Needs: pension_core.py and index.html in the same folder. No extra packages (stdlib http.server).
"""
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from pension_core import Params, policy_iteration, simulate, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = {}


def fan(a):
    q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return [x.round(4).tolist() for x in q] + [a.mean(axis=0).round(4).tolist()]


def setup(d, alpha=None):
    p = Params(r=float(d.get("r", 0.01)), mu=float(d.get("mu", 0.0297)), sigma=float(d.get("sigma", 0.1175)),
               lam=float(d.get("lam", 0.0212)), alpha=float(d.get("alpha", 0.2) if alpha is None else alpha),
               V0=float(d.get("V0", 10000)), n_V=int(d.get("n_V", 1000)))
    p.validate()
    T = max(1, min(30, int(d.get("T", 10))))
    key = repr(p)
    if key not in CACHE:
        CACHE[key] = policy_iteration(p)
    return p, CACHE[key], T


def run(d):
    p, sol, T = setup(d)
    opt = simulate(p, sol, 5000, T, 42)
    safe = simulate(p, None, 5000, T, 42, const_action=0.0)
    strat = {"Our plan": metrics(opt, p), "100% safe": metrics(safe, p)}
    for a in (0.25, 0.5, 1.0):
        strat[f"{int(a * 100)}% stocks"] = metrics(simulate(p, None, 5000, T, 42, const_action=a), p)
    t = np.arange(T)
    shocks = {"Boom": 0.9 + 0.4 * np.sin(1.7 * t), "Crash": -0.9 + 0.4 * np.sin(1.7 * t),
              "Flat": 1.2 * np.where(t % 2 == 0, 1.0, -1.0)}
    scen = {k: simulate(p, sol, 1, T, z_paths=z[None, :]).rel_P[0].round(4).tolist() for k, z in shocks.items()}
    pol = sol.policy_grid()
    ccr0 = np.array([p.V0 * p.rl / p.P0()])
    return {
        "p0": p.P0(), "iters": sol.iterations, "secs": round(sol.seconds, 2), "states": sol.grids.n_S, "T": T,
        "today_a": float(sol.action_at(np.array([p.V0]), ccr0)[0]),
        "ccr": (sol.grids.ccr * 100).round(2).tolist(), "avg_a": pol.mean(axis=0).round(4).tolist(),
        "opt": strat["Our plan"], "safe": strat["100% safe"], "strat": strat, "scen": scen,
        "heat": pol[::max(1, p.n_V // 60)].round(2).tolist(),
        "pension": fan(opt.rel_P), "wealth": fan(opt.rel_V),
    }


def compare(d):
    out = []
    for a in (0.0, 0.2, 0.4):
        p, sol, T = setup(d, alpha=a)
        out.append({"alpha": a, "iters": sol.iterations, "m": metrics(simulate(p, sol, 5000, T, 42), p)})
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        fn = {"/api/run": run, "/api/compare": compare}.get(self.path)
        if not fn:
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            self._send(200, json.dumps(fn(json.loads(self.rfile.read(n) or b"{}"))))
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Pension planner running at http://localhost:8000  (Ctrl+C to stop)")
    webbrowser.open("http://localhost:8000")
    srv.serve_forever()
