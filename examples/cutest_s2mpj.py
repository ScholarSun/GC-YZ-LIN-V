"""Reproduce entries of the README benchmark: GC-YZ-LIN and GC-YZ-V on CUTEst problems from
benchmarks/cutest_n30_problems.txt, built with S2MPJ (https://github.com/GrattonToint/S2MPJ).

    git clone https://github.com/GrattonToint/S2MPJ
    python examples/cutest_s2mpj.py --s2mpj ./S2MPJ                      # frame 1, three small problems
    python examples/cutest_s2mpj.py --s2mpj ./S2MPJ --seed 3 WATSON      # another frame, chosen problems
    python examples/cutest_s2mpj.py --s2mpj ./S2MPJ --all                 # the whole list (many hours)

The README data profiles average the start frames listed in the reference file. Frame s draws a random orthogonal matrix Q after
numpy.random.seed(s) and starts from the rotated design x0 +/- 0.5 q_i (gcyz option init_frame='random'); this
script does the same for --seed s, with the benchmark's options (the defaults, with the small-gradient test
switched off: tr_toexpand=0) and a budget of 100 simplex gradients, i.e.
100 (n + 1) function evaluations. For each tolerance tau it prints how many simplex gradients (evaluations /
(n + 1)) the run needed to reach f <= f_ref + tau (f(x0) - f_ref), where f_ref is the best value any of the three
benchmarked solvers reached in that frame ('-' = not reached within the budget); f_ref and the problem sizes come
from benchmarks/cutest_n30_reference.json. The data profiles in the README are the fraction of problems whose
count is at most the value on the x-axis, averaged over the frames. Stock NEWUOA is not run here (it is not part
of this package); its counts are in the benchmark data. Runs are deterministic: they match the benchmark runs bit
for bit with the same BLAS kernel and one thread (OPENBLAS_NUM_THREADS=1).
"""
import argparse
import contextlib
import importlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

import gcyz

REFERENCE = Path(__file__).resolve().parents[1] / "benchmarks" / "cutest_n30_reference.json"
TAUS = (1e-2, 1e-4, 1e-6)
DEFAULT_PROBLEMS = ("WATSON", "BROYDN3DLS", "ARWHEAD")


def build(s2mpj, name, arg):
    root = Path(s2mpj).resolve()
    for d in (root / "python_problems", root):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    cls = getattr(importlib.import_module(name), name)
    with contextlib.redirect_stdout(io.StringIO()):
        return cls() if arg is None else cls(arg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", help=f"problem names (default: {' '.join(DEFAULT_PROBLEMS)})")
    ap.add_argument("--s2mpj", required=True, help="path to an S2MPJ checkout (contains s2mpjlib.py)")
    ap.add_argument("--seed", type=int, default=1, help="start frame of the README benchmark (default 1; see the reference file for the frames)")
    ap.add_argument("--all", action="store_true", help="run every problem in the list")
    args = ap.parse_args()
    ref = json.loads(REFERENCE.read_text())
    table, budget_sg = ref["problems"], ref["budget_simplex_gradients"]
    names = sorted(table) if args.all else (args.problems or list(DEFAULT_PROBLEMS))
    frame = f"R{args.seed}"
    print(f"start frame {frame} (numpy.random.seed({args.seed}), init_frame='random'); budget {budget_sg} simplex gradients")
    print(f"{'':<12s} {'':>4s}  {'':<10s} {'':>12s} {'':>6s}  {'simplex gradients to reach tau':^36s}")
    print(f"{'problem':<12s} {'n':>4s}  {'solver':<10s} {'best f':>12s} {'evals':>6s}  "
          + "".join(f"{f'{t:.0e}':>12s}" for t in TAUS))
    for name in names:
        entry = table.get(name)
        if entry is None:
            sys.exit(f"{name}: not in the reference file (see benchmarks/cutest_n30_problems.txt for the list)")
        n, arg, fref = entry["n"], entry["arg"], entry["f_ref"].get(frame)
        if fref is None:
            sys.exit(f"{name}: no reference value for frame {frame}")
        prob = build(args.s2mpj, name, arg)
        assert prob.n == n, f"{name}: S2MPJ built n={prob.n}, the reference file says {n}"
        x0 = np.asarray(prob.x0, dtype=float).ravel()
        fun = lambda x: float(np.asarray(prob.fx(np.asarray(x, dtype=float))).ravel()[0])
        budget = budget_sg * (n + 1)
        for variant in ("LIN", "V"):
            t0 = time.time()
            np.random.seed(args.seed)
            res = gcyz.minimize(fun, x0, budget, variant=variant, options={"init_frame": "random", "tr_toexpand": 0.0})
            # the stopping test runs at the end of an iteration, so a run can evaluate a point or two past
            # the budget; as in the benchmark, only the first `budget` values count
            h = np.minimum.accumulate(np.asarray(res.fval_history, dtype=float)[:budget])
            f0 = h[0]
            reach = []
            for tau in TAUS:
                hit = np.nonzero(h <= fref + tau * (f0 - fref))[0]
                reach.append(f"{(hit[0] + 1) / (n + 1):12.2f}" if len(hit) else f"{'-':>12s}")
            print(f"{name:<12s} {n:>4d}  GC-YZ-{variant:<4s} {h[-1]:12.4e} {len(h):>6d}  "
                  + "".join(reach) + f"   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
