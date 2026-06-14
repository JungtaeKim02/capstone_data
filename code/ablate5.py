"""Per-feature MARGINAL importance on the 5 held-out trajectories, under the
optimal constants. Two complementary, overfitting-safe measures:

  forward-gain : regime-CV R2 increment when the feature was ADDED during greedy
                 selection (from selected5.json path) -- its value given the
                 features chosen before it.
  drop-one     : loss in mean held-out (5-traj) R2 when the feature is REMOVED
                 from the final 10 and the model refit -- its value given ALL
                 the others. Small drop = redundant; large drop = irreplaceable.

Reported together they bracket each feature's contribution (collinear features
read low on drop-one but their group still matters; forward-gain catches that).
"""
import json
import numpy as np
import featlib as F
from featlib import LIB, GJ, build_X, DEFAULTS
from search import design_pool, GramCV
from sel5 import TESTS, load_pool_runs, load_tests

P_OPT = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
ALPHA = 10.0
CANDS = [n for n in LIB if n != "bias"]
SEL = json.load(open("selected5.json"))["selected"]
PATH = {p["feat"]: p for p in json.load(open("selected5.json"))["path"]}


def main():
    chs, steps = load_pool_runs(P_OPT)
    tests = load_tests(P_OPT)
    G = {}
    for j in GJ:
        X, y, folds = design_pool(chs, steps, j, CANDS, P_OPT)
        G[j] = GramCV(X, y, folds, ALPHA)
    Xt = {t: {j: build_X(CANDS, tests[t], j, P_OPT) for j in GJ} for t in TESTS}

    def mean_test(names):
        idx = [CANDS.index(n) for n in names]
        bj = {j: G[j].fit_full(idx) for j in GJ}
        per = []
        for t in TESTS:
            r2s = []
            for j in GJ:
                mu, sd = G[j].mu[idx], G[j].sd[idx]
                yh = G[j].ybar + ((Xt[t][j][:, idx] - mu) / sd) @ bj[j]
                y = tests[t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            per.append(np.mean(r2s))
        return float(np.mean(per))

    full = mean_test(SEL)
    print(f"full slim-10 @ optimal: mean held-out R2 = {full:.4f}\n")
    print(f"  {'feature':<26}{'group':<11}{'fwd-gain':>9}{'drop-one dR2':>14}")
    rows = []
    for n in SEL:
        without = mean_test([m for m in SEL if m != n])
        drop = full - without
        fg = PATH[n]["gain"] if n in PATH else float("nan")
        rows.append((drop, n, LIB[n]["group"], fg))
    for drop, n, grp, fg in sorted(rows, reverse=True):
        print(f"  {n:<26}{grp:<11}{fg:>9.4f}{drop:>14.4f}")


if __name__ == "__main__":
    main()
