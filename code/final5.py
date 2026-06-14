"""Finalize the model under the EXPANDED 5-trajectory held-out set.

  1. Build train Grams under BOTH default and CV-optimal constants (honest 2x2).
  2. Re-run greedy selection UNDER the optimal constants -> confirm the 10 picks
     are stable to the constant choice (closes the feature<->constant loop).
  3. Honest 2x2: {x0-31, slim-10} x {default, optimal} with 5-trajectory test R2.
  4. Per-trajectory generalization of the FINAL model (must hold on all 5).
  5. Payload sensitivity (residual must still grow with mass) under optimal P.
  6. Write final_model5.json (features + constants + metrics + per-traj).

NO fabricated numbers -- everything computed from the recorded CSVs.
"""
import json
import numpy as np
import pandas as pd
import featlib as F
from featlib import LIB, EXC, GJ, channels, build_X, rms, DEFAULTS
from search import design_pool, GramCV
from sel5 import TESTS, load_pool_runs, load_tests

ALPHA = 10.0   # ridge: CV was flat over 1..100 -> keep production value for deploy robustness
P_OPT = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
CANDS = [n for n in LIB if n != "bias"]
SEL10 = json.load(open("selected5.json"))["selected"]


def main():
    print("CV-optimal constants: sg_va=61, sg_vp=161, lag_a=25, fc_slow=0.02, alpha=10\n")
    print("building train Grams + 5 test designs under default AND optimal constants ...")
    G, Xtest, tests_by = {}, {}, {}
    for tag, P in (("def", DEFAULTS), ("opt", P_OPT)):
        chs, steps = load_pool_runs(P)
        tests = load_tests(P)
        tests_by[tag] = tests
        G[tag] = {}
        for j in GJ:
            X, y, folds = design_pool(chs, steps, j, CANDS, P)
            G[tag][j] = GramCV(X, y, folds, ALPHA)
        Xtest[tag] = {t: {j: build_X(CANDS, tests[t], j, P) for j in GJ} for t in TESTS}

    def mean_cv(names, tag):
        idx = [CANDS.index(n) for n in names]
        rs = [G[tag][j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(max(w for _, w in rs))

    def test_per_traj(names, tag):
        idx = [CANDS.index(n) for n in names]
        bj = {j: G[tag][j].fit_full(idx) for j in GJ}
        out = {}
        for t in TESTS:
            r2s = []
            for j in GJ:
                mu, sd = G[tag][j].mu[idx], G[tag][j].sd[idx]
                Xt = (Xtest[tag][t][j][:, idx] - mu) / sd
                yh = G[tag][j].ybar + Xt @ bj[j]
                y = tests_by[tag][t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            out[t] = float(np.mean(r2s))
        return out

    def mean_test(names, tag):
        d = test_per_traj(names, tag)
        v = np.array(list(d.values()))
        return float(v.mean()), float(v.min())

    # --- re-run greedy under optimal constants (confirm stability) ---
    print("\nRe-selection under OPTIMAL constants (first 12 greedy picks):")
    chosen, best = [], -1e9
    for _ in range(12):
        sc = sorted(((mean_cv([CANDS[i] for i in chosen] + [CANDS[c]], "opt")[0], c)
                     for c in range(len(CANDS)) if c not in chosen), reverse=True)
        r2, c = sc[0]; chosen.append(c)
        print(f"  {len(chosen):>2} {CANDS[c]:<26}{LIB[CANDS[c]]['group']:<11}meanCV={r2:.4f} (+{r2-best:.4f})")
        best = r2
    reselected = [CANDS[i] for i in chosen]
    same = set(reselected[:10]) == set(SEL10)
    print(f"  re-selected top-10 == default-constant selection? {same}")
    if not same:
        print(f"    only in opt: {set(reselected[:10]) - set(SEL10)}")
        print(f"    only in def: {set(SEL10) - set(reselected[:10])}")

    # --- honest 2x2 with 5-trajectory test ---
    x0_31 = [n for n in F.X0_LOAD_32 if n != "bias"]
    print("\n=== model x constants (mean over 5 held-out trajectories) ===")
    print(f"  {'model / constants':<30}{'nfeat':>6}{'meanCV':>8}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    for label, names, tag in (
        ("x0-31  @ default (current)", x0_31, "def"),
        ("slim-10 @ default", SEL10, "def"),
        ("x0-31  @ optimal", x0_31, "opt"),
        ("slim-10 @ optimal (FINAL)", SEL10, "opt"),
        ("re-selected-10 @ optimal", reselected[:10], "opt"),
    ):
        cv, w = mean_cv(names, tag)
        mt, wt = mean_test(names, tag)
        print(f"  {label:<30}{len(names):>6}{cv:>8.4f}{w:>8.3f}{mt:>9.4f}{wt:>10.4f}")

    # --- per-trajectory generalization of the FINAL model ---
    per = test_per_traj(SEL10, "opt")
    print("\n=== FINAL slim-10 @ optimal: per-trajectory held-out R2 ===")
    for t in TESTS:
        kind = "p2p (DIFFERENT TYPE)" if "p2p" in t else "excitation (diff seed)"
        print(f"  {t:<18}{per[t]:>8.4f}   {kind}")

    # --- payload sensitivity under optimal constants ---
    print("\nPAYLOAD SENSITIVITY (slim-10 @ optimal): dr(m) must grow with mass")
    grams = G["opt"]; idx = [CANDS.index(n) for n in SEL10]
    refA, refS = channels(f"{EXC}/run_1/excitation_recording.csv", P_OPT, keep_step=True)
    pays = {m: channels(f"{EXC}/payload_{m}g/run_1/excitation_recording.csv", P_OPT, keep_step=True)
            for m in (84, 146, 227)}
    W30 = int(30.0 / F.DT)

    def resid(ch, j):
        b = grams[j].fit_full(idx)
        X = (build_X(SEL10, ch, j, P_OPT) - grams[j].mu[idx]) / grams[j].sd[idx]
        return ch["load"][j] - (grams[j].ybar + X @ b)

    print(f"  {'joint':<6}{'dr(84g)':>9}{'dr(146g)':>10}{'dr(227g)':>10}{'monotone?':>11}")
    for j in GJ:
        rr = pd.Series(resid(refA, j), index=refS)
        d = []
        for m in (84, 146, 227):
            ch_p, st_p = pays[m]
            rp = pd.Series(resid(ch_p, j), index=st_p)
            df = pd.concat([rr.rename("r"), rp.rename("p")], axis=1, join="inner").dropna().iloc[W30:]
            d.append(rms((df["p"] - df["r"]).to_numpy()))
        mono = "yes" if d[0] < d[1] < d[2] else "NO"
        print(f"  J{j:<5}{d[0]:>9.2f}{d[1]:>10.2f}{d[2]:>10.2f}{mono:>11}")

    cv, w = mean_cv(SEL10, "opt"); mt, wt = mean_test(SEL10, "opt")
    json.dump(dict(features=SEL10, reselected_under_opt=reselected[:10],
                   stable_to_constants=bool(same),
                   constants=dict(sg_v=61, sg_a=61, sg_vp=161, lag_a=25, lag_b=150,
                                  fc_slow=0.02, fc_fast=0.2, fc_strib=0.1, alpha=ALPHA),
                   meanCV_R2=cv, worstNRMSE=w, meanTest_R2=mt, worstTest_R2=wt,
                   per_traj=per),
              open("final_model5.json", "w"), indent=2)
    print("\nwrote final_model5.json")


if __name__ == "__main__":
    main()
