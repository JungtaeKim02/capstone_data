"""Feature selection under the EXPANDED held-out set: 5 diverse no-load test
trajectories (4 different KRR excitation seeds + 1 different-TYPE point-to-point).

Methodology (overfitting-safe):
  - SELECTION criterion = regime-CV R2 inside the Traj-A train pool (run_1..5).
    The 5 test trajectories are NEVER used to choose features -> they stay a
    genuine held-out generalization confirmation.
  - REPORT, at every greedy step, the mean and worst R2 across the 5 held-out
    trajectories (so we can watch generalization track the in-train CV).
  - STABILITY (repeated-experiment evidence): re-run the whole greedy selection
    5 times under leave-one-train-run-out; a feature that keeps appearing in the
    top-10 regardless of which training run is removed is not a fit to one run.
  - PER-TRAJECTORY breakdown: the final set must generalize to ALL 5 held-out
    shapes individually, including the different-type p2p, not just on average.

NO fabricated numbers -- everything is computed from the recorded CSVs.
Writes selected5.json.
"""
import json
import numpy as np
import featlib as F
from featlib import LIB, EXC, GJ, DEFAULTS, channels, build_X
from search import design_pool, GramCV, K_FOLDS

CANDS = [n for n in LIB if n != "bias"]
KNEE_GAIN = 0.002

# 5 held-out no-load trajectories (diverse seeds + a different trajectory TYPE)
TESTS = ["test_noload_2", "test_noload_3", "test_noload_4", "test_noload_5", "test_noload_p2p"]


def load_pool_runs(P=DEFAULTS):
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def load_tests(P=DEFAULTS):
    return {t: channels(f"{EXC}/{t}/excitation_recording.csv", P) for t in TESTS}


def main():
    print("loading train pool (run_1..5) + 5 held-out test trajectories ...")
    chs, steps = load_pool_runs()
    tests = load_tests()

    # per-joint train Gram engines on the full candidate library
    grams = {}
    for j in GJ:
        X, y, folds = design_pool(chs, steps, j, CANDS, DEFAULTS)
        grams[j] = GramCV(X, y, folds, alpha=10.0)

    # precompute the full 43-col raw design for every (test, joint) ONCE
    Xtest = {t: {j: build_X(CANDS, tests[t], j, DEFAULTS) for j in GJ} for t in TESTS}

    def test_r2_per_traj(idx):
        """R2 of the train-fit subset on each held-out trajectory (mean over joints)."""
        out = {}
        bj = {j: grams[j].fit_full(idx) for j in GJ}
        for t in TESTS:
            r2s = []
            for j in GJ:
                mu, sd = grams[j].mu[idx], grams[j].sd[idx]
                Xt = (Xtest[t][j][:, idx] - mu) / sd
                yh = grams[j].ybar + Xt @ bj[j]
                y = tests[t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            out[t] = float(np.mean(r2s))
        return out

    def mean_cv(idx):
        rs = [grams[j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(np.max([w for _, w in rs]))

    def summ_test(idx):
        d = test_r2_per_traj(idx)
        vals = np.array(list(d.values()))
        return float(vals.mean()), float(vals.min()), d

    # ---- reference sets ----
    refs = {
        "posture-only(10)": [n for n in F.X0_LOAD_32 if LIB[n]["group"] == "posture"],
        "posture+history": [n for n in F.X0_LOAD_32 if LIB[n]["group"] in ("posture", "history")],
        "x0-31 (production)": [n for n in F.X0_LOAD_32 if n != "bias"],
    }
    print(f"\n=== reference sets under 5 held-out trajectories ===")
    print(f"  {'set':<22}{'nfeat':>6}{'meanCV':>8}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    for nm, sub in refs.items():
        idx = [CANDS.index(n) for n in sub]
        cv, w = mean_cv(idx)
        mt, wt, _ = summ_test(idx)
        print(f"  {nm:<22}{len(idx):>6}{cv:>8.4f}{w:>8.3f}{mt:>9.4f}{wt:>10.4f}")

    # ---- unified greedy forward selection (criterion = regime-CV only) ----
    print(f"\n=== greedy forward selection (criterion: mean regime-CV; test = held-out report) ===")
    print(f"{'step add-feature':<30}{'group':<11}{'meanCV':>8}{'gain':>8}{'worstN':>7}{'meanTest':>9}{'worstTest':>10}")
    chosen, path, best = [], [], -1e9
    while len(chosen) < 14:
        scored = []
        for c in range(len(CANDS)):
            if c in chosen:
                continue
            r2, _ = mean_cv(chosen + [c])
            scored.append((r2, c))
        scored.sort(reverse=True)
        r2, c = scored[0]
        gain = r2 - best
        chosen.append(c); best = r2
        idx = chosen[:]
        _, worst = mean_cv(idx)
        mt, wt, _ = summ_test(idx)
        path.append(dict(step=len(chosen), feat=CANDS[c], group=LIB[CANDS[c]]["group"],
                         meanCV=r2, gain=gain, worstN=worst, meanTest=mt, worstTest=wt))
        print(f"{len(chosen):>2} {CANDS[c]:<26}{LIB[CANDS[c]]['group']:<11}"
              f"{r2:>8.4f}{gain:>+8.4f}{worst:>7.3f}{mt:>9.4f}{wt:>10.4f}")

    # knee: first step after which all further CV gains < KNEE_GAIN
    knee = len(path)
    for i in range(len(path)):
        if all(p["gain"] < KNEE_GAIN for p in path[i:]):
            knee = path[i]["step"] - 1
            break
    knee = max(knee, 1)
    sel = [CANDS[i] for i in chosen[:knee]]
    print(f"\nKNEE at {knee} features (further regime-CV gains < {KNEE_GAIN}):")
    print("  " + ", ".join(sel))

    # ---- per-trajectory R2 of the selected set ----
    idx_sel = [CANDS.index(n) for n in sel]
    mt, wt, per = summ_test(idx_sel)
    print(f"\n=== selected-{knee} per-trajectory held-out R2 ===")
    for t in TESTS:
        kind = "p2p (diff TYPE)" if "p2p" in t else "excitation"
        print(f"  {t:<18}{per[t]:>8.4f}   {kind}")
    print(f"  {'MEAN':<18}{mt:>8.4f}   worst={wt:.4f}")

    # ---- STABILITY: leave-one-train-run-out re-selection ----
    print(f"\n=== leave-one-train-run-out selection stability (top-{knee} each) ===")
    appear = {n: 0 for n in CANDS}
    loo_sets = []
    for hold in range(5):
        sub_chs = [chs[i] for i in range(5) if i != hold]
        sub_st = [steps[i] for i in range(5) if i != hold]
        gloo = {}
        for j in GJ:
            X, y, folds = design_pool(sub_chs, sub_st, j, CANDS, DEFAULTS)
            gloo[j] = GramCV(X, y, folds, alpha=10.0)

        def mcv(idx):
            return float(np.mean([gloo[j].cv_score(idx)[0] for j in GJ]))

        ch2, bst = [], -1e9
        while len(ch2) < knee:
            sc = sorted(((mcv(ch2 + [c]), c) for c in range(len(CANDS)) if c not in ch2), reverse=True)
            ch2.append(sc[0][1])
        s2 = [CANDS[i] for i in ch2]
        loo_sets.append(s2)
        for n in s2:
            appear[n] += 1
        print(f"  hold run_{hold+1}: {', '.join(s2)}")

    print(f"\n  feature appearance across 5 leave-one-out runs (of 5):")
    core = [n for n in sel if appear[n] == 5]
    for n in sorted(CANDS, key=lambda x: -appear[x]):
        if appear[n] > 0:
            mark = "  <-- in selected" if n in sel else ""
            print(f"    {appear[n]}/5  {n:<26}{LIB[n]['group']:<11}{mark}")
    print(f"\n  STABLE CORE (in selected AND appears 5/5): {', '.join(core)}")

    json.dump(dict(selected=sel, knee=knee, core=core,
                   meanTest=mt, worstTest=wt, per_traj=per,
                   loo_sets=loo_sets, appearance=appear, path=path),
              open("selected5.json", "w"), indent=2)
    print("\nwrote selected5.json")


if __name__ == "__main__":
    main()
