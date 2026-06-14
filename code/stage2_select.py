"""STAGE 2 -- re-run the WHOLE feature selection on the EXTENDED library
(base + cross-joint inertia) under the CV-optimal constants, judged on the full
set of 6 held-out trajectories.

Same overfitting-safe protocol as sel5:
  - SELECTION criterion = regime-CV R2 inside the Traj-A train pool ONLY.
  - The 6 test trajectories are pure held-out generalisation report.
  - leave-one-train-run-out re-selection = stability / repeated-experiment evidence.
Goal: does the new cross-joint inertia family earn a place, and does the optimal
10-set change?  Compare head-to-head against the current slim-10.

NO fabricated numbers.  Writes results/stage2_selection.txt + selected_ext.json
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels
import featlib_ext as E
from featlib_ext import EXTLIB, CANDS_EXT, build_Xe
from search import GramCV, K_FOLDS

P_OPT = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
ALPHA = 10.0
KNEE_GAIN = 0.002
SEL10 = json.load(open("selected5.json"))["selected"]
TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]

OUT = open("results/stage2_selection.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def load_pool_runs(P):
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def load_tests(P):
    return {t: channels(f"{EXC}/{t}/excitation_recording.csv", P) for t in TESTS}


def design_pool_e(chs, steps, j, names, P):
    Xs, ys, fs = [], [], []
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    for ch, st in zip(chs, steps):
        Xs.append(build_Xe(names, ch, j, P)); ys.append(ch["load"][j])
        fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(fs)


def main():
    tee("=" * 84)
    tee("STAGE 2  EXTENDED-LIBRARY RE-SELECTION  (6 held-out trajectories, P_OPT)")
    tee(f"  candidates = {len(CANDS_EXT)} (base 43 + cross-joint inertia 9)")
    tee(f"  new family (xinertia): {[n for n in CANDS_EXT if EXTLIB[n]['group']=='xinertia']}")
    tee(f"  tests (6): {TESTS}")
    tee("=" * 84)

    chs, steps = load_pool_runs(P_OPT)
    tests = load_tests(P_OPT)
    grams = {}
    for j in GJ:
        X, y, folds = design_pool_e(chs, steps, j, CANDS_EXT, P_OPT)
        grams[j] = GramCV(X, y, folds, ALPHA)
    Xtest = {t: {j: build_Xe(CANDS_EXT, tests[t], j, P_OPT) for j in GJ} for t in TESTS}

    def mean_cv(idx):
        rs = [grams[j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(np.max([w for _, w in rs]))

    def test_per_traj(idx):
        bj = {j: grams[j].fit_full(idx) for j in GJ}; out = {}
        for t in TESTS:
            r2s = []
            for j in GJ:
                mu, sd = grams[j].mu[idx], grams[j].sd[idx]
                yh = grams[j].ybar + ((Xtest[t][j][:, idx] - mu) / sd) @ bj[j]
                y = tests[t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            out[t] = float(np.mean(r2s))
        return out

    def summ_test(idx):
        d = test_per_traj(idx); v = np.array(list(d.values()))
        return float(v.mean()), float(v.min()), d

    # ---- greedy forward selection on the extended pool ----
    tee("\n=== greedy forward selection (criterion: mean regime-CV; report=6 held-out) ===")
    tee(f"{'step add-feature':<30}{'group':<11}{'meanCV':>8}{'gain':>8}{'worstN':>7}{'mTest6':>8}{'wTest6':>8}")
    chosen, path, best = [], [], -1e9
    while len(chosen) < 16:
        scored = sorted(((mean_cv(chosen + [c])[0], c) for c in range(len(CANDS_EXT)) if c not in chosen), reverse=True)
        r2, c = scored[0]; gain = r2 - best; chosen.append(c); best = r2
        _, worst = mean_cv(chosen)
        mt, wt, _ = summ_test(chosen)
        path.append(dict(step=len(chosen), feat=CANDS_EXT[c], group=EXTLIB[CANDS_EXT[c]]["group"],
                         meanCV=r2, gain=gain, worstN=worst, meanTest=mt, worstTest=wt))
        tee(f"{len(chosen):>2} {CANDS_EXT[c]:<26}{EXTLIB[CANDS_EXT[c]]['group']:<11}"
            f"{r2:>8.4f}{gain:>+8.4f}{worst:>7.3f}{mt:>8.4f}{wt:>8.4f}")

    knee = len(path)
    for i in range(len(path)):
        if all(p["gain"] < KNEE_GAIN for p in path[i:]):
            knee = path[i]["step"] - 1; break
    knee = max(knee, 1)
    sel = [CANDS_EXT[i] for i in chosen[:knee]]
    tee(f"\nKNEE at {knee} features (further regime-CV gains < {KNEE_GAIN}):")
    tee("  " + ", ".join(sel))
    new_feats = [n for n in sel if n not in SEL10]
    dropped = [n for n in SEL10 if n not in sel]
    tee(f"  vs slim-10 -> NEW: {new_feats or '(none)'}   DROPPED: {dropped or '(none)'}")

    # ---- head-to-head: slim-10 vs new selection on 6 tests ----
    tee("\n=== head-to-head on 6 held-out trajectories ===")
    tee(f"  {'model':<26}{'nfeat':>6}{'meanCV':>8}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    for label, names in (("slim-10 (current)", SEL10), (f"extended-{knee} (new)", sel)):
        idx = [CANDS_EXT.index(n) for n in names]
        cv, w = mean_cv(idx); mt, wt, _ = summ_test(idx)
        tee(f"  {label:<26}{len(names):>6}{cv:>8.4f}{w:>8.3f}{mt:>9.4f}{wt:>10.4f}")

    # per-traj of the new selection
    _, _, per = summ_test([CANDS_EXT.index(n) for n in sel])
    tee(f"\n=== extended-{knee} per-trajectory held-out R2 ===")
    for t in TESTS:
        kind = "p2p (diff TYPE)" if "p2p" in t else "excitation"
        tee(f"  {t:<18}{per[t]:>8.4f}   {kind}")

    # ---- leave-one-train-run-out stability ----
    tee(f"\n=== leave-one-train-run-out re-selection stability (top-{knee}) ===")
    appear = {n: 0 for n in CANDS_EXT}; loo_sets = []
    for hold in range(5):
        sc_, st_ = [chs[i] for i in range(5) if i != hold], [steps[i] for i in range(5) if i != hold]
        gloo = {}
        for j in GJ:
            X, y, folds = design_pool_e(sc_, st_, j, CANDS_EXT, P_OPT)
            gloo[j] = GramCV(X, y, folds, ALPHA)
        def mcv(idx): return float(np.mean([gloo[j].cv_score(idx)[0] for j in GJ]))
        ch2 = []
        while len(ch2) < knee:
            ch2.append(sorted(((mcv(ch2 + [c]), c) for c in range(len(CANDS_EXT)) if c not in ch2), reverse=True)[0][1])
        s2 = [CANDS_EXT[i] for i in ch2]; loo_sets.append(s2)
        for n in s2: appear[n] += 1
        tee(f"  hold run_{hold+1}: {', '.join(s2)}")
    core = [n for n in sel if appear[n] == 5]
    tee(f"\n  appearance across 5 LOO runs:")
    for n in sorted(CANDS_EXT, key=lambda x: -appear[x]):
        if appear[n] > 0:
            tee(f"    {appear[n]}/5  {n:<14}{EXTLIB[n]['group']:<11}{'<-- selected' if n in sel else ''}")
    tee(f"\n  STABLE CORE (selected AND 5/5): {', '.join(core)}")

    json.dump(dict(selected=sel, knee=knee, core=core, new_vs_slim10=new_feats,
                   dropped_vs_slim10=dropped, per_traj=per, path=path,
                   loo_sets=loo_sets, appearance=appear, tests=TESTS),
              open("selected_ext.json", "w"), indent=2)
    tee("\nwrote selected_ext.json")
    OUT.close()


if __name__ == "__main__":
    main()
