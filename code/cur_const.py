"""STAGE 3 -- constant optimisation for the slim current model.

The dominant feature |load_pred| inherits the load model's already-optimised
constants (P_LOAD).  The only current-specific knobs with leverage are alpha and
the velocity smoothing sg_v used by the |vL|, vL^2 friction-magnitude terms.
Small grid, regime-CV optimum, 6 held-out as witness.

NO fabricated numbers.  Writes results/cur_const.txt + cur_constants.json
"""
import json
import numpy as np
import gc
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X
from search import GramCV, K_FOLDS

TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]
SEL = json.load(open("selected5.json"))["selected"]
C = json.load(open("constants6.json"))
P_LOAD = {**DEFAULTS, "sg_v": C["sg_v"], "sg_a": C["sg_a"], "sg_as": C["sg_as"],
          "sg_vp": C["sg_vp"], "lag_a": C["lag_a"], "fc_slow": C["fc_slow"]}
SELC = ["absLpred", "absvL", "Lpred2", "vL2", "absG_c4", "absG_c3"]

OUT = open("results/cur_const.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def cols(ch, j):
    q = ch["q"]
    return {
        "absLpred": np.abs(ch["Lpred"][j]),
        "Lpred2":   ch["Lpred"][j] ** 2,
        "absvL":    np.abs(ch["vL"][j]),
        "vL2":      ch["vL"][j] ** 2,
        "absG_c4":  np.abs(np.cos(q[2] + q[3] + q[4])),
        "absG_c3":  np.abs(np.cos(q[2] + q[3])),
    }


def design(ch, j):
    cc = cols(ch, j)
    return np.column_stack([cc[n] for n in SELC])


_LP = {}
def get_load_fits():
    """slim-10 load model per joint, once (P_LOAD)."""
    if _LP:
        return _LP
    chsL, steps = [], []
    for i in range(1, 6):
        chL, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_LOAD, keep_step=True)
        chsL.append(chL); steps.append(st)
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    folds = [np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1) for st in steps]
    for j in GJ:
        Xs, ys, fs = [], [], []
        for chL, fo in zip(chsL, folds):
            Xs.append(build_X(SEL, chL, j, P_LOAD)); ys.append(chL["load"][j]); fs.append(fo)
        _LP[j] = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), 10.0)
    _LP["steps"] = steps; _LP["folds"] = folds; _LP["edges"] = edges
    return _LP


def inject(ch_def, ch_load):
    ch_def["Lpred"] = {}
    for j in GJ:
        g = _LP[j]; b = g.fit_full(list(range(len(SEL))))
        X = (build_X(SEL, ch_load, j, P_LOAD) - g.mu) / g.sd
        ch_def["Lpred"][j] = g.ybar + X @ b


_CACHE = {}
def get_data(sg_v):
    """default-constants channels at a given sg_v (size-1 cache to bound RAM)."""
    if sg_v not in _CACHE:
        _CACHE.clear(); gc.collect()
        P = {**DEFAULTS, "sg_v": sg_v}
        chs, steps = [], []
        for i in range(1, 6):
            ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
            chL = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_LOAD)
            inject(ch, chL)
            chs.append(ch); steps.append(st)
        tests = {}
        for t in TESTS:
            ch = channels(f"{EXC}/{t}/excitation_recording.csv", P)
            inject(ch, channels(f"{EXC}/{t}/excitation_recording.csv", P_LOAD))
            tests[t] = ch
        _CACHE[sg_v] = (chs, steps, tests)
    return _CACHE[sg_v]


def metrics(sg_v, alpha):
    chs, steps, tests = get_data(sg_v)
    edges = _LP["edges"]
    folds = [np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1) for st in steps]
    cvs, per = [], {t: [] for t in TESTS}
    idx = list(range(len(SELC)))
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, fo in zip(chs, folds):
            Xs.append(design(ch, j)); ys.append(ch["cur"][j]); fs.append(fo)
        g = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), alpha)
        cvs.append(g.cv_score(idx)[0]); b = g.fit_full(idx)
        for t in TESTS:
            Xt = (design(tests[t], j) - g.mu) / g.sd
            yh = g.ybar + Xt @ b; y = tests[t]["cur"][j]
            per[t].append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    mt = float(np.mean([np.mean(per[t]) for t in TESTS]))
    return float(np.mean(cvs)), mt


def main():
    get_load_fits()
    tee("=" * 70)
    tee("STAGE 3  current-model constants (slim: " + ", ".join(SELC) + ")")
    tee("=" * 70)
    tee("\n--- alpha grid (sg_v=41) ---")
    tee(f"  {'alpha':>8}{'meanCV':>9}{'meanTest':>10}")
    bestA = None
    for a in (3.0, 10.0, 30.0, 100.0):
        cv, mt = metrics(41, a)
        tee(f"  {a:>8.0f}{cv:>9.4f}{mt:>10.4f}")
        if bestA is None or cv > bestA[1]:
            bestA = (a, cv, mt)
    alpha = bestA[0]
    tee(f"  -> alpha={alpha}")

    tee(f"\n--- sg_v grid (alpha={alpha}) ---")
    tee(f"  {'sg_v':>8}{'meanCV':>9}{'meanTest':>10}")
    bestV = None
    for v in (41, 61, 81, 91):
        cv, mt = metrics(v, alpha)
        tee(f"  {v:>8}{cv:>9.4f}{mt:>10.4f}")
        if bestV is None or cv > bestV[1]:
            bestV = (v, cv, mt)
    tee(f"  -> sg_v={bestV[0]}  meanCV={bestV[1]:.4f} meanTest={bestV[2]:.4f}")

    json.dump(dict(features=SELC, alpha=alpha, sg_v=bestV[0],
                   meanCV=bestV[1], meanTest=bestV[2]),
              open("cur_constants.json", "w"), indent=2)
    tee("\nwrote cur_constants.json")
    OUT.close()


if __name__ == "__main__":
    main()
