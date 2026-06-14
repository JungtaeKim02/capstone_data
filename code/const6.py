"""STAGE 3 -- FINE constant optimisation for the slim-10, by coordinate descent
over the 6 held-out trajectories.  Directly answers "are the numbers (e.g. the
183ms = 61-tap SG window) actually optimal, or hand-picked?".

Each constant gets a fine grid; we sweep one at a time holding the others at the
current best, take the regime-CV optimum (the selection metric), and iterate two
full passes so couplings between constants settle.  meanTest over the 6 held-out
trajectories is reported alongside as the generalisation witness (never the
selection criterion).

Constants that matter for the slim-10 and their physical meaning:
  sg_v  : SG window for vL              (velocity smoothing)          -> 3ms/tap
  sg_a  : SG window for aL              (acceleration smoothing)
  sg_as : SG window for aS  (adiff = aS-aL, the fast-accel residual)
  sg_vp : SG window for vP  (tanh_vP, the long-band drivetrain memory)
  lag_a : transport/group delay for lag_vL_a, lag_aL_a (samples x3ms)
  fc_slow: Coulomb/stiction velocity scale (rad/s) for tanh_vP, coul_sin_c3
  alpha : ridge strength

NO fabricated numbers.  Writes results/const6.txt + constants6.json
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X
from search import GramCV, K_FOLDS

SEL = json.load(open("selected5.json"))["selected"]
IDX = list(range(len(SEL)))
TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]

OUT = open("results/const6.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")

import gc
_CACHE = {}
def get_data(P):
    """channels for pool+tests. SIZE-1 cache keyed by the SG-tuple: only the
    current smoothing is held in RAM (coordinate descent visits one SG at a time),
    so memory stays ~0.4GB instead of accumulating every SG-tuple (the freeze)."""
    key = (P["sg_v"], P["sg_a"], P["sg_vs"], P["sg_as"], P["sg_vp"])
    if key not in _CACHE:
        _CACHE.clear(); gc.collect()      # evict previous set BEFORE building new
        chs, steps = [], []
        for i in range(1, 6):
            ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
            chs.append(ch); steps.append(st)
        tests = {t: channels(f"{EXC}/{t}/excitation_recording.csv", P) for t in TESTS}
        _CACHE[key] = (chs, steps, tests)
    return _CACHE[key]


def metrics(P, alpha):
    chs, steps, tests = get_data(P)
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    cvs, worsts, per = [], [], {t: [] for t in TESTS}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, st in zip(chs, steps):
            Xs.append(build_X(SEL, ch, j, P)); ys.append(ch["load"][j])
            fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
        g = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), alpha)
        r2, w = g.cv_score(IDX); b = g.fit_full(IDX)
        cvs.append(r2); worsts.append(w)
        for t in TESTS:
            Xt = (build_X(SEL, tests[t], j, P) - g.mu) / g.sd
            yh = g.ybar + Xt @ b; y = tests[t]["load"][j]
            per[t].append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    tmean = float(np.mean([np.mean(per[t]) for t in TESTS]))
    tworst = float(np.min([np.mean(per[t]) for t in TESTS]))
    return float(np.mean(cvs)), float(max(worsts)), tmean, tworst


# (name, grid, is_alpha)
GRIDS = [
    ("fc_slow", [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06], False),
    ("lag_a",   [10, 15, 20, 25, 30, 35, 45, 60], False),
    ("alpha",   [3.0, 5.0, 10.0, 20.0, 40.0], True),
    ("sg_v",    [61, 71, 81, 91, 101, 121, 151], False),
    ("sg_a",    [41, 51, 61, 71, 81], False),
    ("sg_as",   [7, 11, 15, 21, 31], False),
    ("sg_vp",   [121, 141, 161, 181, 201, 241], False),
]


def main():
    tee("=" * 78)
    tee("STAGE 3  FINE CONSTANT OPTIMISATION (coordinate descent, 6 held-out)")
    tee("  slim-10:", ", ".join(SEL))
    tee("=" * 78)
    P = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
    alpha = 10.0
    cv0, w0, mt0, wt0 = metrics(P, alpha)
    tee(f"\nstart  meanCV={cv0:.4f} worstN={w0:.3f} meanTest={mt0:.4f} worstTest={wt0:.4f}")
    tee(f"  start constants: sg_v=61 sg_a=61 sg_as=11 sg_vp=161 lag_a=25 fc_slow=0.02 alpha=10")

    for it in range(2):
        tee(f"\n########## PASS {it+1} ##########")
        for name, grid, is_alpha in GRIDS:
            tee(f"\n--- {name}  (current={alpha if is_alpha else P[name]}) ---")
            tee(f"  {'value':>8}{'meanCV':>9}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
            best = None
            for v in grid:
                if is_alpha:
                    cv, w, mt, wt = metrics(P, v)
                else:
                    cv, w, mt, wt = metrics({**P, name: v}, alpha)
                star = ""
                if best is None or cv > best[1]:
                    best = (v, cv, w, mt, wt)
                tee(f"  {str(v):>8}{cv:>9.4f}{w:>8.3f}{mt:>9.4f}{wt:>10.4f}")
            tee(f"  -> CV-optimal {name} = {best[0]}  (meanCV={best[1]:.4f}, meanTest={best[3]:.4f})")
            if is_alpha:
                alpha = best[0]
            else:
                P[name] = best[0]

    cv, w, mt, wt = metrics(P, alpha)
    tee("\n" + "=" * 78)
    tee("CONVERGED constants (coordinate-descent optimum on regime-CV):")
    for k in ("sg_v", "sg_a", "sg_as", "sg_vp", "lag_a", "fc_slow"):
        tee(f"   {k:<9}= {P[k]:<6}  ({P[k]*3}ms)" if k.startswith(("sg", "lag")) else f"   {k:<9}= {P[k]}")
    tee(f"   alpha    = {alpha}")
    tee(f"\n  final  meanCV={cv:.4f} worstN={w:.3f} meanTest={mt:.4f} worstTest={wt:.4f}")
    tee(f"  start  meanCV={cv0:.4f}                  meanTest={mt0:.4f}")
    tee(f"  net gain over start: dCV={cv-cv0:+.4f}  dTest={mt-mt0:+.4f}")
    json.dump(dict(sg_v=P["sg_v"], sg_a=P["sg_a"], sg_as=P["sg_as"], sg_vp=P["sg_vp"],
                   lag_a=P["lag_a"], fc_slow=P["fc_slow"], alpha=alpha,
                   meanCV=cv, meanTest=mt, worstTest=wt),
              open("constants6.json", "w"), indent=2)
    tee("\nwrote constants6.json")
    OUT.close()


if __name__ == "__main__":
    main()
