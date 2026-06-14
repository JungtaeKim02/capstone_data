"""Justify the NUMERIC CONSTANTS of the selected 10-feature model under the
EXPANDED 5-trajectory held-out set. For each constant we sweep a physically
plausible range, refit the selected set, and score by regime-CV (selection
metric) + mean R2 over the 5 held-out trajectories (generalization). The chosen
value is the CV optimum -- never a hand-pick.

Constants and their physical meaning (hardware-class level, not STS3215-specific):
  fc_slow : Coulomb sign / stiction velocity scale (rad/s) -> tanh_vP, coul_sin_c3
  lag_a   : drivetrain transport / group delay (samples*3ms) -> lag_vL_a, lag_aL_a
  sg_va   : SG window for velocity & acceleration (taps)   -> vL, aL, adiff
  sg_vp   : SG window for the long-band velocity (taps)    -> tanh_vP
  alpha   : ridge regularization strength
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, channels, build_X, DEFAULTS
from search import design_pool, GramCV
from sel5 import TESTS, load_pool_runs, load_tests

SEL = json.load(open("selected5.json"))["selected"]
IDX = list(range(len(SEL)))


def metrics(P, alpha, chs, steps, tests):
    cvs, worsts, test_means = [], [], []
    per = {t: [] for t in TESTS}
    for j in GJ:
        X, y, folds = design_pool(chs, steps, j, SEL, P)
        g = GramCV(X, y, folds, alpha)
        r2, w = g.cv_score(IDX)
        b = g.fit_full(IDX)
        cvs.append(r2); worsts.append(w)
        for t in TESTS:
            Xt = (build_X(SEL, tests[t], j, P) - g.mu) / g.sd
            yh = g.ybar + Xt @ b
            yb = tests[t]["load"][j]
            per[t].append(1 - np.sum((yb - yh) ** 2) / np.sum((yb - yb.mean()) ** 2))
    tmean = float(np.mean([np.mean(per[t]) for t in TESTS]))
    tworst = float(np.min([np.mean(per[t]) for t in TESTS]))
    return float(np.mean(cvs)), float(max(worsts)), tmean, tworst


def sweep(name, values, make_P, alpha=10.0, rebuild=False,
          chs0=None, steps0=None, tests0=None):
    print(f"\n--- {name} ---")
    print(f"  {'value':>8}{'meanCV':>9}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    best = None
    for v in values:
        P = make_P(v)
        if rebuild:
            chs, steps = load_pool_runs(P)
            tests = load_tests(P)
        else:
            chs, steps, tests = chs0, steps0, tests0
        cv, w, tm, tw = metrics(P, alpha, chs, steps, tests)
        if best is None or cv > best[1]:
            best = (v, cv, w, tm, tw)
        print(f"  {str(v):>8}{cv:>9.4f}{w:>8.3f}{tm:>9.4f}{tw:>10.4f}")
    print(f"  -> CV-optimal {name} = {best[0]}  (meanCV={best[1]:.4f}, meanTest={best[3]:.4f})")
    return best[0]


def main():
    print("selected 10:", ", ".join(SEL))
    chs0, steps0 = load_pool_runs(DEFAULTS)
    tests0 = load_tests(DEFAULTS)
    base = dict(DEFAULTS)

    fc = sweep("fc_slow (Coulomb velocity scale, rad/s)",
               [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
               lambda v: {**base, "fc_slow": v},
               chs0=chs0, steps0=steps0, tests0=tests0)
    la = sweep("lag_a (transport delay, samples x3ms)",
               [10, 25, 50, 75, 100, 150],
               lambda v: {**base, "lag_a": v},
               chs0=chs0, steps0=steps0, tests0=tests0)

    print("\n--- alpha (ridge strength) ---")
    print(f"  {'value':>8}{'meanCV':>9}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    best_a = None
    for a in [1.0, 3.0, 10.0, 30.0, 100.0]:
        cv, w, tm, tw = metrics(base, a, chs0, steps0, tests0)
        if best_a is None or cv > best_a[1]:
            best_a = (a, cv, w, tm, tw)
        print(f"  {a:>8}{cv:>9.4f}{w:>8.3f}{tm:>9.4f}{tw:>10.4f}")
    print(f"  -> CV-optimal alpha = {best_a[0]}  (meanCV={best_a[1]:.4f}, meanTest={best_a[3]:.4f})")

    sg = sweep("sg_va (SG window vel/accel, taps)",
               [11, 21, 31, 41, 61, 81],
               lambda v: {**base, "sg_v": v, "sg_a": v}, rebuild=True)
    sgp = sweep("sg_vp (SG window long-band velocity, taps)",
                [41, 61, 81, 121, 161, 201],
                lambda v: {**base, "sg_vp": v}, rebuild=True)

    print("\n=== CV-optimal constants vs current production defaults ===")
    for k, v, d in (("fc_slow", fc, DEFAULTS["fc_slow"]), ("lag_a", la, DEFAULTS["lag_a"]),
                    ("alpha", best_a[0], 10.0), ("sg_va", sg, DEFAULTS["sg_v"]),
                    ("sg_vp", sgp, DEFAULTS["sg_vp"])):
        print(f"  {k:<10} production={d:<6} CV-optimal={v}")

    json.dump(dict(fc_slow=fc, lag_a=la, alpha=best_a[0], sg_va=sg, sg_vp=sgp),
              open("constants5.json", "w"), indent=2)
    print("\nwrote constants5.json")


if __name__ == "__main__":
    main()
