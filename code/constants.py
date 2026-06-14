"""Experimental justification of the NUMERIC CONSTANTS inside the selected
features. For each constant we sweep a physically-plausible range, refit the
SELECTED feature set, and score by regime-CV (mean over J2/J3/J4) + held-out
Traj-B. The chosen value is the CV optimum, not a hand-pick.

Constants covered (and their physical meaning):
  fc_slow : Coulomb sign / stiction velocity scale (rad/s)   -> tanh_vP, coul_*
  lag_a   : drivetrain transport / group delay (samples*3ms) -> lag_vL_a, lag_aL_a
  sg_va   : SG window for velocity & acceleration (taps)     -> vL, aL, adiff, ...
  sg_vp   : SG window for the long-band velocity (taps)      -> tanh_vP
  alpha   : ridge regularization strength
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, channels, build_X, DEFAULTS
from search import design_pool, GramCV

SEL = json.load(open("selected_features.json"))["selected"]
IDX = list(range(len(SEL)))


def load_pool_P(P):
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def metrics(P, alpha, chs, steps, chB):
    cvs, worsts, Bs = [], [], []
    for j in GJ:
        X, y, folds = design_pool(chs, steps, j, SEL, P)
        g = GramCV(X, y, folds, alpha)
        r2, w = g.cv_score(IDX)
        b = g.fit_full(IDX)
        Xt = (build_X(SEL, chB, j, P) - g.mu) / g.sd
        yhat = g.ybar + Xt @ b
        yb = chB["load"][j]
        Bs.append(1 - np.sum((yb - yhat) ** 2) / np.sum((yb - yb.mean()) ** 2))
        cvs.append(r2); worsts.append(w)
    return float(np.mean(cvs)), float(max(worsts)), float(np.mean(Bs))


def sweep(name, values, make_P, alpha=10.0, rebuild_channels=False,
          chs0=None, steps0=None, chB0=None):
    print(f"\n--- {name} ---")
    print(f"  {'value':>8}{'meanCV_R2':>11}{'worstNRMSE':>12}{'meanB_R2':>10}")
    best = None
    for v in values:
        P = make_P(v)
        if rebuild_channels:
            chs, steps = load_pool_P(P)
            chB = channels(f"{EXC}/test_noload/excitation_recording.csv", P)
        else:
            chs, steps, chB = chs0, steps0, chB0
        cv, worst, B = metrics(P, alpha, chs, steps, chB)
        star = ""
        if best is None or cv > best[1]:
            best = (v, cv, worst, B)
        print(f"  {str(v):>8}{cv:>11.4f}{worst:>12.3f}{B:>10.4f}")
    print(f"  -> CV-optimal {name} = {best[0]}  (meanCV_R2={best[1]:.4f}, meanB_R2={best[3]:.4f})")
    return best[0]


def main():
    print("selected features:", ", ".join(SEL))
    chs0, steps0 = load_pool_P(DEFAULTS)
    chB0 = channels(f"{EXC}/test_noload/excitation_recording.csv", DEFAULTS)

    base = dict(DEFAULTS)
    # constants that do NOT change SG channels -> reuse cached channels
    fc = sweep("fc_slow (Coulomb velocity scale, rad/s)",
               [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
               lambda v: {**base, "fc_slow": v},
               chs0=chs0, steps0=steps0, chB0=chB0)
    la = sweep("lag_a (transport delay, samples x3ms)",
               [10, 25, 50, 75, 100, 150],
               lambda v: {**base, "lag_a": v},
               chs0=chs0, steps0=steps0, chB0=chB0)
    # alpha varies the regularizer (not P) -> dedicated loop
    print("\n--- alpha (ridge strength) ---")
    print(f"  {'value':>8}{'meanCV_R2':>11}{'worstNRMSE':>12}{'meanB_R2':>10}")
    best_a = None
    for a in [1.0, 3.0, 10.0, 30.0, 100.0]:
        cv, worst, B = metrics(base, a, chs0, steps0, chB0)
        if best_a is None or cv > best_a[1]:
            best_a = (a, cv, worst, B)
        print(f"  {a:>8}{cv:>11.4f}{worst:>12.3f}{B:>10.4f}")
    print(f"  -> CV-optimal alpha = {best_a[0]}  (meanCV_R2={best_a[1]:.4f}, meanB_R2={best_a[3]:.4f})")

    # SG-window constants -> must rebuild channels
    sg = sweep("sg_va (SG window for vel/accel, taps)",
               [11, 21, 31, 41, 61, 81],
               lambda v: {**base, "sg_v": v, "sg_a": v},
               rebuild_channels=True)
    sgp = sweep("sg_vp (SG window for long-band velocity, taps)",
                [41, 61, 81, 121, 161],
                lambda v: {**base, "sg_vp": v},
                rebuild_channels=True)

    print("\n=== CV-optimal constants vs current defaults ===")
    cur = DEFAULTS
    for k, v in (("fc_slow", fc), ("lag_a", la), ("alpha", best_a[0]),
                 ("sg_va", sg), ("sg_vp", sgp)):
        d = cur.get(k if k != "sg_va" else "sg_v", 10.0 if k == "alpha" else None)
        print(f"  {k:<10} default={d}   CV-optimal={v}")


if __name__ == "__main__":
    main()
