"""Unified (joint-shared) feature selection for the x0 LOAD model.

Production x0 uses ONE feature function for J2/J3/J4 (only the fitted
coefficients differ). So we select ONE feature list that is best ON AVERAGE
across the three joints, by greedy forward selection on regime-CV R2 (mean over
joints). We track, at every step:
  - mean / worst regime-CV R2 inside Traj-A
  - mean Traj-B (held-out shape) R2  -> guards against overfitting to Traj-A
and finally validate PAYLOAD SENSITIVITY (residual must still grow with mass,
i.e. the slimmer feature set must NOT absorb the external-force signal).

NO fabricated numbers. Writes selected_features.json for downstream steps.
"""
import json
import numpy as np
import featlib as F
from featlib import LIB, EXC, GJ, channels, build_X, rms
from search import load_pool, design_pool, GramCV, K_FOLDS

CANDS = [n for n in LIB if n != "bias"]
KNEE_GAIN = 0.002          # stop-of-interest: mean cvR2 gain below this == plateau


def resid(gram, names_sel, ch, j):
    """Residual of the (fit-on-A) model on any recording, joint j."""
    idx = [CANDS.index(n) for n in names_sel]
    b = gram.fit_full(idx)
    X = build_X(names_sel, ch, j)
    X = (X - gram.mu[idx]) / gram.sd[idx]
    return ch["load"][j] - (gram.ybar + X @ b)


def main():
    print("loading Traj-A pool (run_1..5) + held-out shapes ...")
    chs, steps = load_pool()
    ch_B = channels(f"{EXC}/test_noload/excitation_recording.csv")

    # per-joint Gram engines on the full candidate library
    grams = {}
    for j in GJ:
        X, y, folds = design_pool(chs, steps, j, CANDS)
        grams[j] = GramCV(X, y, folds)

    def mean_cv(idx):
        rs = [grams[j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(np.max([w for _, w in rs]))

    def mean_B(names_sel):
        idx = [CANDS.index(n) for n in names_sel]
        out = []
        for j in GJ:
            b = grams[j].fit_full(idx)
            Xt = build_X(names_sel, ch_B, j)
            Xt = (Xt - grams[j].mu[idx]) / grams[j].sd[idx]
            yhat = grams[j].ybar + Xt @ b
            y = ch_B["load"][j]
            out.append(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
        return float(np.mean(out))

    # reference: production 31 non-bias features, same harness
    x0_31 = [n for n in F.X0_LOAD_32 if n != "bias"]
    r2_31, worst_31 = mean_cv([CANDS.index(n) for n in x0_31])
    B_31 = mean_B(x0_31)
    print(f"\nREFERENCE x0 (31 non-bias): meanCV_R2={r2_31:.4f} worstNRMSE={worst_31:.3f} "
          f"meanB_R2={B_31:.4f}")

    # ---- unified greedy forward selection ----
    print("\nUNIFIED forward selection (greedy on mean regime-CV R2 over J2,J3,J4)")
    print(f"{'step add-feature':<30}{'group':<12}{'meanCV':>8}{'gain':>8}{'worstN':>8}{'meanB':>8}")
    chosen, path, best = [], [], -1e9
    while len(chosen) < 20:
        scored = []
        for c in range(len(CANDS)):
            if c in chosen:
                continue
            r2, worst = mean_cv(chosen + [c])
            scored.append((r2, worst, c))
        scored.sort(reverse=True)
        r2, worst, c = scored[0]
        gain = r2 - best
        chosen.append(c); best = r2
        names_sel = [CANDS[i] for i in chosen]
        B = mean_B(names_sel)
        path.append(dict(step=len(chosen), feat=CANDS[c], group=LIB[CANDS[c]]["group"],
                         meanCV=r2, gain=gain, worstN=worst, meanB=B))
        print(f"{len(chosen):>2} {CANDS[c]:<26}{LIB[CANDS[c]]['group']:<12}"
              f"{r2:>8.4f}{gain:>+8.4f}{worst:>8.3f}{B:>8.4f}")

    # knee = first step after which all further gains < KNEE_GAIN
    knee = len(path)
    for i in range(len(path)):
        if all(p["gain"] < KNEE_GAIN for p in path[i:]):
            knee = path[i]["step"] - 1
            break
    knee = max(knee, 1)
    sel = [CANDS[i] for i in chosen[:knee]]
    print(f"\nKNEE at {knee} features (further gains < {KNEE_GAIN}):")
    print("  " + ", ".join(sel))
    sr2, sworst = mean_cv([CANDS.index(n) for n in sel])
    sB = mean_B(sel)
    print(f"  selected: meanCV_R2={sr2:.4f} worstNRMSE={sworst:.3f} meanB_R2={sB:.4f}  "
          f"(vs x0-31: {r2_31:.4f}/{worst_31:.3f}/{B_31:.4f})")

    # ---- payload sensitivity: slim set must keep detecting external load ----
    print("\nPAYLOAD SENSITIVITY  -- RMS of payload-induced residual increment dr(m)")
    print("  dr = resid(payload run_1) - resid(no-load run_1), aligned by step_idx, skip 30s")
    refA, refS = channels(f"{EXC}/run_1/excitation_recording.csv", keep_step=True)
    pays = {m: channels(f"{EXC}/payload_{m}g/run_1/excitation_recording.csv", keep_step=True)
            for m in (84, 146, 227)}
    W30 = int(30.0 / F.DT)
    import pandas as pd

    def dr_rms(names_sel, j, m):
        ch_p, st_p = pays[m]
        rr = pd.Series(resid(grams[j], names_sel, refA, j), index=refS)
        rp = pd.Series(resid(grams[j], names_sel, ch_p, j), index=st_p)
        d = pd.concat([rr.rename("r"), rp.rename("p")], axis=1, join="inner").dropna().iloc[W30:]
        return rms((d["p"] - d["r"]).to_numpy())

    print(f"  {'model':<14}{'joint':<6}{'dr(84g)':>9}{'dr(146g)':>10}{'dr(227g)':>10}{'monotone?':>11}")
    for label, names_sel in (("slim", sel), ("x0-31", x0_31)):
        for j in GJ:
            d = [dr_rms(names_sel, j, m) for m in (84, 146, 227)]
            mono = "yes" if d[0] < d[1] < d[2] else "NO"
            print(f"  {label:<14}J{j:<5}{d[0]:>9.2f}{d[1]:>10.2f}{d[2]:>10.2f}{mono:>11}")

    json.dump(dict(selected=sel, knee=knee,
                   meanCV_R2=sr2, worstNRMSE=sworst, meanB_R2=sB,
                   ref_x0_31=dict(meanCV_R2=r2_31, worstNRMSE=worst_31, meanB_R2=B_31),
                   path=path),
              open("selected_features.json", "w"), indent=2)
    print("\nwrote selected_features.json")


if __name__ == "__main__":
    main()
