"""CURRENT open-ended feature search (Stage 1 discovery + Stage 2 honest validation).

Mirrors the load pipeline EXACTLY (same pool run_1..5, same 6 held-out, regime-CV
selection only).  current is an UNSIGNED magnitude channel, so every candidate is
even/magnitude form.

KEY hypothesis from the probe: current ~ |tau_motor| (rectified torque) + I0.
The production 21-feature model sums |term_i| (|sin|+|cos|+|aL|+...), but
|sum| != sum|.|.  The right feature is the MAGNITUDE OF THE TORQUE SUM, which we
get STATE-ONLY as |load_pred| (the slim-10 state->load model, then rectified).

Stage 1: residual of the 21-feature model -> partial-R2 of the missing magnitude
         features (esp. |load_pred|).
Stage 2: forward-select from scratch over the magnitude library by regime-CV;
         report 6 held-out R2 ceiling + knee.

NO fabricated numbers.  Writes results/cur_search.txt + cur_selected.json
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X, lag
from search import GramCV, K_FOLDS

TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]
SEL = json.load(open("selected5.json"))["selected"]
C = json.load(open("constants6.json"))
P_LOAD = {**DEFAULTS, "sg_v": C["sg_v"], "sg_a": C["sg_a"], "sg_as": C["sg_as"],
          "sg_vp": C["sg_vp"], "lag_a": C["lag_a"], "fc_slow": C["fc_slow"]}
ALPHA = 10.0

OUT = open("results/cur_search.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def feat_cur21(ch, j):
    """production 21-feature magnitude basis (baseline)."""
    q = ch["q"]; vL = ch["vL"][j]; aL = ch["aL"][j]
    c3, c4 = q[2] + q[3], q[2] + q[3] + q[4]; n = len(vL)
    cols = [np.ones(n)]
    for a in (q[2], q[3], q[4], c3, c4):
        cols += [np.abs(np.sin(a)), np.abs(np.cos(a))]
    cols += [np.abs(vL), vL * vL, np.abs(vL) * np.exp(-(vL / 0.1) ** 2)]
    cols += [np.abs(aL), np.abs(aL * np.cos(c3))]
    cols += [np.abs(ch["vS"][j]), np.abs(ch["aS"][j])]
    cols += [np.abs(np.tanh(ch["vP"][j] / 0.02)), np.abs(lag(vL, 50)), np.abs(lag(vL, 150))]
    return np.column_stack(cols)


# magnitude/even candidate library (state-only).  "Lpred" is injected per-joint.
MAG = {
    "absLpred":  lambda ch, j: np.abs(ch["Lpred"][j]),
    "Lpred2":    lambda ch, j: ch["Lpred"][j] ** 2,
    "absvL":     lambda ch, j: np.abs(ch["vL"][j]),
    "vL2":       lambda ch, j: ch["vL"][j] ** 2,
    "absvS":     lambda ch, j: np.abs(ch["vS"][j]),
    "strib_mag": lambda ch, j: np.abs(ch["vL"][j]) * np.exp(-(ch["vL"][j] / 0.1) ** 2),
    "absaL":     lambda ch, j: np.abs(ch["aL"][j]),
    "absaS":     lambda ch, j: np.abs(ch["aS"][j]),
    "absadiff":  lambda ch, j: np.abs(ch["aS"][j] - ch["aL"][j]),
    "abstanhvP": lambda ch, j: np.abs(np.tanh(ch["vP"][j] / 0.02)),
    "abslagvLa": lambda ch, j: np.abs(lag(ch["vL"][j], 50)),
    "absG_q2":   lambda ch, j: np.abs(np.sin(ch["q"][2])),
    "absG_c3":   lambda ch, j: np.abs(np.cos(ch["q"][2] + ch["q"][3])),
    "absG_c4":   lambda ch, j: np.abs(np.cos(ch["q"][2] + ch["q"][3] + ch["q"][4])),
    "absaLcos3": lambda ch, j: np.abs(ch["aL"][j] * np.cos(ch["q"][2] + ch["q"][3])),
}
MAGN = list(MAG)


def inject_Lpred(ch_def, ch_load, fL):
    """add state-only predicted (signed) load to the default-constants ch dict."""
    ch_def["Lpred"] = {}
    for j in GJ:
        g = fL[j]; b = g.fit_full(list(range(len(SEL))))
        X = (build_X(SEL, ch_load, j, P_LOAD) - g.mu) / g.sd
        ch_def["Lpred"][j] = g.ybar + X @ b


def design(ch, names, j):
    return np.column_stack([MAG[n](ch, j) for n in names])


def main():
    tee("=" * 84)
    tee("CURRENT OPEN-ENDED SEARCH  (pool run_1..5, 6 held-out, regime-CV selection)")
    tee("=" * 84)

    chs, steps = [], []
    chsL = []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", DEFAULTS, keep_step=True)
        chL = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_LOAD)
        chs.append(ch); steps.append(st); chsL.append(chL)
    tests = {t: channels(f"{EXC}/{t}/excitation_recording.csv", DEFAULTS) for t in TESTS}
    testsL = {t: channels(f"{EXC}/{t}/excitation_recording.csv", P_LOAD) for t in TESTS}

    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    folds = [np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1) for st in steps]

    # fit slim-10 load model (state->load) per joint, then inject Lpred everywhere
    fL = {}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for chL, st, fo in zip(chsL, steps, folds):
            Xs.append(build_X(SEL, chL, j, P_LOAD)); ys.append(chL["load"][j]); fs.append(fo)
        fL[j] = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)
    for ch, chL in zip(chs, chsL):
        inject_Lpred(ch, chL, fL)
    for t in TESTS:
        inject_Lpred(tests[t], testsL[t], fL)

    # ---------- Stage 1: what is the 21-feature model missing? ----------
    tee("\n=== STAGE 1: partial-R2 of magnitude features over the 21-feature residual ===")
    tee("  (partial-R2 = fraction of the 21-feat residual variance a feature would remove)")
    tee(f"  {'joint':<7}" + "".join(f"{n:>11}" for n in ["absLpred", "Lpred2", "absaS", "absadiff", "vL2"]))
    for j in GJ:
        # base residual from 21-feature OLS (ridge tiny) pooled
        B = np.vstack([feat_cur21(ch, j) for ch in chs])
        y = np.concatenate([ch["cur"][j] for ch in chs])
        Bc = B - B.mean(0); yc = y - y.mean()
        # ridge solve
        A = Bc.T @ Bc + 1e-3 * np.eye(Bc.shape[1])
        beta = np.linalg.solve(A, Bc.T @ yc)
        r = yc - Bc @ beta
        Q, _ = np.linalg.qr(Bc)             # orthonormal basis of base span
        prow = []
        for n in ["absLpred", "Lpred2", "absaS", "absadiff", "vL2"]:
            c = np.concatenate([MAG[n](ch, j) for ch in chs]).astype(float)
            c = c - c.mean()
            cperp = c - Q @ (Q.T @ c)        # orthogonalize against base
            denom = np.dot(cperp, cperp)
            pr = (np.dot(r, cperp) ** 2) / (denom * np.dot(r, r)) if denom > 1e-9 else 0.0
            prow.append(pr)
        tee(f"  J{j:<6}" + "".join(f"{v:>11.4f}" for v in prow))

    # ---------- Stage 2: forward selection over magnitude library ----------
    def gram_for(j):
        Xs, ys, fs = [], [], []
        for ch, fo in zip(chs, folds):
            Xs.append(design(ch, MAGN, j)); ys.append(ch["cur"][j]); fs.append(fo)
        return GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)

    grams = {j: gram_for(j) for j in GJ}

    def mean_cv(idxs):
        return float(np.mean([grams[j].cv_score(idxs)[0] for j in GJ]))

    def mean_test(idxs):
        per = []
        bj = {j: grams[j].fit_full(idxs) for j in GJ}
        for t in TESTS:
            r2s = []
            for j in GJ:
                Xt = (design(tests[t], [MAGN[i] for i in idxs], j) - grams[j].mu[idxs]) / grams[j].sd[idxs]
                yh = grams[j].ybar + Xt @ bj[j]; y = tests[t]["cur"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            per.append(np.mean(r2s))
        return float(np.mean(per)), float(np.min(per))

    tee("\n=== STAGE 2: forward selection over magnitude library (greedy regime-CV) ===")
    tee(f"  {'step add':<14}{'meanCV':>9}{'gain':>8}{'meanTest':>10}{'worstTest':>11}")
    chosen, prev = [], -9.9
    remaining = list(range(len(MAGN)))
    path = []
    while remaining and len(chosen) < 10:
        best = None
        for c in remaining:
            cv = mean_cv(chosen + [c])
            if best is None or cv > best[0]:
                best = (cv, c)
        cv, c = best
        chosen.append(c); remaining.remove(c)
        mt, wt = mean_test(chosen)
        path.append(dict(step=len(chosen), feat=MAGN[c], meanCV=cv, gain=cv - prev,
                         meanTest=mt, worstTest=wt))
        tee(f"  {MAGN[c]:<14}{cv:>9.4f}{cv-prev:>+8.4f}{mt:>10.4f}{wt:>11.4f}")
        prev = cv

    # single-feature reference: absLpred only
    i_lp = MAGN.index("absLpred")
    cv1 = mean_cv([i_lp]); mt1, wt1 = mean_test([i_lp])
    tee(f"\n  reference: |load_pred| ALONE  meanCV={cv1:.4f} meanTest={mt1:.4f} worstTest={wt1:.4f}")
    tee(f"  baseline : 21-feature sum-of-magnitudes meanTest ~ 0.106 (see cur_probe)")

    # knee at gain<0.003
    knee = next((p["step"] for p in path if p["gain"] < 0.003 and p["step"] > 1), len(path))
    sel = [MAGN[i] for i in chosen[:knee]]
    full_cv = mean_cv(chosen[:knee]); full_mt, full_wt = mean_test(chosen[:knee])
    tee(f"\n  KNEE at {knee} features (gain<0.003): {', '.join(sel)}")
    tee(f"  -> slim current model: meanCV={full_cv:.4f} meanTest={full_mt:.4f} worstTest={full_wt:.4f}")

    json.dump(dict(selected=sel, meanCV=full_cv, meanTest=full_mt, worstTest=full_wt,
                   lpred_alone=dict(meanCV=cv1, meanTest=mt1), path=path,
                   baseline21_meanTest=0.106),
              open("cur_selected.json", "w"), indent=2)
    tee("\nwrote cur_selected.json")
    OUT.close()


if __name__ == "__main__":
    main()
