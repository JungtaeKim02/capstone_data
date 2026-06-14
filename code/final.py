"""Finalize: re-run unified selection UNDER the CV-optimal constants (closes the
loop: better channels could in principle change the picks), confirm the tuned
10-feature model beats both baselines, and re-verify payload sensitivity.
Writes final_model.json (features + constants + metrics).
"""
import json
import numpy as np
import pandas as pd
import featlib as F
from featlib import LIB, EXC, GJ, channels, build_X, rms, DEFAULTS
from search import design_pool, GramCV

P_OPT = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
ALPHA = 10.0
CANDS = [n for n in LIB if n != "bias"]
SEL10 = json.load(open("selected_features.json"))["selected"]


def load_pool_P(P):
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def main():
    print(f"CV-optimal constants: sg_va=61, sg_vp=161, lag_a=25, fc_slow=0.02, alpha=10")
    # build engines under BOTH default and optimal constants for an honest 2x2
    G = {}
    chB = {}
    for tag, P in (("def", DEFAULTS), ("opt", P_OPT)):
        chs, steps = load_pool_P(P)
        chB[tag] = channels(f"{EXC}/test_noload/excitation_recording.csv", P)
        G[tag] = {}
        for j in GJ:
            X, y, folds = design_pool(chs, steps, j, CANDS, P)
            G[tag][j] = GramCV(X, y, folds, ALPHA)
    grams = G["opt"]          # selection is done under optimal constants
    P_sel = P_OPT

    def mean_cv(names, tag="opt"):
        idx = [CANDS.index(n) for n in names]
        rs = [G[tag][j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(max(w for _, w in rs))

    def mean_B(names, tag="opt"):
        idx = [CANDS.index(n) for n in names]
        P = P_OPT if tag == "opt" else DEFAULTS
        out = []
        for j in GJ:
            b = G[tag][j].fit_full(idx)
            Xt = (build_X(names, chB[tag], j, P) - G[tag][j].mu[idx]) / G[tag][j].sd[idx]
            yh = G[tag][j].ybar + Xt @ b
            y = chB[tag]["load"][j]
            out.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
        return float(np.mean(out))

    # re-run greedy selection under P_OPT -> confirm the core is stable
    print("\nRe-selection under optimal constants (first 12 greedy picks):")
    chosen, best = [], -1e9
    for _ in range(12):
        scored = []
        for c in range(len(CANDS)):
            if c in chosen:
                continue
            r2, _ = mean_cv([CANDS[i] for i in chosen] + [CANDS[c]])
            scored.append((r2, c))
        scored.sort(reverse=True)
        r2, c = scored[0]
        chosen.append(c)
        print(f"  {len(chosen):>2} {CANDS[c]:<26}{LIB[CANDS[c]]['group']:<12}meanCV={r2:.4f} (+{r2-best:.4f})")
        best = r2
    reselected = [CANDS[i] for i in chosen]

    # honest 2x2: {x0-31, slim-10} x {default consts, optimal consts}
    x0_31 = [n for n in F.X0_LOAD_32 if n != "bias"]
    print("\n=== model comparison (held-out Traj-B is the key column) ===")
    print(f"  {'model / constants':<32}{'nfeat':>6}{'meanCV_R2':>11}{'worstNRMSE':>12}{'meanB_R2':>10}")
    for label, names, tag in (
        ("x0-31  @ default  (current)", x0_31, "def"),
        ("slim-10 @ default", SEL10, "def"),
        ("x0-31  @ optimal", x0_31, "opt"),
        ("slim-10 @ optimal  (FINAL)", SEL10, "opt"),
        ("re-selected-10 @ optimal", reselected[:10], "opt"),
    ):
        cv, w = mean_cv(names, tag)
        B = mean_B(names, tag)
        print(f"  {label:<32}{len(names):>6}{cv:>11.4f}{w:>12.3f}{B:>10.4f}")

    # payload sensitivity of the final slim-10 under optimal constants
    print("\nPAYLOAD SENSITIVITY (slim-10, optimal constants)")
    refA, refS = channels(f"{EXC}/run_1/excitation_recording.csv", P_OPT, keep_step=True)
    pays = {m: channels(f"{EXC}/payload_{m}g/run_1/excitation_recording.csv", P_OPT, keep_step=True)
            for m in (84, 146, 227)}
    W30 = int(30.0 / F.DT)
    idx = [CANDS.index(n) for n in SEL10]

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

    cv, w = mean_cv(SEL10); B = mean_B(SEL10)
    json.dump(dict(features=SEL10, reselected_under_opt=reselected,
                   constants=dict(sg_v=61, sg_a=61, sg_vp=161, lag_a=25, lag_b=150,
                                  fc_slow=0.02, fc_fast=0.2, fc_strib=0.1, alpha=ALPHA),
                   meanCV_R2=cv, worstNRMSE=w, meanB_R2=B),
              open("final_model.json", "w"), indent=2)
    print("\nwrote final_model.json")


if __name__ == "__main__":
    main()
