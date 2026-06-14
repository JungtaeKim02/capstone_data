"""TASK 3 -- LOAD vs CURRENT characteristics + fusion, from data.

Both residuals (measured - state_model) are external-force proxies. Questions:
  A. Per operating region (|accel|, |vel|), where is each channel strong/weak?
  B. Are the two residuals REDUNDANT (correlated) or COMPLEMENTARY (independent)?
  C. Payload-detection SNR per channel (signal = residual shift under added mass,
     noise = residual shift between two no-load repeats of the SAME trajectory).
  D. The doc claim "high accel -> load weak, current strong": true?  If yes, tune a
     gate; if no, say so.
  E. FUSION: does combining beat the best single channel for force detection?
     If not, report fusion is unnecessary (honest).

NO fabricated numbers.  Writes results/fusion.txt
"""
import json
import numpy as np
import pandas as pd
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X, lag, rms
from search import GramCV, K_FOLDS

SEL = json.load(open("selected5.json"))["selected"]
C = json.load(open("constants6.json"))
P_LOAD = {**DEFAULTS, "sg_v": C["sg_v"], "sg_a": C["sg_a"], "sg_as": C["sg_as"],
          "sg_vp": C["sg_vp"], "lag_a": C["lag_a"], "fc_slow": C["fc_slow"]}
P_CUR = {**DEFAULTS, "sg_v": 61}      # Stage-3 current-model smoothing
SELC = ["absLpred", "absvL", "Lpred2", "vL2", "absG_c4", "absG_c3"]
TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]
ALPHA_L, ALPHA_C = 10.0, 100.0
W30 = int(30.0 / F.DT)

OUT = open("results/fusion.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def curcols(ch, j):
    q = ch["q"]
    cc = {"absLpred": np.abs(ch["Lpred"][j]), "Lpred2": ch["Lpred"][j] ** 2,
          "absvL": np.abs(ch["vL"][j]), "vL2": ch["vL"][j] ** 2,
          "absG_c4": np.abs(np.cos(q[2] + q[3] + q[4])), "absG_c3": np.abs(np.cos(q[2] + q[3]))}
    return np.column_stack([cc[n] for n in SELC])


# ---- fit both models on pool run_1..5 ----
def fit_models():
    chsL, steps = [], []
    chsC = []
    for i in range(1, 6):
        chL, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_LOAD, keep_step=True)
        chC = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_CUR)
        chsL.append(chL); steps.append(st); chsC.append(chC)
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    folds = [np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1) for st in steps]
    gL, gC = {}, {}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for chL, fo in zip(chsL, folds):
            Xs.append(build_X(SEL, chL, j, P_LOAD)); ys.append(chL["load"][j]); fs.append(fo)
        gL[j] = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA_L)
    # inject Lpred into pool cur channels, fit current model
    for chC, chL in zip(chsC, chsL):
        inject(chC, chL, gL)
    for j in GJ:
        Xs, ys, fs = [], [], []
        for chC, fo in zip(chsC, folds):
            Xs.append(curcols(chC, j)); ys.append(chC["cur"][j]); fs.append(fo)
        gC[j] = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA_C)
    return gL, gC


def inject(chC, chL, gL):
    chC["Lpred"] = {}
    for j in GJ:
        g = gL[j]; b = g.fit_full(list(range(len(SEL))))
        X = (build_X(SEL, chL, j, P_LOAD) - g.mu) / g.sd
        chC["Lpred"][j] = g.ybar + X @ b


def load_rec(path, gL):
    """return ch_load, ch_cur(with Lpred), step."""
    chL, st = channels(path, P_LOAD, keep_step=True)
    chC = channels(path, P_CUR)
    inject(chC, chL, gL)
    return chL, chC, st


def resid_L(gL, chL, j):
    g = gL[j]; b = g.fit_full(list(range(len(SEL))))
    X = (build_X(SEL, chL, j, P_LOAD) - g.mu) / g.sd
    return chL["load"][j] - (g.ybar + X @ b)


def resid_C(gC, chC, j):
    g = gC[j]; b = g.fit_full(list(range(len(SELC))))
    X = (curcols(chC, j) - g.mu) / g.sd
    return chC["cur"][j] - (g.ybar + X @ b)


def main():
    gL, gC = fit_models()
    tee("=" * 82)
    tee("TASK 3  LOAD vs CURRENT  characteristics + fusion (all from CSV)")
    tee("=" * 82)

    # ---------- A. per-region model error (where each channel is strong/weak) ----------
    tee("\n=== A. per-region residual NRMSE (pooled 6 held-out tests) ===")
    tee("  NRMSE = std(residual)/std(channel); LOWER = channel better predicted there")
    recs = {t: load_rec(f"{EXC}/{t}/excitation_recording.csv", gL) for t in TESTS}
    for axis in ("aL", "vL"):
        tee(f"\n  -- binned by |{axis}| terciles --")
        tee(f"  {'joint':<7}{'|'+axis+'| low':>22}{'mid':>16}{'high':>16}")
        tee(f"  {'':<7}{'load / cur':>22}{'load / cur':>16}{'load / cur':>16}")
        for j in GJ:
            mag = np.concatenate([np.abs(recs[t][1][axis][j]) for t in TESTS])
            rl = np.concatenate([resid_L(gL, recs[t][0], j) for t in TESTS])
            rc = np.concatenate([resid_C(gC, recs[t][1], j) for t in TESTS])
            yl = np.concatenate([recs[t][0]["load"][j] for t in TESTS])
            yc = np.concatenate([recs[t][1]["cur"][j] for t in TESTS])
            q1, q2 = np.quantile(mag, [1/3, 2/3])
            cells = []
            for lo, hi in ((-1, q1), (q1, q2), (q2, 1e9)):
                m = (mag >= lo) & (mag < hi)
                nl = rl[m].std() / yl.std(); nc = rc[m].std() / yc.std()
                cells.append(f"{nl:.3f} / {nc:.3f}")
            tee(f"  J{j:<6}{cells[0]:>22}{cells[1]:>16}{cells[2]:>16}")

    # ---------- B. residual correlation: redundant vs complementary ----------
    tee("\n=== B. corr(resid_load, resid_cur) on held-out tests (high=redundant) ===")
    tee(f"  {'joint':<7}{'corr':>9}")
    for j in GJ:
        rl = np.concatenate([resid_L(gL, recs[t][0], j) for t in TESTS])
        rc = np.concatenate([resid_C(gC, recs[t][1], j) for t in TESTS])
        tee(f"  J{j:<6}{np.corrcoef(rl, rc)[0,1]:>9.3f}")

    # ---------- C/E. payload-detection SNR + fusion ----------
    tee("\n=== C+E. payload detection: SNR per channel + fused (run_1 ref, run_2 noise) ===")
    ref = load_rec(f"{EXC}/run_1/excitation_recording.csv", gL)
    noi = load_rec(f"{EXC}/run_2/excitation_recording.csv", gL)
    pays = {m: load_rec(f"{EXC}/payload_{m}g/run_1/excitation_recording.csv", gL)
            for m in (84, 146, 227)}

    def series(rec, j):
        chL, chC, st = rec
        return pd.DataFrame({"rL": resid_L(gL, chL, j), "rC": resid_C(gC, chC, j)}, index=st)

    tee(f"  {'joint':<6}{'mass':>6}{'SNR_load':>10}{'SNR_cur':>10}{'SNR_fused':>11}{'best_single':>12}{'fusion?':>9}")
    fusion_helps = {j: [] for j in GJ}
    for j in GJ:
        s_ref = series(ref, j); s_noi = series(noi, j)
        # noise floor: matched run1 vs run2
        dn = pd.concat([s_ref.add_suffix("_a"), s_noi.add_suffix("_b")], axis=1, join="inner").dropna().iloc[W30:]
        nL = rms((dn["rL_b"] - dn["rL_a"]).to_numpy()); nC = rms((dn["rC_b"] - dn["rC_a"]).to_numpy())
        zfn = (dn["rL_b"] - dn["rL_a"]).to_numpy() / nL + (dn["rC_b"] - dn["rC_a"]).to_numpy() / nC
        nF = rms(zfn)
        for m in (84, 146, 227):
            s_p = series(pays[m], j)
            d = pd.concat([s_ref.add_suffix("_a"), s_p.add_suffix("_b")], axis=1, join="inner").dropna().iloc[W30:]
            dL = (d["rL_b"] - d["rL_a"]).to_numpy(); dC = (d["rC_b"] - d["rC_a"]).to_numpy()
            snrL = rms(dL) / nL; snrC = rms(dC) / nC
            zf = dL / nL + dC / nC
            snrF = rms(zf) / nF
            best = max(snrL, snrC)
            helps = snrF > best * 1.02
            fusion_helps[j].append(snrF - best)
            tee(f"  J{j:<5}{m:>5}g{snrL:>10.2f}{snrC:>10.2f}{snrF:>11.2f}{best:>12.2f}{('YES' if helps else 'no'):>9}")

    # ---------- D. accel-gating claim ----------
    tee("\n=== D. claim 'high accel -> load weak, current strong' (per-region NRMSE, accel) ===")
    tee("  re-using A: compare load NRMSE low->high |aL| vs current NRMSE low->high.")
    tee("  (if load NRMSE rises with |aL| while cur NRMSE falls -> gating justified)")
    for j in GJ:
        mag = np.concatenate([np.abs(recs[t][1]["aL"][j]) for t in TESTS])
        rl = np.concatenate([resid_L(gL, recs[t][0], j) for t in TESTS])
        rc = np.concatenate([resid_C(gC, recs[t][1], j) for t in TESTS])
        yl = np.concatenate([recs[t][0]["load"][j] for t in TESTS])
        yc = np.concatenate([recs[t][1]["cur"][j] for t in TESTS])
        q1, q2 = np.quantile(mag, [1/3, 2/3])
        def nr(arr, y, lo, hi):
            m = (mag >= lo) & (mag < hi); return arr[m].std() / y.std()
        lL = [nr(rl, yl, *b) for b in ((-1, q1), (q2, 1e9))]
        lC = [nr(rc, yc, *b) for b in ((-1, q1), (q2, 1e9))]
        trL = "rises" if lL[1] > lL[0] else "falls"
        trC = "rises" if lC[1] > lC[0] else "falls"
        tee(f"  J{j}: load NRMSE low->high |aL| {lL[0]:.3f}->{lL[1]:.3f} ({trL}); "
            f"cur {lC[0]:.3f}->{lC[1]:.3f} ({trC})")

    tee("\n=== VERDICT ===")
    anyhelp = any(np.mean(v) > 0.0 for v in fusion_helps.values())
    tee(f"  mean fused-minus-best SNR per joint: " +
        ", ".join(f"J{j}={np.mean(fusion_helps[j]):+.2f}" for j in GJ))
    OUT.close()


if __name__ == "__main__":
    main()
