"""CURRENT channel probe -- understand the signal before searching.

  1. raw cur stats per joint: sign/range (confirm it is an UNSIGNED magnitude)
  2. correlation of cur with |load| and with a state |torque| proxy
  3. reproduce the existing 21-feature current model R2 (regime-CV + 6 held-out),
     under the SAME harness/protocol as the load work -> the ~0.05-0.18 baseline

NO fabricated numbers.  Writes results/cur_probe.txt
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

OUT = open("results/cur_probe.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def feat_cur(ch, j, P=DEFAULTS):
    """EXACT reproduction of production x0_features.feat_cur (21 magnitude cols)."""
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


def main():
    tee("=" * 78)
    tee("CURRENT CHANNEL PROBE  (same data/protocol as load: pool run_1..5, 6 held-out)")
    tee("=" * 78)

    # pool (default constants = what feat_cur uses) + step idx
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", DEFAULTS, keep_step=True)
        chs.append(ch); steps.append(st)
    tests = {t: channels(f"{EXC}/{t}/excitation_recording.csv", DEFAULTS) for t in TESTS}
    # load-model channels (different smoothing) for |load_pred| cross-feature later
    chs_L = [channels(f"{EXC}/run_{i}/excitation_recording.csv", P_LOAD) for i in range(1, 6)]

    # ---- 1. raw current stats: unsigned? ----
    tee("\n--- 1. raw current stats (pooled run_1..5) ---")
    tee(f"  {'joint':<7}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}{'frac<0':>9}")
    for j in GJ:
        cu = np.concatenate([ch["cur"][j] for ch in chs])
        tee(f"  J{j:<6}{cu.min():>10.2f}{cu.max():>10.2f}{cu.mean():>10.2f}{cu.std():>10.2f}{np.mean(cu<0):>9.3f}")

    # ---- 2. correlation of cur with |load| and with state |torque| proxy ----
    tee("\n--- 2. corr(cur, |load|) and corr(cur, |state-torque proxy|) ---")
    tee("    (|load| is measured-channel; the proxy uses STATE only via the slim-10 load fit)")
    # fit slim-10 load model (state->load) per joint to get a state |torque| prediction
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    fL = {}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, st in zip(chs_L, steps):
            Xs.append(build_X(SEL, ch, j, P_LOAD)); ys.append(ch["load"][j])
            fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
        g = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)
        fL[j] = g
    tee(f"  {'joint':<7}{'corr(cur,|load|)':>18}{'corr(cur,|load_pred_state|)':>28}")
    for j in GJ:
        cu = np.concatenate([ch["cur"][j] for ch in chs])
        ld = np.concatenate([ch["load"][j] for ch in chs])
        # state-only predicted load magnitude
        g = fL[j]; b = g.fit_full(list(range(len(SEL))))
        preds = []
        for ch in chs_L:
            X = (build_X(SEL, ch, j, P_LOAD) - g.mu) / g.sd
            preds.append(g.ybar + X @ b)
        lp = np.abs(np.concatenate(preds))
        c1 = np.corrcoef(cu, np.abs(ld))[0, 1]
        c2 = np.corrcoef(cu, lp)[0, 1]
        tee(f"  J{j:<6}{c1:>18.3f}{c2:>28.3f}")

    # ---- 3. reproduce 21-feature current model R2 ----
    tee("\n--- 3. existing 21-feature current model (regime-CV + 6 held-out R2) ---")
    tee(f"  {'joint':<7}{'cvR2':>9}{'worstN':>9}{'meanTest':>10}{'worstTest':>11}")
    cvs, mts = [], []
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, st in zip(chs, steps):
            Xs.append(feat_cur(ch, j)); ys.append(ch["cur"][j])
            fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
        g = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)
        idx = list(range(Xs[0].shape[1]))
        r2, w = g.cv_score(idx); b = g.fit_full(idx)
        per = []
        for t in TESTS:
            Xt = (feat_cur(tests[t], j) - g.mu) / g.sd
            yh = g.ybar + Xt @ b; y = tests[t]["cur"][j]
            per.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
        per = np.array(per)
        tee(f"  J{j:<6}{r2:>9.3f}{w:>9.3f}{per.mean():>10.3f}{per.min():>11.3f}")
        cvs.append(r2); mts.append(per.mean())
    tee(f"  {'MEAN':<7}{np.mean(cvs):>9.3f}{'':>9}{np.mean(mts):>10.3f}")
    tee("\n  (this reproduces the previously-reported low current R2 baseline)")
    OUT.close()


if __name__ == "__main__":
    main()
