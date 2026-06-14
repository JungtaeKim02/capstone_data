"""STAGE 4 -- FINALISE the slim-10 under the fine-optimised constants, on all 6
held-out trajectories.  Pulls the converged constants from constants6.json so the
numbers are never hand-set.

  1. honest 2x2 : {x0-31, slim-10} x {default, final} -- 6-traj test R2
  2. per-trajectory generalisation of the FINAL model (must hold on all 6)
  3. payload sensitivity (residual must still grow with mass) under final P
  4. per-feature justification: forward-gain (experimental, from selection path) +
     drop-one dR2 (experimental, refit-without) + EOM/mechanism + hardware-class
  5. write final_model6.json

NO fabricated numbers.  Writes results/final6.txt + final_model6.json
"""
import json
import numpy as np
import pandas as pd
import featlib as F
from featlib import LIB, EXC, GJ, channels, build_X, rms, DEFAULTS
from search import GramCV, K_FOLDS

SEL = json.load(open("selected5.json"))["selected"]
PATH = {p["feat"]: p for p in json.load(open("selected5.json"))["path"]}
C = json.load(open("constants6.json"))
P_FIN = {**DEFAULTS, "sg_v": C["sg_v"], "sg_a": C["sg_a"], "sg_as": C["sg_as"],
         "sg_vp": C["sg_vp"], "lag_a": C["lag_a"], "fc_slow": C["fc_slow"]}
ALPHA = 10.0          # CV dead-flat over alpha 3..40 -> keep production 10 (robust)
C["alpha"] = ALPHA
CANDS = [n for n in LIB if n != "bias"]
TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]

# physical / hardware-class justification (mechanism that generalises to any
# low-cost, high-gear-ratio geared servo -- NOT STS3215-specific)
JUSTIFY = {
    "vL":          ("EOM friction  b*qdot", "viscous drag (lubricant + brush/bearing) ~ speed", "physical"),
    "cos_c3":      ("EOM gravity  g(q)", "weight of links beyond elbow ~ cos(cumulative angle)", "physical"),
    "tanh_vP":     ("Coulomb sign on low-pass vel", "gear-train lag/windup -> friction sign follows SMOOTHED velocity", "physical+exp"),
    "aL":          ("EOM inertia  M_ii*qddot", "reflected inertia N^2*Jmotor dominates at high gear ratio", "physical"),
    "sin_q2":      ("EOM gravity  g(q)", "orthogonal gravity phase on the shoulder link", "physical"),
    "lag_vL_a":    ("transport delay of vL", "fixed group delay through compliant high-ratio gear train", "physical+exp"),
    "lag_aL_a":    ("transport delay of aL", "same drivetrain delay on the inertia channel", "physical+exp"),
    "cos_c4":      ("EOM gravity  g(q)", "weight of wrist link ~ cos(cumulative angle to wrist)", "physical"),
    "coul_sin_c3": ("load-dependent Coulomb", "gearbox friction ~ normal/tooth load, which tracks gravity torque", "physical"),
    "adiff":       ("band-pass accel aS-aL", "fast drivetrain-compliance mode above the smoothed-inertia band", "experimental"),
}

OUT = open("results/final6.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def load_pool(P):
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def build_grams(chs, steps, P, names):
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    G = {}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, st in zip(chs, steps):
            Xs.append(build_X(names, ch, j, P)); ys.append(ch["load"][j])
            fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
        G[j] = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)
    return G


def main():
    tee("=" * 80)
    tee("STAGE 4  FINAL MODEL  (slim-10 @ fine-optimised constants, 6 held-out)")
    tee(f"  constants: sg_v={C['sg_v']} sg_a={C['sg_a']} sg_as={C['sg_as']} "
        f"sg_vp={C['sg_vp']} lag_a={C['lag_a']} fc_slow={C['fc_slow']} alpha={ALPHA}")
    tee("=" * 80)

    # build pool + tests under default and final constants
    data = {}
    for tag, P in (("def", DEFAULTS), ("fin", P_FIN)):
        chs, steps = load_pool(P)
        tests = {t: channels(f"{EXC}/{t}/excitation_recording.csv", P) for t in TESTS}
        G = build_grams(chs, steps, P, CANDS)
        Xt = {t: {j: build_X(CANDS, tests[t], j, P) for j in GJ} for t in TESTS}
        data[tag] = (G, tests, Xt)

    def mean_cv(names, tag):
        G = data[tag][0]; idx = [CANDS.index(n) for n in names]
        rs = [G[j].cv_score(idx) for j in GJ]
        return float(np.mean([r for r, _ in rs])), float(max(w for _, w in rs))

    def per_traj(names, tag):
        G, tests, Xt = data[tag]; idx = [CANDS.index(n) for n in names]
        bj = {j: G[j].fit_full(idx) for j in GJ}; out = {}
        for t in TESTS:
            r2s = []
            for j in GJ:
                mu, sd = G[j].mu[idx], G[j].sd[idx]
                yh = G[j].ybar + ((Xt[t][j][:, idx] - mu) / sd) @ bj[j]
                y = tests[t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            out[t] = float(np.mean(r2s))
        return out

    def mean_test(names, tag):
        d = per_traj(names, tag); v = np.array(list(d.values()))
        return float(v.mean()), float(v.min())

    # --- 1. honest 2x2 ---
    x0_31 = [n for n in F.X0_LOAD_32 if n != "bias"]
    tee("\n=== model x constants (mean over 6 held-out trajectories) ===")
    tee(f"  {'model / constants':<28}{'nfeat':>6}{'meanCV':>8}{'worstN':>8}{'meanTest':>9}{'worstTest':>10}")
    for label, names, tag in (("x0-31  @ default (current)", x0_31, "def"),
                              ("slim-10 @ default", SEL, "def"),
                              ("x0-31  @ final", x0_31, "fin"),
                              ("slim-10 @ final (FINAL)", SEL, "fin")):
        cv, w = mean_cv(names, tag); mt, wt = mean_test(names, tag)
        tee(f"  {label:<28}{len(names):>6}{cv:>8.4f}{w:>8.3f}{mt:>9.4f}{wt:>10.4f}")

    # --- 2. per-trajectory ---
    per = per_traj(SEL, "fin")
    tee("\n=== FINAL slim-10 @ final constants: per-trajectory held-out R2 ===")
    for t in TESTS:
        kind = "p2p (DIFFERENT TYPE)" if "p2p" in t else "excitation (diff seed)"
        tee(f"  {t:<18}{per[t]:>8.4f}   {kind}")
    tee(f"  {'MEAN':<18}{np.mean(list(per.values())):>8.4f}   worst={min(per.values()):.4f}")

    # --- 3. payload sensitivity ---
    tee("\nPAYLOAD SENSITIVITY (slim-10 @ final): dr(m) must grow with mass")
    G = data["fin"][0]; idx = [CANDS.index(n) for n in SEL]
    refA, refS = channels(f"{EXC}/run_1/excitation_recording.csv", P_FIN, keep_step=True)
    pays = {m: channels(f"{EXC}/payload_{m}g/run_1/excitation_recording.csv", P_FIN, keep_step=True)
            for m in (84, 146, 227)}
    W30 = int(30.0 / F.DT)

    def resid(ch, j):
        b = G[j].fit_full(idx)
        X = (build_X(SEL, ch, j, P_FIN) - G[j].mu[idx]) / G[j].sd[idx]
        return ch["load"][j] - (G[j].ybar + X @ b)

    tee(f"  {'joint':<6}{'dr(84g)':>9}{'dr(146g)':>10}{'dr(227g)':>10}{'monotone?':>11}")
    mono_all = True
    for j in GJ:
        rr = pd.Series(resid(refA, j), index=refS); d = []
        for m in (84, 146, 227):
            ch_p, st_p = pays[m]; rp = pd.Series(resid(ch_p, j), index=st_p)
            df = pd.concat([rr.rename("r"), rp.rename("p")], axis=1, join="inner").dropna().iloc[W30:]
            d.append(rms((df["p"] - df["r"]).to_numpy()))
        mono = d[0] < d[1] < d[2]; mono_all &= mono
        tee(f"  J{j:<5}{d[0]:>9.2f}{d[1]:>10.2f}{d[2]:>10.2f}{('yes' if mono else 'NO'):>11}")

    # --- 4. per-feature justification ---
    full = mean_test(SEL, "fin")[0]
    tee("\n=== per-feature justification (slim-10 @ final) ===")
    tee(f"  {'feature':<13}{'group':<10}{'fwd-gain':>9}{'drop-dR2':>9}  EOM/mechanism")
    rows = []
    for n in SEL:
        without = mean_test([m for m in SEL if m != n], "fin")[0]
        drop = full - without
        fg = PATH[n]["gain"] if (n in PATH and PATH[n]["gain"] < 1) else float("nan")
        rows.append((drop, n, fg))
    for drop, n, fg in sorted(rows, reverse=True):
        eom, mech, kind = JUSTIFY[n]
        tee(f"  {n:<13}{LIB[n]['group']:<10}{fg:>9.4f}{drop:>9.4f}  {eom}  [{kind}]")
        tee(f"  {'':<32} -> {mech}")

    cv, w = mean_cv(SEL, "fin"); mt, wt = mean_test(SEL, "fin")
    json.dump(dict(features=SEL, constants=C, meanCV=cv, worstNRMSE=w,
                   meanTest=mt, worstTest=wt, per_traj=per,
                   payload_monotone=bool(mono_all), tests=TESTS,
                   justification={n: dict(eom=JUSTIFY[n][0], mechanism=JUSTIFY[n][1],
                                          kind=JUSTIFY[n][2]) for n in SEL}),
              open("final_model6.json", "w"), indent=2)
    tee(f"\n  payload all-monotone: {mono_all}")
    tee("wrote final_model6.json")
    OUT.close()


if __name__ == "__main__":
    main()
