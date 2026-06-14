"""STAGE 2b -- the most direct possible test of "does ANY feature beyond the
slim-10 improve GENERALISATION".  Add each candidate (one at a time) to the FULL
slim-10, refit, and measure the actual mean/worst held-out R2 over the 6 test
trajectories AND the regime-CV.  No greedy path-dependence: every candidate is
judged against the same complete 10-feature baseline.

Candidates include families that live in NO library yet (jerk, sign-dependent
inertia, velocity*accel power, J2 cogging harmonic) so the search is genuinely
open, not constrained to the x0 vocabulary.

NO fabricated numbers.  Writes results/stage2b_augment.txt
"""
import json
import numpy as np
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X, lag
from search import GramCV, K_FOLDS

P = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
ALPHA = 10.0
SEL10 = json.load(open("selected5.json"))["selected"]
TESTS = ["test_noload", "test_noload_2", "test_noload_3",
         "test_noload_4", "test_noload_5", "test_noload_p2p"]

OUT = open("results/stage2b_augment.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")

# candidate add-on features (fn -> column or 2-col block), beyond the slim-10
def _c3(ch): return ch["q"][2] + ch["q"][3]
def _c4(ch): return ch["q"][2] + ch["q"][3] + ch["q"][4]
CAND = {
    "aL2g_cc3 (Xinertia M.2*cosc3)": lambda ch, j: ch["aL"][2] * np.cos(_c3(ch)),
    "aL2g     (Xinertia M.2)":       lambda ch, j: ch["aL"][2],
    "aL3g_cc3 (Xinertia M.3*cosc3)": lambda ch, j: ch["aL"][3] * np.cos(_c3(ch)),
    "strib    (Stribeck fc.1)":      lambda ch, j: ch["vL"][j] * np.exp(-(ch["vL"][j] / 0.1) ** 2),
    "coul_cos_c4 (load-Coulomb)":    lambda ch, j: np.tanh(ch["vL"][j] / 0.02) * np.cos(_c4(ch)),
    "coul_slow (plain Coulomb)":     lambda ch, j: np.tanh(ch["vL"][j] / 0.02),
    "lag_dir  (backlash)":           lambda ch, j: lag(np.tanh(ch["vL"][j] / 0.02), 25),
    "jerk     (d aL/dt)":            lambda ch, j: np.gradient(ch["aL"][j], F.DT),
    "relu_aL  (accel+ inertia)":     lambda ch, j: np.maximum(ch["aL"][j], 0.0),
    "vL_aL    (vel*accel power)":    lambda ch, j: ch["vL"][j] * ch["aL"][j],
    "cog_q_w7.5 (J-cogging pair)":   lambda ch, j: np.column_stack([np.sin(7.5 * ch["q"][j]), np.cos(7.5 * ch["q"][j])]),
    "vS       (short-band vel)":     lambda ch, j: ch["vS"][j],
}


def cols(fn, ch, j):
    a = fn(ch, j)
    return a if a.ndim == 2 else a[:, None]


def main():
    tee("=" * 80)
    tee("STAGE 2b  AUGMENT slim-10 BY ONE FEATURE  (6 held-out trajectories, P_OPT)")
    tee(f"  baseline slim-10: {', '.join(SEL10)}")
    tee("=" * 80)

    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P, keep_step=True)
        chs.append(ch); steps.append(st)
    tests = {t: channels(f"{EXC}/{t}/excitation_recording.csv", P) for t in TESTS}

    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    folds = {ch_i: np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1)
             for ch_i, st in zip(range(5), steps)}

    def build_design(extra_fn):
        """per joint: pooled (X, y, fold) for slim-10 (+ optional extra block)."""
        out = {}
        for j in GJ:
            Xs, ys, fs = [], [], []
            for i, ch in enumerate(chs):
                base = build_X(SEL10, ch, j, P)
                if extra_fn is not None:
                    base = np.hstack([base, cols(extra_fn, ch, j)])
                Xs.append(base); ys.append(ch["load"][j]); fs.append(folds[i])
            out[j] = (np.vstack(Xs), np.concatenate(ys), np.concatenate(fs))
        return out

    def eval_set(extra_fn):
        des = build_design(extra_fn)
        grams = {j: GramCV(*des[j], ALPHA) for j in GJ}
        idx = list(range(des[GJ[0]][0].shape[1]))
        cv = float(np.mean([grams[j].cv_score(idx)[0] for j in GJ]))
        bj = {j: grams[j].fit_full(idx) for j in GJ}
        per = []
        for t in TESTS:
            r2s = []
            for j in GJ:
                Xt = build_X(SEL10, tests[t], j, P)
                if extra_fn is not None:
                    Xt = np.hstack([Xt, cols(extra_fn, tests[t], j)])
                mu, sd = grams[j].mu[idx], grams[j].sd[idx]
                yh = grams[j].ybar + ((Xt - mu) / sd) @ bj[j]
                y = tests[t]["load"][j]
                r2s.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
            per.append(np.mean(r2s))
        per = np.array(per)
        return cv, float(per.mean()), float(per.min())

    base_cv, base_mt, base_wt = eval_set(None)
    tee(f"\n  {'candidate added':<32}{'meanCV':>8}{'dCV':>8}{'meanTest':>9}{'dTest':>8}{'worstTest':>10}")
    tee(f"  {'(slim-10 baseline)':<32}{base_cv:>8.4f}{'':>8}{base_mt:>9.4f}{'':>8}{base_wt:>10.4f}")
    rows = []
    for nm, fn in CAND.items():
        cv, mt, wt = eval_set(fn)
        rows.append((mt - base_mt, nm, cv, cv - base_cv, mt, wt))
    for dmt, nm, cv, dcv, mt, wt in sorted(rows, reverse=True):
        tee(f"  {nm:<32}{cv:>8.4f}{dcv:>+8.4f}{mt:>9.4f}{dmt:>+8.4f}{wt:>10.4f}")

    tee("\n  READ-OUT: dTest = change in mean 6-traj held-out R2 vs slim-10.")
    tee("  A feature only earns inclusion if dTest is clearly positive AND not")
    tee("  costing worstTest.  Otherwise the slim-10 is at the generalisation ceiling.")
    OUT.close()


if __name__ == "__main__":
    main()
