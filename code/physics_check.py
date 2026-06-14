"""PHYSICS-FIRST verification of the x0 LOAD slim-10.

Order is fixed and auditable:
  1. physics_prior.json was FROZEN first (chmod 444, sha256 recorded) -- it holds
     the EOM derivation and the falsifiable predictions, written before any
     coefficient was looked at.  THIS script only READS it.
  2. Fit slim-10 @ final constants on the train pool (run_1..5), per joint.
  3. Extract per-joint coefficients in RAW (un-standardised) space and in a
     scale-free per-joint contribution (|b_std|/sigma_y).
  4. Check each pre-registered prediction (sign / zero-structure / ordering).
  5. Classify every feature into one of three honest grades.

Because load is UNCALIBRATED we never compare absolute N*m: only sign, the
zero/non-zero structure across joints, and relative ordering -- exactly the two
calibration-free comparisons the brief asks for.

NO fabricated numbers.  Writes results/physics_check.txt
"""
import json
import hashlib
import numpy as np
import featlib as F
from featlib import EXC, GJ, DEFAULTS, channels, build_X
from search import GramCV, K_FOLDS

PRIOR_PATH = "physics_prior.json"
SEL = json.load(open("selected5.json"))["selected"]
C = json.load(open("constants6.json"))
P_FIN = {**DEFAULTS, "sg_v": C["sg_v"], "sg_a": C["sg_a"], "sg_as": C["sg_as"],
         "sg_vp": C["sg_vp"], "lag_a": C["lag_a"], "fc_slow": C["fc_slow"]}
ALPHA = 10.0   # production ridge (CV dead-flat 3..40); same value used in final6

OUT = open("results/physics_check.txt", "w")
def tee(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")


def fit_per_joint():
    """Per joint: raw + standardised slim-10 coefficients, feature stds, sigma_y."""
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", P_FIN, keep_step=True)
        chs.append(ch); steps.append(st)
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1)); edges[-1] += 1
    res = {}
    for j in GJ:
        Xs, ys, fs = [], [], []
        for ch, st in zip(chs, steps):
            Xs.append(build_X(SEL, ch, j, P_FIN)); ys.append(ch["load"][j])
            fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
        g = GramCV(np.vstack(Xs), np.concatenate(ys), np.concatenate(fs), ALPHA)
        idx = list(range(len(SEL)))
        b_std = g.fit_full(idx)
        b_raw = b_std / g.sd                 # coefficient on the un-standardised column
        contrib = np.abs(b_std) / g.sigma_y  # scale-free per-joint importance (frac of load std per 1 sigma)
        res[j] = dict(b_std=b_std, b_raw=b_raw, contrib=contrib, sd=g.sd,
                      sigma_y=float(g.sigma_y))
    return res


def col(res, feat, key):
    i = SEL.index(feat)
    return {j: float(res[j][key][i]) for j in GJ}


def sgn(x, tol):
    return "+" if x > tol else ("-" if x < -tol else "0")


def main():
    h = hashlib.sha256(open(PRIOR_PATH, "rb").read()).hexdigest()
    prior = json.load(open(PRIOR_PATH))
    tee("=" * 84)
    tee("PHYSICS-FIRST CHECK  (slim-10 @ final constants, train pool run_1..5)")
    tee(f"  prior frozen_at = {prior['_meta']['frozen_at_utc']}")
    tee(f"  prior sha256    = {h}")
    tee(f"  constants: sg_v={C['sg_v']} sg_a={C['sg_a']} sg_as={C['sg_as']} "
        f"sg_vp={C['sg_vp']} lag_a={C['lag_a']} fc_slow={C['fc_slow']} alpha={ALPHA}")
    tee("=" * 84)

    res = fit_per_joint()

    # contribution threshold for "present vs negligible": a feature is NEGLIGIBLE on
    # joint j if its scale-free contribution is < NEG_FRAC of its own MAX over joints.
    NEG_FRAC = 0.25

    # ---- raw + contribution table ----
    tee("\n--- per-joint coefficients (b_raw) and scale-free contribution (|b_std|/sigma_y) ---")
    tee(f"  {'feature':<13}" + "".join(f"{'J'+str(j)+' b_raw':>13}" for j in GJ)
        + "".join(f"{'J'+str(j)+' contr':>11}" for j in GJ))
    for n in SEL:
        br = col(res, n, "b_raw"); ct = col(res, n, "contrib")
        tee(f"  {n:<13}" + "".join(f"{br[j]:>13.4f}" for j in GJ)
            + "".join(f"{ct[j]:>11.4f}" for j in GJ))

    # gravity-column collinearity context (ridge can split correlated gravity terms)
    chs = channels(f"{EXC}/run_1/excitation_recording.csv", P_FIN)
    grav = ["sin_q2", "cos_c3", "cos_c4"]
    M = np.column_stack([build_X([g], chs, 2, P_FIN)[:, 0] for g in grav])
    Corr = np.corrcoef(M.T)
    tee("\n--- gravity-column correlation (run_1, J2 design) -- context for sign reads ---")
    tee("            " + "".join(f"{g:>9}" for g in grav))
    for a, g in enumerate(grav):
        tee(f"  {g:<10}" + "".join(f"{Corr[a, b]:>9.3f}" for b in range(len(grav))))

    # ---- prediction-by-prediction verdicts ----
    tee("\n" + "=" * 84)
    tee("PREDICTION CHECKS  (PASS / FAIL / n.a.)  -- each was frozen before this fit")
    tee("=" * 84)
    verdict = {}

    def contr(n):
        return col(res, n, "contrib")

    # sin_q2: present J2, negligible J3,J4 ; on-J2 order sin_q2>=cos_c3>=cos_c4
    c = contr("sin_q2"); cmax = max(abs(v) for v in c.values())
    neg34 = (abs(c[3]) < NEG_FRAC * cmax) and (abs(c[4]) < NEG_FRAC * cmax)
    pres2 = abs(c[2]) >= NEG_FRAC * cmax and c[2] == max(c.values(), key=abs)
    on_j2 = abs(contr("sin_q2")[2]) >= abs(contr("cos_c3")[2]) >= abs(contr("cos_c4")[2])
    tee(f"\nsin_q2  (gravity i=2):")
    tee(f"  present on J2, negligible on J3&J4 ?  contr J2={c[2]:.3f} J3={c[3]:.3f} J4={c[4]:.3f}"
        f"  -> {'PASS' if (pres2 and neg34) else 'FAIL'}")
    tee(f"  on-J2 gravity order sin_q2>=cos_c3>=cos_c4 ? "
        f"{contr('sin_q2')[2]:.3f}>={contr('cos_c3')[2]:.3f}>={contr('cos_c4')[2]:.3f}"
        f"  -> {'PASS' if on_j2 else 'FAIL'}")
    verdict["sin_q2"] = pres2 and neg34

    # cos_c3: present J2,J3 ; negligible J4 ; sign(J2)==sign(J3)
    c = contr("cos_c3"); b = col(res, "cos_c3", "b_raw"); cmax = max(abs(v) for v in c.values())
    tol = 0.02 * cmax  # sign tolerance in contrib units is awkward; use b_raw sign with small tol
    neg4 = abs(c[4]) < NEG_FRAC * cmax
    same23 = sgn(b[2], 1e-9) == sgn(b[3], 1e-9) and sgn(b[2], 1e-9) != "0"
    tee(f"\ncos_c3  (gravity i=3):")
    tee(f"  present J2&J3, negligible J4 ?  contr J2={c[2]:.3f} J3={c[3]:.3f} J4={c[4]:.3f}"
        f"  -> {'PASS' if (abs(c[2])>=NEG_FRAC*cmax and abs(c[3])>=NEG_FRAC*cmax and neg4) else 'FAIL'}")
    tee(f"  sign(J2)==sign(J3) ?  {sgn(b[2],1e-9)} {sgn(b[3],1e-9)}  -> {'PASS' if same23 else 'FAIL'}")
    verdict["cos_c3"] = same23 and neg4

    # cos_c4: present all ; sign equal across J2,J3,J4
    c = contr("cos_c4"); b = col(res, "cos_c4", "b_raw")
    s = {j: sgn(b[j], 1e-9) for j in GJ}
    same_all = (s[2] == s[3] == s[4]) and s[2] != "0"
    tee(f"\ncos_c4  (gravity i=4):")
    tee(f"  present on all ?  contr J2={c[2]:.3f} J3={c[3]:.3f} J4={c[4]:.3f}")
    tee(f"  sign equal across J2,J3,J4 ?  {s[2]} {s[3]} {s[4]}  -> {'PASS' if same_all else 'FAIL'}")
    verdict["cos_c4"] = same_all

    # aL: sign equal across joints ; |b_raw| order J2>=J3>=J4
    b = col(res, "aL", "b_raw"); s = {j: sgn(b[j], 1e-9) for j in GJ}
    same_all = (s[2] == s[3] == s[4]) and s[2] != "0"
    order = abs(b[2]) >= abs(b[3]) >= abs(b[4])
    tee(f"\naL  (diagonal inertia M_jj):")
    tee(f"  b_raw J2={b[2]:.4f} J3={b[3]:.4f} J4={b[4]:.4f}")
    tee(f"  sign equal across joints ?  {s[2]} {s[3]} {s[4]}  -> {'PASS' if same_all else 'FAIL'}")
    tee(f"  |b_raw| order J2>=J3>=J4 ?  -> {'PASS' if order else 'FAIL (weak prediction)'}")
    verdict["aL"] = same_all  # ordering is the weak/secondary test

    # vL & tanh_vP: friction family sign-consistency per joint
    bv = col(res, "vL", "b_raw"); bp = col(res, "tanh_vP", "b_raw")
    sv = {j: sgn(bv[j], 1e-9) for j in GJ}; sp = {j: sgn(bp[j], 1e-9) for j in GJ}
    vL_consistent = len(set(sv.values())) == 1 and "0" not in sv.values()
    fric_match = all(sv[j] == sp[j] for j in GJ)
    tee(f"\nvL (viscous) + tanh_vP (Coulomb)  -- friction family sign:")
    tee(f"  vL sign      J2={sv[2]} J3={sv[3]} J4={sv[4]}  (same across joints ? {'PASS' if vL_consistent else 'FAIL'})")
    tee(f"  tanh_vP sign J2={sp[2]} J3={sp[3]} J4={sp[4]}")
    tee(f"  sign(vL)==sign(tanh_vP) per joint ?  -> {'PASS' if fric_match else 'FAIL'}")
    verdict["vL"] = vL_consistent
    verdict["tanh_vP"] = fric_match

    # coul_sin_c3: non-negligible somewhere
    c = contr("coul_sin_c3"); nonneg = max(abs(v) for v in c.values()) > 0.02
    tee(f"\ncoul_sin_c3 (load-dependent Coulomb):")
    tee(f"  contr J2={c[2]:.3f} J3={c[3]:.3f} J4={c[4]:.3f}  non-negligible ? -> {'PASS' if nonneg else 'FAIL'}")
    verdict["coul_sin_c3"] = nonneg

    # empirical features: no prior prediction -> n.a.
    for n in ("lag_vL_a", "lag_aL_a", "adiff"):
        c = contr(n)
        tee(f"\n{n}  (EMPIRICAL -- no rigid-body prediction):")
        tee(f"  contr J2={c[2]:.3f} J3={c[3]:.3f} J4={c[4]:.3f}  -> n.a. (graded empirical a priori)")

    # ---- final three-grade classification ----
    GRADE_MAP = {
        "sin_q2": "physics", "cos_c3": "physics", "cos_c4": "physics",
        "aL": "physics", "vL": "physics", "tanh_vP": "physics",
        "coul_sin_c3": "physics_weak",
        "lag_vL_a": "empirical", "lag_aL_a": "empirical", "adiff": "empirical",
    }
    tee("\n" + "=" * 84)
    tee("FINAL THREE-GRADE CLASSIFICATION")
    tee("=" * 84)
    tee(f"  {'feature':<13}{'EOM origin':<22}{'prior grade':<14}{'data verdict':<14}{'final grade'}")
    final = {}
    for n in SEL:
        eom = prior["features"][n]["eom_term"]
        pg = GRADE_MAP[n]
        if pg == "empirical":
            v = "n.a."
            fg = "empirical (mechanism unverified)"
        else:
            ok = verdict.get(n, None)
            v = "matches" if ok else "MISMATCH"
            if pg == "physics":
                fg = "physics + data-confirmed" if ok else "physics-derived but DATA MISMATCH"
            else:
                fg = "physics(weak) + data-confirmed" if ok else "physics(weak), DATA MISMATCH"
        final[n] = fg
        tee(f"  {n:<13}{eom[:21]:<22}{pg:<14}{v:<14}{fg}")

    json.dump(dict(prior_sha256=h, prior_frozen=prior["_meta"]["frozen_at_utc"],
                   verdicts={k: bool(v) for k, v in verdict.items()},
                   final_grade=final,
                   coeff_raw={n: col(res, n, "b_raw") for n in SEL},
                   contrib={n: col(res, n, "contrib") for n in SEL}),
              open("physics_check.json", "w"), indent=2)
    tee("\nwrote physics_check.json + results/physics_check.txt")
    OUT.close()


if __name__ == "__main__":
    main()
