"""STAGE 1 -- open-ended residual discovery (NOT constrained to the x0 vocabulary).

Fit the current slim-10 model (OLS, per joint) on the Traj-A train pool under the
CV-optimal constants, take the residual, and ask the DATA what structure remains:

  A. residual budget                -- how much variance is left to explain
  B. existing-library screen        -- which UNSELECTED library cols still carry
                                       residual signal (partial-R2 vs the 10)
  C. spatial-harmonic sweep         -- cogging / gear-eccentricity: sin/cos(w*qj)
                                       over a w grid; a peak at w>1 = real ripple
  D. friction velocity-curve        -- signed-velocity binned residual (asymmetry,
                                       stiction, Stribeck) + relu/strib candidates
  E. cross-joint inertia & Coriolis -- off-diagonal M (aL of OTHER joints) and
                                       velocity products v_i v_j / v_i^2
  F. load-modulated friction        -- Coulomb scaled by inertial/gravity normal load

Metric = PARTIAL R2: fraction of the CURRENT 10-feature residual variance removed
by adding that one candidate (orthogonalised against the 10) -- so a candidate that
is just a linear combo of the existing features reads ~0. Pure hypothesis
generation on TRAIN ONLY; every winner is later validated on the 6 held-out tests.

NO fabricated numbers -- everything computed from the recorded CSVs.
"""
import json
import numpy as np
import featlib as F
from featlib import LIB, GJ, DEFAULTS, build_X, lag
from sel5 import load_pool_runs

P_OPT = {**DEFAULTS, "sg_v": 61, "sg_a": 61, "sg_vp": 161, "lag_a": 25}
SEL = json.load(open("selected5.json"))["selected"]

OUT = open("results/discovery.txt", "w")


def tee(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


def pool_col(chs, fn, j):
    """Concatenate a per-run feature column across the 5 train runs."""
    return np.concatenate([fn(ch, j) for ch in chs])


def fit_residual(chs, j):
    """OLS residual of the slim-10 on the pooled train data for joint j."""
    X = np.vstack([build_X(SEL, ch, j, P_OPT) for ch in chs])
    y = np.concatenate([ch["load"][j] for ch in chs])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xc = Xs - Xs.mean(0)
    yc = y - y.mean()
    Q, _ = np.linalg.qr(Xc)                     # orthonormal basis of the 10
    b, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    r = yc - Xc @ b                             # residual, exactly _|_ span(Q)
    return r, Q, y


def partial_r2(r, Xc, Q):
    """Fraction of residual variance r removed by adding candidate block Xc
    (orthogonalised against Q). Xc: (N,) or (N,k), already centred is fine."""
    Xc = np.atleast_2d(Xc.T).T if Xc.ndim == 1 else Xc
    Xp = Xc - Q @ (Q.T @ Xc)                    # part of candidate _|_ the 10
    nrm = np.sqrt((Xp ** 2).sum(0)); nrm[nrm == 0] = 1.0
    Xp = Xp / nrm
    b, *_ = np.linalg.lstsq(Xp, r, rcond=None)
    rr = r - Xp @ b
    return float(1.0 - (rr @ rr) / (r @ r))


def main():
    tee("=" * 78)
    tee("STAGE 1  RESIDUAL DISCOVERY  (slim-10 OLS residual on train pool, P_OPT)")
    tee("  slim-10:", ", ".join(SEL))
    tee("  constants: sg_v=sg_a=61, sg_vp=161, lag_a=25")
    tee("=" * 78)

    chs, _ = load_pool_runs(P_OPT)
    res, Qb, yraw = {}, {}, {}
    for j in GJ:
        res[j], Qb[j], yraw[j] = fit_residual(chs, j)

    # ---- A. residual budget --------------------------------------------------
    tee("\nA. RESIDUAL BUDGET (how much is left to explain)")
    tee(f"  {'joint':<7}{'rms(load)':>11}{'rms(resid)':>12}{'trainR2':>9}{'resid/load%':>12}")
    for j in GJ:
        rl = F.rms(yraw[j] - yraw[j].mean()); rr = F.rms(res[j])
        tee(f"  J{j:<6}{rl:>11.2f}{rr:>12.2f}{1-(rr/rl)**2:>9.4f}{100*rr/rl:>11.1f}%")

    # ---- B. existing-library screen -----------------------------------------
    tee("\nB. UNSELECTED LIBRARY COLUMNS still carrying residual signal")
    tee("   (partial-R2 = extra train-resid variance each would remove, given the 10)")
    rest = [n for n in LIB if n not in SEL and n != "bias"]
    rowsB = []
    for n in rest:
        pr = [partial_r2(res[j], pool_col(chs, lambda c, jj: LIB[n]["fn"](c, jj, P_OPT), j), Qb[j]) for j in GJ]
        rowsB.append((float(np.mean(pr)), n, LIB[n]["group"], pr))
    tee(f"  {'candidate':<16}{'group':<11}{'J2':>8}{'J3':>8}{'J4':>8}{'mean':>9}")
    for m, n, g, pr in sorted(rowsB, reverse=True)[:14]:
        tee(f"  {n:<16}{g:<11}{pr[0]:>8.4f}{pr[1]:>8.4f}{pr[2]:>8.4f}{m:>9.4f}")

    # ---- C. spatial-harmonic sweep (cogging / eccentricity) -----------------
    tee("\nC. SPATIAL-HARMONIC SWEEP  sin/cos(w*qj)  -- cogging / gear ripple")
    tee("   (partial-R2 of the sin+cos PAIR at spatial frequency w, per joint)")
    ws = np.concatenate([np.arange(0.5, 12.01, 0.5), np.arange(13, 41, 1.0)])
    for j in GJ:
        q = pool_col(chs, lambda c, jj: c["q"][jj], j)
        peaks = []
        for w in ws:
            pair = np.column_stack([np.sin(w * q), np.cos(w * q)])
            peaks.append((partial_r2(res[j], pair, Qb[j]), float(w)))
        peaks.sort(reverse=True)
        span = q.max() - q.min()
        tee(f"  J{j}: range={np.degrees(span):.0f}deg  top spatial freqs (cyc/rad):")
        for pr, w in peaks[:4]:
            tee(f"        w={w:>5.1f}  partial-R2={pr:>7.4f}   (~{w*span/(2*np.pi):.1f} cycles over range)")

    # ---- D. friction velocity-curve ----------------------------------------
    tee("\nD. FRICTION VELOCITY-CURVE  (signed-velocity binned residual)")
    tee("   nonzero binned mean => unmodeled friction shape (asymmetry / Stribeck)")
    for j in GJ:
        v = pool_col(chs, lambda c, jj: c["vL"][jj], j)
        r = res[j]
        vmax = np.quantile(np.abs(v), 0.995)
        edges = np.linspace(-vmax, vmax, 13)
        tee(f"  J{j}  (vmax~{np.degrees(vmax):.0f}deg/s):")
        line = "       v(deg/s):"
        for k in range(len(edges) - 1):
            m = (v >= edges[k]) & (v < edges[k + 1])
            c = np.degrees((edges[k] + edges[k + 1]) / 2)
            mr = r[m].mean() if m.sum() > 50 else np.nan
            line += f"{c:>6.0f}:{mr:>6.1f}"
            if k % 4 == 3:
                tee(line); line = "                "
        if line.strip():
            tee(line)
    tee("   friction-shape candidates (partial-R2, mean over joints):")
    fric = {
        "relu_vp": lambda c, jj: np.maximum(c["vL"][jj], 0.0),
        "relu_vn": lambda c, jj: np.minimum(c["vL"][jj], 0.0),
        "sqrt_sgn_v": lambda c, jj: np.sign(c["vL"][jj]) * np.sqrt(np.abs(c["vL"][jj])),
        "stiction_bump": lambda c, jj: np.exp(-(c["vL"][jj] / 0.1) ** 2),
        "strib_fc05": lambda c, jj: c["vL"][jj] * np.exp(-(c["vL"][jj] / 0.05) ** 2),
        "strib_fc20": lambda c, jj: c["vL"][jj] * np.exp(-(c["vL"][jj] / 0.2) ** 2),
        "vL5": lambda c, jj: c["vL"][jj] ** 5,
    }
    for nm, fn in sorted(fric.items(), key=lambda kv: -np.mean([partial_r2(res[j], pool_col(chs, kv[1], j), Qb[j]) for j in GJ])):
        pr = [partial_r2(res[j], pool_col(chs, fn, j), Qb[j]) for j in GJ]
        tee(f"     {nm:<16}{pr[0]:>8.4f}{pr[1]:>8.4f}{pr[2]:>8.4f}   mean={np.mean(pr):>7.4f}")

    # ---- E. cross-joint inertia & Coriolis ----------------------------------
    tee("\nE. CROSS-JOINT COUPLING  (off-diagonal inertia + Coriolis/centrifugal)")
    others = {2: (3, 4), 3: (2, 4), 4: (2, 3)}
    tee("   off-diagonal inertia  aL[other]  and  aL[other]*cos(c3):")
    cross = {}
    for j in GJ:
        for o in others[j]:
            cross[f"J{j}<-aL{o}"] = (j, lambda c, jj, oo=o: c["aL"][oo])
            cross[f"J{j}<-aL{o}*cosc3"] = (j, lambda c, jj, oo=o: c["aL"][oo] * np.cos(c["q"][2] + c["q"][3]))
    rowsE = []
    for nm, (j, fn) in cross.items():
        pr = partial_r2(res[j], pool_col(chs, fn, j), Qb[j])
        rowsE.append((pr, nm))
    for pr, nm in sorted(rowsE, reverse=True)[:8]:
        tee(f"     {nm:<18}partial-R2={pr:>7.4f}")
    tee("   velocity products / squares (Coriolis & centrifugal):")
    prod = {
        "v2v4": lambda c, jj: c["vL"][2] * c["vL"][4],
        "v2sq": lambda c, jj: c["vL"][2] ** 2,
        "v3sq": lambda c, jj: c["vL"][3] ** 2,
        "v4sq": lambda c, jj: c["vL"][4] ** 2,
        "v2v3": lambda c, jj: c["vL"][2] * c["vL"][3],
        "v3v4": lambda c, jj: c["vL"][3] * c["vL"][4],
    }
    for nm, fn in prod.items():
        pr = [partial_r2(res[j], pool_col(chs, fn, j), Qb[j]) for j in GJ]
        tee(f"     {nm:<8}{pr[0]:>8.4f}{pr[1]:>8.4f}{pr[2]:>8.4f}   mean={np.mean(pr):>7.4f}")

    # ---- F. load-modulated friction -----------------------------------------
    tee("\nF. LOAD-MODULATED FRICTION  (Coulomb scaled by normal-load proxy)")
    tee("   gearbox friction ~ transmitted torque; proxies = |aL| (inertia), gravity")
    fc = P_OPT["fc_slow"]
    loadf = {
        "coul*|aL|": lambda c, jj: np.tanh(c["vL"][jj] / fc) * np.abs(c["aL"][jj]),
        "coul*|sinq2|": lambda c, jj: np.tanh(c["vL"][jj] / fc) * np.abs(np.sin(c["q"][2])),
        "coul*cosc4": lambda c, jj: np.tanh(c["vL"][jj] / fc) * np.cos(c["q"][2] + c["q"][3] + c["q"][4]),
        "coul*sinc4": lambda c, jj: np.tanh(c["vL"][jj] / fc) * np.sin(c["q"][2] + c["q"][3] + c["q"][4]),
        "|vL|*cosc3": lambda c, jj: np.abs(c["vL"][jj]) * np.cos(c["q"][2] + c["q"][3]),
    }
    for nm, fn in sorted(loadf.items(), key=lambda kv: -np.mean([partial_r2(res[j], pool_col(chs, kv[1], j), Qb[j]) for j in GJ])):
        pr = [partial_r2(res[j], pool_col(chs, fn, j), Qb[j]) for j in GJ]
        tee(f"     {nm:<16}{pr[0]:>8.4f}{pr[1]:>8.4f}{pr[2]:>8.4f}   mean={np.mean(pr):>7.4f}")

    tee("\n" + "=" * 78)
    tee("READ-OUT: any family with mean partial-R2 >~ 0.01 is worth adding to the")
    tee("library and re-validating on the 6 held-out trajectories (Stage 2).")
    tee("=" * 78)
    OUT.close()


if __name__ == "__main__":
    main()
