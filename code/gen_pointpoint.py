#!/usr/bin/env python3
"""Point-to-point (rest-to-rest) NO-LOAD test trajectory -- a DIFFERENT type
from the Fourier excitation, to test feature generalization across trajectory
TYPES (not just seeds).

It visits random collision-safe poses via cosine rest-to-rest moves at varied
speeds, with a short dwell (hold) at each pose. This deliberately samples
regimes the space-filling excitation under-covers:
  - quasi-static gravity (the dwells: v~0, a~0 -> pure posture/gravity)
  - clean single-direction moves at near-constant speed (steady friction)

Reuses the validated FK collision / limit checks + CSV writer.
Output format matches the excitation CSV (t,q1..q5,dq1..q5,ddq1..q5; q=steps).
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/kimjungtae/sts_3215_ws")
from generate_excitation_trajectory import (
    MEASURED_LIMITS_STEPS, JOINT_NAMES, STEP_TO_DEG, NEUTRAL_STEP,
    check_collision, save_csv,
)

DT = 0.003


def cosine_transition(q0, q1, n):
    k = np.arange(1, n + 1)
    a = 0.5 * (1.0 - np.cos(np.pi * k / n))
    return q0[None, :] * (1 - a[:, None]) + q1[None, :] * a[:, None]


def sample_pose(rng, limits, inset=0.08):
    js = sorted(limits)
    q = np.empty(5)
    for i, j in enumerate(limits):
        lo, hi = limits[j]
        m = (hi - lo) * inset
        q[i] = rng.uniform(lo + m, hi - m)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--target-sec", type=float, default=220.0)
    ap.add_argument("--min-dist", type=float, default=2.5)
    ap.add_argument("--floor-z", type=float, default=1.0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    limits = MEASURED_LIMITS_STEPS
    joints = sorted(limits)
    rng = np.random.default_rng(args.seed)
    md, fz = args.min_dist / 100.0, args.floor_z / 100.0
    home = np.array([NEUTRAL_STEP] * 5, float)

    segs = [np.tile(home, (int(1.0 / DT), 1))]
    prev = home.copy()
    n_moves = 0
    total = 1.0
    while total < args.target_sec:
        # find a safe target + safe transition
        for _ in range(200):
            tgt = sample_pose(rng, limits)
            disp_deg = float(np.max(np.abs(tgt - prev)) * STEP_TO_DEG)
            v_tar = rng.uniform(20.0, 38.0)          # cosine peak ~1.57*v_tar <= 60 deg/s (servo saturation cap)
            Tmove = max(1.2, disp_deg / v_tar)        # cosine peak ~ (pi/2)*disp/Tmove
            nmove = int(round(Tmove / DT))
            trans = cosine_transition(prev, tgt, nmove)
            ok, _ = check_collision(trans, md, fz, check_interval=3)
            if ok:
                break
        else:
            continue
        dwell = int(round(rng.uniform(0.4, 1.2) / DT))
        segs.append(trans)
        segs.append(np.tile(tgt, (dwell, 1)))
        prev = tgt.copy()
        n_moves += 1
        total += nmove * DT + dwell * DT

    segs.append(cosine_transition(prev, home, int(round(1.6 / DT))))
    segs.append(np.tile(home, (int(1.0 / DT), 1)))

    q_full = np.vstack(segs)
    N = len(q_full)
    t_full = np.arange(N) * DT
    dq = np.gradient(q_full, DT, axis=0)
    ddq = np.gradient(dq, DT, axis=0)

    safe, info = check_collision(q_full, md, fz, check_interval=1)
    lim_ok = True
    for i, j in enumerate(joints):
        lo, hi = limits[j]
        mn, mx = q_full[:, i].min(), q_full[:, i].max()
        ok = (mn >= lo) and (mx <= hi)
        lim_ok &= ok
        vmax = float(np.max(np.abs(dq[:, i])) * STEP_TO_DEG)
        print(f"  J{j} {JOINT_NAMES[j]:14s}: [{mn:.0f},{mx:.0f}] limit[{lo},{hi}] "
              f"v_max={vmax:6.1f} deg/s {'OK' if ok else 'VIOLATION'}")
    print(f"\n{n_moves} moves, {N} rows, {N*DT:.1f}s ({N*DT/60:.1f}min)")
    print(f"full collision: min_link={info['min_link_dist_cm']:.1f}cm "
          f"min_z={info['min_z_cm']:.1f}cm -> {'SAFE' if safe else 'COLLISION'}")

    csv = args.output_dir / "krr_excitation_trajectory.csv"   # same filename the executor expects
    save_csv(str(csv), t_full, q_full, dq, ddq, joints)
    (args.output_dir / "pointpoint_meta.json").write_text(json.dumps(dict(
        type="point_to_point_rest_to_rest", seed=args.seed, dt=DT, n_rows=N,
        total_sec=N * DT, n_moves=n_moves, collision_safe_full=bool(safe),
        min_link_cm=round(info["min_link_dist_cm"], 2), limits_ok=bool(lim_ok),
        created=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=2))
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
