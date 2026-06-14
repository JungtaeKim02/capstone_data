"""Feature library + data loading for EXPERIMENTAL feature selection.

Reuses the EXACT x0 channel pipeline (SG filters, RAD conversion, valid==1)
from model_info/code/x0_features.py so results transfer to production.

Every candidate feature is an individually toggleable column, and every numeric
constant lives in a params dict P so it can be grid-searched. State-only: all
features come from joint POSITION; load/current are TARGETS, never inputs.

NO fabricated numbers -- every value is computed from the recorded CSVs.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

EXC = "/home/kimjungtae/sts_3215_ws/test16_new_arm/excitation"
RAD = 2 * np.pi / 4096.0
DT = 0.003
JOINTS = range(1, 6)
GJ = (2, 3, 4)

# Default constants == the ones x0_final.py / x0_features.py currently use.
# The constant grid-search overrides individual keys to justify each number.
DEFAULTS = dict(
    sg_v=41, sg_a=41, sg_vs=11, sg_as=11, sg_vp=81,   # SG window taps
    fc_slow=0.02, fc_fast=0.2, fc_strib=0.1,          # friction velocity scales (rad/s)
    lag_a=50, lag_b=150,                              # history lag depths (samples; *3ms)
)


def sg(x, w, d):
    return savgol_filter(x, w, 3, deriv=d, delta=DT)


def lag(x, k):
    o = np.empty_like(x)
    o[:k] = x[0]
    o[k:] = x[:-k]
    return o


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def channels(path, P=DEFAULTS, keep_step=False):
    """Smoothed kinematic channels + raw load/current targets from one recording."""
    cols = (["valid", "step_idx"] + [f"q{j}_meas" for j in JOINTS]
            + [f"load{j}" for j in JOINTS] + [f"cur{j}_raw" for j in JOINTS])
    d = pd.read_csv(path, usecols=cols)
    d = d[d["valid"] == 1].reset_index(drop=True)
    ch = {"q": {}, "vL": {}, "aL": {}, "vS": {}, "aS": {}, "vP": {},
          "load": {}, "cur": {}}
    for j in JOINTS:
        qr = (d[f"q{j}_meas"].to_numpy(float) - 2048.0) * RAD
        ch["q"][j] = sg(qr, 41, 0)                       # position never grid-searched
        ch["vL"][j] = sg(qr, P["sg_v"], 1)
        ch["aL"][j] = sg(qr, P["sg_a"], 2)
        ch["vS"][j] = sg(qr, P["sg_vs"], 1)
        ch["aS"][j] = sg(qr, P["sg_as"], 2)
        ch["vP"][j] = sg(qr, P["sg_vp"], 1)
        ch["load"][j] = d[f"load{j}"].to_numpy(float)
        ch["cur"][j] = d[f"cur{j}_raw"].to_numpy(float)
    if keep_step:
        return ch, d["step_idx"].to_numpy()
    return ch


# ----------------------------------------------------------------------------
# Candidate feature library.  Each entry: name -> (group, fn(ch, j, P) -> 1d).
# Groups let us reason about physics; selection operates on individual columns.
# ----------------------------------------------------------------------------
def _cum(ch):
    q = ch["q"]
    return q[2] + q[3], q[2] + q[3] + q[4]      # c3, c4 cumulative link angles


def build_library():
    R = {}      # name -> dict(group=, fn=, uses=set_of_constant_keys)

    def add(name, group, fn, uses=()):
        R[name] = dict(group=group, fn=fn, uses=set(uses))

    # --- bias ---
    add("bias", "bias", lambda ch, j, P: np.ones(len(ch["vL"][j])))

    # --- posture / gravity (global arm configuration; shared across joints) ---
    add("sin_q2", "posture", lambda ch, j, P: np.sin(ch["q"][2]))
    add("cos_q2", "posture", lambda ch, j, P: np.cos(ch["q"][2]))
    add("sin_q3", "posture", lambda ch, j, P: np.sin(ch["q"][3]))
    add("cos_q3", "posture", lambda ch, j, P: np.cos(ch["q"][3]))
    add("sin_q4", "posture", lambda ch, j, P: np.sin(ch["q"][4]))
    add("cos_q4", "posture", lambda ch, j, P: np.cos(ch["q"][4]))
    add("sin_c3", "posture", lambda ch, j, P: np.sin(_cum(ch)[0]))
    add("cos_c3", "posture", lambda ch, j, P: np.cos(_cum(ch)[0]))
    add("sin_c4", "posture", lambda ch, j, P: np.sin(_cum(ch)[1]))
    add("cos_c4", "posture", lambda ch, j, P: np.cos(_cum(ch)[1]))

    # --- posture harmonics (output-side cogging / gear eccentricity, per joint) ---
    add("sin2_qj", "posture_harm", lambda ch, j, P: np.sin(2 * ch["q"][j]))
    add("cos2_qj", "posture_harm", lambda ch, j, P: np.cos(2 * ch["q"][j]))
    add("sin3_qj", "posture_harm", lambda ch, j, P: np.sin(3 * ch["q"][j]))
    add("cos3_qj", "posture_harm", lambda ch, j, P: np.cos(3 * ch["q"][j]))

    # --- viscous friction (symmetric) ---
    add("vL", "visc", lambda ch, j, P: ch["vL"][j])
    add("vL_absvL", "visc", lambda ch, j, P: ch["vL"][j] * np.abs(ch["vL"][j]))
    add("vL3", "visc", lambda ch, j, P: ch["vL"][j] ** 3)

    # --- Coulomb friction (smooth sign at two velocity scales) ---
    add("coul_slow", "coulomb", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_slow"]), uses=["fc_slow"])
    add("coul_fast", "coulomb", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_fast"]), uses=["fc_fast"])

    # --- Stribeck (velocity-weakening) ---
    add("strib", "stribeck", lambda ch, j, P: ch["vL"][j] * np.exp(-(ch["vL"][j] / P["fc_strib"]) ** 2), uses=["fc_strib"])

    # --- asymmetric friction (+v vs -v different magnitude) ---
    add("relu_vp", "asym", lambda ch, j, P: np.maximum(ch["vL"][j], 0.0))
    add("relu_vn", "asym", lambda ch, j, P: np.minimum(ch["vL"][j], 0.0))

    # --- load-modulated Coulomb friction (gearbox normal force ~ gravity torque) ---
    add("coul_cos_c3", "fric_load", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_slow"]) * np.cos(_cum(ch)[0]), uses=["fc_slow"])
    add("coul_cos_c4", "fric_load", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_slow"]) * np.cos(_cum(ch)[1]), uses=["fc_slow"])
    add("coul_sin_c3", "fric_load", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_slow"]) * np.sin(_cum(ch)[0]), uses=["fc_slow"])
    add("coul_sin_c4", "fric_load", lambda ch, j, P: np.tanh(ch["vL"][j] / P["fc_slow"]) * np.sin(_cum(ch)[1]), uses=["fc_slow"])

    # --- inertia + configuration-dependent coupling ---
    add("aL", "inertia", lambda ch, j, P: ch["aL"][j])
    add("aL_cos_c3", "inertia", lambda ch, j, P: ch["aL"][j] * np.cos(_cum(ch)[0]))
    add("aL_cos_c4", "inertia", lambda ch, j, P: ch["aL"][j] * np.cos(_cum(ch)[1]))

    # --- centrifugal / Coriolis ---
    add("v2v3", "coriolis", lambda ch, j, P: ch["vL"][2] * ch["vL"][3])
    add("v3v4", "coriolis", lambda ch, j, P: ch["vL"][3] * ch["vL"][4])
    add("vL_sq", "coriolis", lambda ch, j, P: ch["vL"][j] ** 2)

    # --- short-band (11-tap) velocity/accel: faster transients ---
    add("vS", "shortband", lambda ch, j, P: ch["vS"][j])
    add("aS", "shortband", lambda ch, j, P: ch["aS"][j])
    add("aS_cos_c3", "shortband", lambda ch, j, P: ch["aS"][j] * np.cos(_cum(ch)[0]))

    # --- history / lag (drivetrain memory, transport delay) ---
    add("tanh_vP", "history", lambda ch, j, P: np.tanh(ch["vP"][j] / P["fc_slow"]), uses=["fc_slow"])
    add("lag_vL_a", "history", lambda ch, j, P: lag(ch["vL"][j], P["lag_a"]), uses=["lag_a"])
    add("lag_vL_b", "history", lambda ch, j, P: lag(ch["vL"][j], P["lag_b"]), uses=["lag_b"])
    add("lag_aL_a", "history", lambda ch, j, P: lag(ch["aL"][j], P["lag_a"]), uses=["lag_a"])

    # --- accel-band difference (gated short vs long accel) ---
    add("adiff", "adiff", lambda ch, j, P: ch["aS"][j] - ch["aL"][j])
    add("adiff_cos_c3", "adiff", lambda ch, j, P: (ch["aS"][j] - ch["aL"][j]) * np.cos(_cum(ch)[0]))
    add("adiff_cos_c4", "adiff", lambda ch, j, P: (ch["aS"][j] - ch["aL"][j]) * np.cos(_cum(ch)[1]))

    # --- backlash / hysteresis: lagged direction memory ---
    add("lag_dir", "backlash", lambda ch, j, P: lag(np.tanh(ch["vL"][j] / P["fc_slow"]), P["lag_a"]), uses=["fc_slow", "lag_a"])

    return R


LIB = build_library()

# The current production 32-feature load model, expressed as a name list, so we
# can reproduce it exactly inside the same harness and compare apples-to-apples.
X0_LOAD_32 = [
    "bias",
    "sin_q2", "cos_q2", "sin_q3", "cos_q3", "sin_q4", "cos_q4",
    "sin_c3", "cos_c3", "sin_c4", "cos_c4",
    "coul_slow", "coul_fast", "vL", "vL_absvL", "strib",
    "aL", "aL_cos_c3", "aL_cos_c4",
    "v2v3", "v3v4", "vL_sq",
    "vS", "aS", "aS_cos_c3",
    "tanh_vP", "lag_vL_a", "lag_vL_b", "lag_aL_a",
    "adiff", "adiff_cos_c3", "adiff_cos_c4",
]


def build_X(names, ch, j, P=DEFAULTS):
    """Stack the selected feature columns into a design matrix."""
    return np.column_stack([LIB[n]["fn"](ch, j, P) for n in names])


if __name__ == "__main__":
    # sanity: reproduce the 32-col load matrix and confirm it matches x0_features
    import sys
    sys.path.insert(0, "/home/kimjungtae/sts_3215_ws/model_info/code")
    import x0_features as xf
    ch = channels(f"{EXC}/run_1/excitation_recording.csv")
    for j in GJ:
        X_new = build_X(X0_LOAD_32, ch, j)
        X_ref = xf.feat_load(ch, j)
        err = np.max(np.abs(X_new - X_ref))
        print(f"J{j}: lib 32-col vs x0_features.feat_load  max|diff|={err:.3e}  "
              f"shape={X_new.shape}  {'OK' if err < 1e-9 else 'MISMATCH'}")
    print(f"library size = {len(LIB)} candidate columns")
