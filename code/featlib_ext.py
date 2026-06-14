"""EXTENDED candidate library = base x0 library + the new family that Stage-1
residual discovery proved is missing: CROSS-JOINT (off-diagonal) inertia.

Physics (EOM): tau_i = sum_k M_ik(q) qddot_k + ... .  The base library only has
the DIAGONAL term M_ii (self `aL`); discovery showed the elbow/wrist torque is
strongly driven by the SHOULDER's acceleration (J3<-aL2 partial-R2=0.065), i.e.
the off-diagonal M_ik is real and unmodeled.

Encoding matches the library's existing cross-term convention (v2v3, v3v4): the
feature is a GLOBAL signal identical for every joint, each joint gets its own
coefficient -- so coefficient(joint i, feature aLk) = M_ik.  Off-diagonal inertia
in a serial arm depends on the relative link angle, so each global accel is also
offered modulated by cos(c3), cos(c4).

build_Xe / EXTLIB / CANDS_EXT mirror featlib's build_X / LIB so the same GramCV
and greedy harness run unchanged.
"""
import numpy as np
import featlib as F
from featlib import LIB as BASE


def _cum(ch):
    q = ch["q"]
    return q[2] + q[3], q[2] + q[3] + q[4]


def build_ext():
    R = dict(BASE)                      # start from the full 43-col base library

    def add(name, group, fn):
        R[name] = dict(group=group, fn=fn, uses=set())

    # cross-joint off-diagonal inertia: global accel of joint k, modulated by config
    for k in (2, 3, 4):
        add(f"aL{k}g", "xinertia", lambda ch, j, P, kk=k: ch["aL"][kk])
        add(f"aL{k}g_cc3", "xinertia", lambda ch, j, P, kk=k: ch["aL"][kk] * np.cos(_cum(ch)[0]))
        add(f"aL{k}g_cc4", "xinertia", lambda ch, j, P, kk=k: ch["aL"][kk] * np.cos(_cum(ch)[1]))
    return R


EXTLIB = build_ext()
CANDS_EXT = [n for n in EXTLIB if n != "bias"]


def build_Xe(names, ch, j, P=F.DEFAULTS):
    return np.column_stack([EXTLIB[n]["fn"](ch, j, P) for n in names])
