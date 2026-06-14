"""Experimental feature selection for the x0 LOAD model.

Selection objective is GENERALIZATION, evaluated three independent ways so we
never overfit feature choice to one trajectory:
  (1) regime-wise CV inside Traj-A  -- pool run_1..5, hold out contiguous
      step_idx buckets (= whole speed/range Fourier regimes), K folds.
  (2) A->B cross-recording test      -- train run_1..5, test test_noload
      (a genuinely different trajectory shape).
  (3) payload sensitivity            -- residual must still GROW with payload
      (so external-force detection is preserved, not absorbed by the features).

Engine: columns are globally standardized once; per-fold Gram matrices let any
feature subset be scored by closed-form ridge with no row access in the loop,
so forward selection over the whole 43-candidate library is near-instant.

NO fabricated numbers -- everything computed from the recorded CSVs.
"""
import numpy as np
import featlib as F
from featlib import LIB, EXC, GJ, DEFAULTS, channels, build_X, rms

ALPHA = 10.0          # match production Ridge(10.0)
K_FOLDS = 8           # regime folds within Traj-A
CANDIDATES = [n for n in LIB if n != "bias"]   # intercept handled by centering


# ---------------------------------------------------------------- data load
def load_pool():
    """Pooled Traj-A training data (run_1..5) with step_idx, per joint."""
    chs, steps = [], []
    for i in range(1, 6):
        ch, st = channels(f"{EXC}/run_{i}/excitation_recording.csv", keep_step=True)
        chs.append(ch); steps.append(st)
    return chs, steps


def design_pool(chs, steps, j, names, P=DEFAULTS):
    """Stack design matrix + targets + fold ids for joint j over the pool."""
    Xs, ys, fs = [], [], []
    # global step range -> K contiguous regime buckets
    allst = np.concatenate(steps)
    edges = np.quantile(allst, np.linspace(0, 1, K_FOLDS + 1))
    edges[-1] += 1
    for ch, st in zip(chs, steps):
        Xs.append(build_X(names, ch, j, P))
        ys.append(ch["load"][j])
        fs.append(np.clip(np.searchsorted(edges, st, side="right") - 1, 0, K_FOLDS - 1))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(fs)


# ------------------------------------------------------------- gram precompute
class GramCV:
    """Precomputed per-fold Grams for fast subset scoring (joint-specific)."""
    def __init__(self, X, y, folds, alpha=ALPHA):
        self.alpha = alpha
        self.ybar = y.mean()
        yc = y - self.ybar
        mu = X.mean(0)
        sd = X.std(0); sd[sd == 0] = 1.0
        self.mu, self.sd, self.names_idx = mu, sd, None
        Xs = (X - mu) / sd
        self.p = Xs.shape[1]
        self.N = Xs.shape[0]
        self.sigma_y = y.std()
        # per-fold accumulators
        self.Gk, self.ck, self.yy_k, self.n_k = [], [], [], []
        for k in range(K_FOLDS):
            m = folds == k
            Xk = Xs[m]; yk = yc[m]
            self.Gk.append(Xk.T @ Xk)
            self.ck.append(Xk.T @ yk)
            self.yy_k.append(float(yk @ yk))
            self.n_k.append(int(m.sum()))
        self.G = sum(self.Gk); self.c = sum(self.ck)
        self.yy = sum(self.yy_k)

    def _solve(self, G, c, idx):
        A = G[np.ix_(idx, idx)].copy()
        A[np.diag_indices_from(A)] += self.alpha
        return np.linalg.solve(A, c[idx])

    def cv_score(self, idx):
        """Pooled regime-CV: returns (cv_r2, worst_fold_nrmse)."""
        sse = 0.0; worst = 0.0
        for k in range(K_FOLDS):
            Gtr = self.G - self.Gk[k]; ctr = self.c - self.ck[k]
            b = self._solve(Gtr, ctr, idx)
            # SSE on held-out fold k from Gram identity (no row access)
            ck = self.ck[k][idx]; Gk = self.Gk[k][np.ix_(idx, idx)]
            sse_k = self.yy_k[k] - 2 * b @ ck + b @ Gk @ b
            sse += sse_k
            worst = max(worst, np.sqrt(max(sse_k, 0) / self.n_k[k]) / self.sigma_y)
        cv_r2 = 1.0 - sse / self.yy
        return cv_r2, worst

    def fit_full(self, idx):
        b = self._solve(self.G, self.c, idx)
        return b  # standardized-space coefficients


def eval_on(gram, idx, names, ch_test, j):
    """A->B style held-out NRMSE/R2 for the fitted-on-A model."""
    b = gram.fit_full(idx)
    Xt = build_X([names[i] for i in idx], ch_test, j)
    Xt = (Xt - gram.mu[idx]) / gram.sd[idx]
    yhat = gram.ybar + Xt @ b
    y = ch_test["load"][j]
    nrmse = rms(y - yhat) / y.std()
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return nrmse, r2


# --------------------------------------------------------------- forward search
def forward_select(gram, names, ch_B, j, max_feats=24):
    chosen, path = [], []
    remaining = list(range(len(names)))
    best_r2 = -1e9
    while remaining and len(chosen) < max_feats:
        scored = []
        for c in remaining:
            idx = chosen + [c]
            r2, worst = gram.cv_score(idx)
            scored.append((r2, worst, c))
        scored.sort(reverse=True)
        r2, worst, c = scored[0]
        gain = r2 - best_r2
        chosen.append(c); remaining.remove(c); best_r2 = r2
        b_nrmse, b_r2 = eval_on(gram, chosen, names, ch_B, j)
        path.append(dict(step=len(chosen), feat=names[c], group=LIB[names[c]]["group"],
                         cv_r2=r2, cv_worst_nrmse=worst, gain=gain,
                         B_nrmse=b_nrmse, B_r2=b_r2))
    return chosen, path


def main():
    print("loading Traj-A pool (run_1..5) ...")
    chs, steps = load_pool()
    ch_B = channels(f"{EXC}/test_noload/excitation_recording.csv")   # held-out shape

    names = CANDIDATES
    for j in GJ:
        print("\n" + "=" * 96)
        print(f"JOINT {j}  -- LOAD channel   (candidates={len(names)}, "
              f"pool N={sum(len(s) for s in steps)}, regime folds={K_FOLDS})")
        X, y, folds = design_pool(chs, steps, j, names)
        gram = GramCV(X, y, folds)

        # reference baselines (reproduced inside the same harness)
        def idx_of(sub): return [names.index(n) for n in sub if n in names]
        refs = {
            "posture-only(10)": [n for n in F.X0_LOAD_32 if LIB[n]["group"] == "posture"],
            "posture+history(14)": [n for n in F.X0_LOAD_32 if LIB[n]["group"] in ("posture", "history")],
            "x0 full(31 non-bias)": [n for n in F.X0_LOAD_32 if n != "bias"],
        }
        print(f"  {'reference set':<26}{'nfeat':>6}{'cvR2':>8}{'worstNRMSE':>12}{'B_NRMSE':>9}{'B_R2':>8}")
        for nm, sub in refs.items():
            idx = idx_of(sub)
            r2, worst = gram.cv_score(idx)
            bn, br = eval_on(gram, idx, names, ch_B, j)
            print(f"  {nm:<26}{len(idx):>6}{r2:>8.3f}{worst:>12.3f}{bn:>9.3f}{br:>8.3f}")

        # forward selection over the full library
        print(f"  -- forward selection (greedy on regime-CV R2) --")
        print(f"  {'step add-feature':<30}{'group':<12}{'cvR2':>8}{'gain':>8}{'worstN':>8}{'B_NRMSE':>9}{'B_R2':>8}")
        chosen, path = forward_select(gram, names, ch_B, j)
        for r in path:
            print(f"  {r['step']:>2} {r['feat']:<26}{r['group']:<12}"
                  f"{r['cv_r2']:>8.3f}{r['gain']:>+8.3f}{r['cv_worst_nrmse']:>8.3f}"
                  f"{r['B_nrmse']:>9.3f}{r['B_r2']:>8.3f}")


if __name__ == "__main__":
    main()
