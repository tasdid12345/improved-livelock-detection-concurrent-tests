"""Livelock detection on synthetic thread traces: no-progress timeout baseline vs class-balanced LightGBM."""
import json
import sys
from sklearn.base import clone
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve

SEED = 42
SEEDS = [42, 43, 44]          # seeds for the randomized improved model (baseline is run with them too)
N_TRACES = 6000               # synthetic episodes; far below the 200k cap
OUT = "results.json"


def simulate(rng):
    """Simulate one thread wait episode -> (states, shared-write flags, label). label 1 = livelock."""
    L = rng.integers(60, 301)
    t = np.arange(L)
    if rng.random() < 0.5:
        # Livelock: busy-wait cycling over a few states, no real progress.
        k, p = rng.integers(2, 5), rng.integers(2, 5)
        cycle = rng.integers(0, k, size=p)
        states = np.where(rng.random(L) < 0.15, rng.integers(0, k, L), cycle[t % p])
        # 30% of livelocks rewrite shared memory with the same value (spurious "progress").
        rate = 0.02 if rng.random() < 0.3 else 0.0
        return states, rng.random(L) < rate, 1
    if rng.random() < 0.5:
        # Slow lock holder: mostly waiting in one state, released late, then progresses.
        k = rng.integers(2, 4)
        states = np.where(rng.random(L) < 0.9, 0, rng.integers(1, k + 1, L))
        rel = rng.integers(int(0.3 * L), L)
        writes = (t >= rel) & (rng.random(L) < 0.2)
        writes[rel] = True
        return states, writes, 0
    # Legitimate sleeper: backoff sleeps with counter-like states and periodic progress writes.
    states = np.where(rng.random(L) < 0.6, 0, rng.integers(1, 20, L))
    return states, rng.random(L) < 0.05, 0


def features(states, writes):
    """Windowed thread-state features; column 1 (max gap between shared writes) drives the baseline."""
    L = len(states)
    w_idx = np.flatnonzero(writes)
    edges = np.concatenate(([-1], w_idx, [L - 1]))
    max_gap = np.diff(edges).max()                     # longest no-progress stretch
    tslw = L - 1 - (w_idx[-1] if len(w_idx) else -1)   # time since last shared write
    counts = np.unique(states, return_counts=True)[1]
    changes = np.mean(states[1:] != states[:-1])       # state-change rate
    period = max(np.mean(states[p:] == states[:-p]) for p in range(1, 5))  # repeat-cycle match
    last = states[-30:]                                # trailing window
    last_top = np.unique(last, return_counts=True)[1].max() / len(last)
    last_distinct = len(np.unique(last)) / len(last)
    return [L, max_gap, tslw, len(w_idx), len(w_idx) / L, len(counts) / L,
            counts.max() / L, changes, period, last_top, last_distinct]


def evaluate(pred, y):
    """Main metric (livelock F1), secondary: accuracy and false-positive rate on legitimate waits."""
    fpr = np.sum((pred == 1) & (y == 0)) / np.sum(y == 0)
    return f1_score(y, pred), accuracy_score(y, pred), fpr


def main():
    print("Simulating synthetic thread traces...")
    rng = np.random.default_rng(SEED)
    data = [simulate(rng) for _ in range(N_TRACES)]
    X = np.array([features(s, w) for s, w, _ in data], dtype=float)
    y = np.array([lab for _, _, lab in data])
    if not np.all(np.isfinite(X)) or len(np.unique(y)) < 2:
        sys.exit("Feature matrix invalid; cannot produce metrics.")

    # Single stratified split, fixed seed; test set is only used at the end.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"Train {len(ytr)} / test {len(yte)}")

    # Baseline: no-progress timeout; window tuned on TRAINING data only.
    grid_w = np.arange(10, 201, 10)
    train_f1 = [f1_score(ytr, Xtr[:, 1] > w) for w in grid_w]
    W = grid_w[int(np.argmax(train_f1))]
    print(f"Baseline timeout window (train-tuned): {W}")

    # Improved: class-balanced LightGBM; hyperparameters by CV on train only.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = lgb.LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=SEED, verbose=-1)
    params = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1],
              "n_estimators": [200, 400], "min_child_samples": [10, 30]}
    gs = GridSearchCV(base, params, scoring="f1", cv=cv).fit(Xtr, ytr)
    print(f"Best CV params: {gs.best_params_}, CV F1={gs.best_score_:.3f}")

    # Decision threshold chosen on out-of-fold training predictions (no test info).
    oof = cross_val_predict(clone(gs.best_estimator_), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    p, r, thr = precision_recall_curve(ytr, oof)
    f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    thr_best = thr[int(np.argmax(f))]
    print(f"OOF-selected threshold: {thr_best:.3f}")

    # Test evaluation, same data and metric for both models, over seeds 42-44.
    b_res, i_res = [], []
    for s in SEEDS:
        b_pred = (Xte[:, 1] > W).astype(int)
        b_res.append(evaluate(b_pred, yte))
        model = clone(gs.best_estimator_).set_params(random_state=s).fit(Xtr, ytr)
        i_pred = (model.predict_proba(Xte)[:, 1] >= thr_best).astype(int)
        i_res.append(evaluate(i_pred, yte))
        print(f"seed {s}: baseline F1={b_res[-1][0]:.3f}  improved F1={i_res[-1][0]:.3f}")
    b, i = np.array(b_res), np.array(i_res)
    if not np.all(np.isfinite(i)):
        sys.exit("Non-finite metrics.")

    bm, im = b.mean(0), i.mean(0)
    results = {
        "question": "Can a class-balanced gradient-boosting classifier on windowed thread-state features "
                    "separate true livelocks from long legitimate waits better than a standard no-progress timeout detector?",
        "method": "Synthetic thread wait traces; stratified 75/25 split (seed 42); baseline = max-gap no-progress "
                  "timeout with window tuned on train; improved = class-balanced LightGBM on windowed features "
                  "with CV-tuned hyperparameters and OOF-tuned threshold; test set used once.",
        "dataset": {"name": "Synthetic concurrent-thread wait traces (numpy-generated)",
                    "source_url": "https://scikit-learn.org",
                    "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "livelock-class F1",
        "higher_is_better": True,
        "baseline": f"No-progress timeout: flag a thread if no shared-state write occurs within a window of {int(W)} steps (window tuned on train).",
        "improved": "Class-balanced LightGBM on windowed features (state-repeat counts, time since last shared write, "
                    "wait length, distinct-state ratio, etc.) with CV-tuned hyperparameters and threshold.",
        "metrics": {
            "baseline_score": float(bm[0]), "improved_score": float(im[0]),
            "improvement": float(im[0] - bm[0]),
            "baseline_score_std": float(b[:, 0].std()), "improved_score_std": float(i[:, 0].std()),
            "baseline_accuracy": float(bm[1]), "improved_accuracy": float(im[1]),
            "baseline_fpr": float(bm[2]), "improved_fpr": float(im[2]),
            "baseline_window": float(W), "improved_threshold": float(thr_best),
        },
        "notes": "Synthetic traces with hand-designed generative assumptions, not the paper's programs or logs; "
                 "results depend on those assumptions. The baseline is deterministic, so its seed runs are identical "
                 "(std 0). Only the LightGBM model varies with seed; the CV folds and threshold use seed 42. "
                 "Single split, so differences are indicative rather than statistically established.",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Optional chart: baseline vs improved livelock F1 and legit FPR.
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["F1 (livelock)", "FPR (legit)"]
    xs = np.arange(2)
    ax.bar(xs - 0.2, [bm[0], bm[2]], 0.4, label="baseline timeout")
    ax.bar(xs + 0.2, [im[0], im[2]], 0.4, label="improved LightGBM")
    ax.set_xticks(xs, labels)
    ax.legend()
    ax.set_title("Baseline vs improved (test set)")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    print(f"Wrote {OUT} and figures/baseline_vs_improved.png")


if __name__ == "__main__":
    main()
