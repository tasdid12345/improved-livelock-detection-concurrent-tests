# Livelock vs. long legitimate waits: timeout baseline vs. gradient boosting

Motivation: As described in "Spontaneous detection of infinite loops and livelocks in dynamic testing of concurrent programs", detecting livelocks during dynamic testing of concurrent programs is hard because a thread that is merely waiting for a slow lock holder looks similar to one stuck in a busy-wait loop. Simple no-progress timeouts flag both, producing false alarms.

Question: Can a class-balanced gradient-boosting classifier on windowed thread-state features separate true livelocks from long legitimate waits better than a standard no-progress timeout detector?

Dataset: Synthetic thread wait traces generated with numpy at run time. This is not the paper's own dataset; its programs and logs are not public.

Baseline: a no-progress timeout that flags a thread when no shared-state write occurs within a fixed window. The window is tuned on the training split only.

Improved model: a class-balanced LightGBM on windowed features (state-repeat counts, time since last shared write, wait length, distinct-state ratio, and others). Hyperparameters are chosen by cross-validation on the training split, and the decision threshold is chosen from out-of-fold training predictions. The test set is used once.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and a chart to figures/).

## Results (latest run)

On a synthetic concurrent-thread wait-trace dataset (1,500 held-out test traces), a standard no-progress timeout detector reached 0.743 livelock-class F1, while a class-balanced LightGBM on windowed thread-state features reached 0.992 on the same test set, from a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.7426 |
| improved_score | 0.9924 |
| improvement | 0.2498 |
| baseline_score_std | 0 |
| improved_score_std | 0.0003166 |
| baseline_accuracy | 0.6807 |
| improved_accuracy | 0.9924 |
| baseline_fpr | 0.5606 |
| improved_fpr | 0.009321 |
| baseline_window | 60 |
| improved_threshold | 0.02717 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic traces with hand-designed generative assumptions, not the paper's programs or logs; results depend on those assumptions. The baseline is deterministic, so its seed runs are identical (std 0). Only the LightGBM model varies with seed; the CV folds and threshold use seed 42. Single split, so differences are indicative rather than statistically established.

## Reference

This project is motivated by the following paper:

> CS Lin, GH Hwang (2009). *Spontaneous detection of infinite loops and livelocks in dynamic testing of concurrent programs*. 2009 Third IEEE International Symposium. https://ieeexplore.ieee.org/abstract/document/5198516/

Paper author: Prof. Gwan-Hwan Hwang, National Taiwan Normal University (ghhwang@csie.ntnu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
