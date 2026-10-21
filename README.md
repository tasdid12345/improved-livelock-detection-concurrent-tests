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

## Reference

This project is motivated by the following paper:

> CS Lin, GH Hwang (2009). *Spontaneous detection of infinite loops and livelocks in dynamic testing of concurrent programs*. 2009 Third IEEE International Symposium. https://ieeexplore.ieee.org/abstract/document/5198516/

Paper author: Prof. Gwan-Hwan Hwang, National Taiwan Normal University (ghhwang@csie.ntnu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
