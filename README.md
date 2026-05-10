# Toward Understanding Adversarial Distillation: Why Robust Teachers Fail

This repository contains the official implementation of the ICML 2026 paper  
**"Toward Understanding Adversarial Distillation: Why Robust Teachers Fail."**

[Paper Link](https://arxiv.org/abs/2605.21999) | [OpenReview](https://openreview.net/forum?id=USejwpvj0a)

## TL;DR

- We study why robust teachers can fail in adversarial distillation.
- We identify a **robustly unlearnable set** that drives teacher-dependent robust overfitting.
- We show that confident teacher supervision on this set causes the student to memorize spurious noise.


## Experiments

Synthetic experiments:

See [synthetic_exp/README.md](synthetic_exp/README.md) for details.

Real-data experiments:

See [real_exp/README.md](real_exp/README.md) for details.

## Previous Work

This work builds on our previous empirical study of teacher-dependent failures in adversarial distillation:

**Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation**
Published in Transactions on Machine Learning Research, 2026.

[Paper Link](https://arxiv.org/abs/2512.10275) | [OpenReview](https://openreview.net/forum?id=ek45VamPCE) | [GitHub](https://github.com/HongsinLee/saad)

