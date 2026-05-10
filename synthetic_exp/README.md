# Synthetic Experiments

This directory contains the synthetic experiments for validating the learning dynamics in the paper.

## Installation

The synthetic scripts require:

```bash
pip install numpy pandas matplotlib tqdm
```

## Run

Run commands from the repository root.

Run adversarial training:

```bash
python synthetic_exp/run_at.py
```

Run adversarial distillation:

```bash
python synthetic_exp/run_ad.py
```

By default, the scripts run:

```text
p_un = 0.00, 0.05, 0.10, 0.15, 0.20
```

For AD, both teacher types are run by default:

```text
teacher = bad, good
```

These defaults match Appendix A.5: `d=100`, `n_train=200`, `n_test=100`,
`P=4`, `alpha=5`, `sigma_n=0.4`, `m=80`, `sigma0=0.01`, `steps=4000`,
`lr=0.01`, `eps=0.5`, and `PGD-20` evaluation.

## Plot

```bash
python synthetic_exp/plot.py --no_show
```

This saves:

```text
synthetic_exp/figures/split_synt_at.pdf
synthetic_exp/figures/split_synt_bad.pdf
synthetic_exp/figures/split_synt_good.pdf
```

