# Real-data Experiments

This directory contains code for training reference robust models and identifying robustly unlearnable training samples.

## Installation

All experiments in this repo were run with:

```text
Python 3.8
PyTorch 2.4.1
````

Install the necessary packages using pip:

```bash
pip install torch torchvision numpy tqdm wandb robustbench
```

For loading pre-trained teacher models, we rely on RobustBench. Please visit the [RobustBench repository](https://github.com/RobustBench/robustbench) for setup and model downloads.

## Train reference models

```bash
bash real_exp/scripts/run_table1.sh
```

Default: ResNet-18 on CIFAR-10 with the paper hyperparameters
(`eps=8`, `epochs=200`, `batch=128`, `lr=0.1`, `wd=2e-4`).

Other architectures:

```bash
bash real_exp/scripts/run_table1.sh mnv2
bash real_exp/scripts/run_table1.sh wrn28
bash real_exp/scripts/run_table1.sh wrn34
```

This trains 60 reference models:

```text
PGD-AT, TRADES, Chen, Rebuffi, Bartoldson, Gowal
× 10 random seeds
```

Other datasets can be selected with the `DATASET` environment variable:

```bash
DATASET=cifar100 bash real_exp/scripts/run_table1.sh res18
DATASET=tinyimg bash real_exp/scripts/run_table1.sh res18
```

For CIFAR-100, the script uses `PGD-AT`, `TRADES`, `Chen`, `Wang28`,
`Wang70`, and `Gowal` (60 models). For Tiny-ImageNet, it uses `PGD-AT`,
`TRADES`, and the Tiny-ImageNet Wang teacher checkpoint (30 models).

> **Note.**
> - The full Table 1 setup trains 60 reference models, each for 200 epochs with 10-step PGD adversarial training, so it can be computationally expensive.
> - Jobs are automatically assigned to the GPUs listed in `GPUS` in `real_exp/scripts/run_table1.sh`.
> - If exact reproduction is not required, you may reduce the `SEEDS` list in `real_exp/scripts/run_table1.sh`.
> - Other RobustBench teachers can be used by modifying the `TEACHERS` list in `real_exp/scripts/run_table1.sh`.
> - To enable Weights & Biases logging, set `NOWAND=0` and configure the wandb options in `real_exp/scripts/run_table1.sh`.
> - The training script also supports other AD methods: `ard` (default), `rslad`, `iad`, `adaad`, and `igdm`.
> - CIFAR-100 and Tiny-ImageNet are also supported through `DATASET=cifar100` and `DATASET=tinyimg`. For Tiny-ImageNet, only `res18` is supported in this codebase. RobustBench is not used for Tiny-ImageNet; please download the Tiny-ImageNet teacher checkpoint and place it under `models/` as described in `utils.py`.


## Identify train unlearnset

After training the reference models, run:

```bash
bash real_exp/scripts/identify_train_unlearnset.sh
```

Use the same `DATASET` value used during training:

```bash
DATASET=cifar100 bash real_exp/scripts/identify_train_unlearnset.sh res18
DATASET=tinyimg bash real_exp/scripts/identify_train_unlearnset.sh res18
```

Other architectures:

```bash
bash real_exp/scripts/identify_train_unlearnset.sh mnv2
bash real_exp/scripts/identify_train_unlearnset.sh wrn28
bash real_exp/scripts/identify_train_unlearnset.sh wrn34
```

Outputs are saved to:

```text
analysis_results/train_unlearnset/{dataset}/eps{eps}/{student}/
```

Main outputs:

```text
train_unlearnset.json
summary.json
learnable_indices.npy
unlearnable_indices.npy
```
