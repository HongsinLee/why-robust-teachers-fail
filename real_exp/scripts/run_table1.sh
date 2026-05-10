#!/bin/bash
set -e

ARCH=${1:-res18}
DATASET=${DATASET:-cifar10}
SESSION=table1_${DATASET}_${ARCH}

tmux has-session -t ${SESSION} 2>/dev/null && tmux kill-session -t ${SESSION}

EPS=${EPS:-8}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-128}
LR=${LR:-0.1}
WD=${WD:-2e-4}

# Set to 1 to disable Weights & Biases logging.
# Set to 0 to enable wandb, and configure wandb options below.
NOWAND=0
WANDB_ENTITY="your_wandb_entity"
WANDB_PROJECT="robust-teachers-fail"
WANDB_NAME_PREFIX="table1"
WANDB_TAG_PREFIX="table1"

# GPU IDs to use. Add more IDs if more GPUs are available.
# e.g., GPUS=(0 1 2 3 4 5 6 7)
GPUS=(0 1 2 3)

if [[ ${ARCH} == "res18" ]]; then
  STUDENT="RES-18"; DEPTH=0; WIDEN=0
elif [[ ${ARCH} == "mnv2" ]]; then
  STUDENT="MN-V2"; DEPTH=0; WIDEN=0
elif [[ ${ARCH} == "wrn28" ]]; then
  STUDENT="WRN"; DEPTH=28; WIDEN=10
elif [[ ${ARCH} == "wrn34" ]]; then
  STUDENT="WRN"; DEPTH=34; WIDEN=10
else
  echo "Unknown architecture: ${ARCH}"
  echo "Use one of: res18, mnv2, wrn28, wrn34"
  exit 1
fi

if [[ ${DATASET} == "tinyimg" && ${STUDENT} != "RES-18" ]]; then
  echo "Only res18 is supported for Tiny-ImageNet."
  exit 1
fi

SEEDS=(0 1 2 3 4 5 6 7 8 9)

if [[ ${DATASET} == "cifar10" ]]; then
  TEACHERS=(
    "Chen2021LTD_WRN34_10"
    "Rebuffi2021Fixing_70_16_cutmix_extra"
    "Bartoldson2024Adversarial_WRN-94-16"
    "Gowal2021Improving_28_10_ddpm_100m"
  )
elif [[ ${DATASET} == "cifar100" ]]; then
  TEACHERS=(
    "Chen2021LTD_WRN34_10"
    "Wang2023Better_WRN-28-10"
    "Wang2023Better_WRN-70-16"
    "Gowal2020Uncovering_extra"
  )
elif [[ ${DATASET} == "tinyimg" ]]; then
  TEACHERS=(
    "tiny_linf_wrn28-10"
  )
else
  echo "Unknown dataset: ${DATASET}"
  echo "Use one of: cifar10, cifar100, tinyimg"
  exit 1
fi

TMP_DIR=$(mktemp -d)
JOB_FILE="${TMP_DIR}/jobs.txt"
LOCK_FILE="${TMP_DIR}/lock"

for SEED in "${SEEDS[@]}"; do
  echo "--method pgd --seed ${SEED}" >> ${JOB_FILE}
  echo "--method trades --seed ${SEED}" >> ${JOB_FILE}
  for TEACHER in "${TEACHERS[@]}"; do
    echo "--method ard --teacher ${TEACHER} --seed ${SEED}" >> ${JOB_FILE}
  done
done

cat > "${TMP_DIR}/worker.sh" <<EOF
#!/bin/bash
GPU=\$1
JOB_FILE="$JOB_FILE"
LOCK_FILE="$LOCK_FILE"

while true; do
  CMD=""

  {
    flock -x 200
    if [[ -s \${JOB_FILE} ]]; then
      CMD=\$(head -n 1 \${JOB_FILE})
      sed -i '1d' \${JOB_FILE}
    fi
  } 200>\${LOCK_FILE}

  if [[ -z "\${CMD}" ]]; then
    echo "[GPU \${GPU}] no more jobs"
    break
  fi

  METHOD=\$(echo "\${CMD}" | sed -n 's/.*--method \([^ ]*\).*/\1/p')
  SEED_NUM=\$(echo "\${CMD}" | sed -n 's/.*--seed \([^ ]*\).*/\1/p')
  TEACHER_NAME=\$(echo "\${CMD}" | sed -n 's/.*--teacher \([^ ]*\).*/\1/p')

  if [[ -z "\${TEACHER_NAME}" ]]; then
    RUN_NAME="${WANDB_NAME_PREFIX}_${DATASET}_${ARCH}_\${METHOD}_seed\${SEED_NUM}"
    RUN_TAGS="${WANDB_TAG_PREFIX},${DATASET},${ARCH},\${METHOD},seed\${SEED_NUM}"
  else
    if [[ "\${TEACHER_NAME}" == "Chen2021LTD_WRN34_10" ]]; then
      TEACHER_SHORT="chen"
    elif [[ "\${TEACHER_NAME}" == "Rebuffi2021Fixing_70_16_cutmix_extra" ]]; then
      TEACHER_SHORT="rebuffi"
    elif [[ "\${TEACHER_NAME}" == "Bartoldson2024Adversarial_WRN-94-16" ]]; then
      TEACHER_SHORT="bartoldson"
    elif [[ "\${TEACHER_NAME}" == "Gowal2021Improving_28_10_ddpm_100m" ]]; then
      TEACHER_SHORT="gowal"
    elif [[ "\${TEACHER_NAME}" == "Wang2023Better_WRN-28-10" ]]; then
      TEACHER_SHORT="wang28"
    elif [[ "\${TEACHER_NAME}" == "Wang2023Better_WRN-70-16" ]]; then
      TEACHER_SHORT="wang70"
    elif [[ "\${TEACHER_NAME}" == "Gowal2020Uncovering_extra" ]]; then
      TEACHER_SHORT="gowal"
    elif [[ "\${TEACHER_NAME}" == "tiny_linf_wrn28-10" ]]; then
      TEACHER_SHORT="wang"
    else
      TEACHER_SHORT="\${TEACHER_NAME}"
    fi

    RUN_NAME="${WANDB_NAME_PREFIX}_${DATASET}_${ARCH}_\${METHOD}_\${TEACHER_SHORT}_seed\${SEED_NUM}"
    RUN_TAGS="${WANDB_TAG_PREFIX},${DATASET},${ARCH},\${METHOD},\${TEACHER_SHORT},seed\${SEED_NUM}"
  fi

  echo "[GPU \${GPU}] \${CMD}"
  echo "[GPU \${GPU}] wandb_name=\${RUN_NAME}"

  CUDA_VISIBLE_DEVICES=\${GPU} python real_exp/train.py \
    --dataset $DATASET \
    --student $STUDENT \
    --depth $DEPTH \
    --widen_factor $WIDEN \
    --eps $EPS \
    --epochs $EPOCHS \
    --batch $BATCH \
    --lr $LR \
    --wd $WD \
    --nowand $NOWAND \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "\${RUN_NAME}" \
    --wandb_tags "\${RUN_TAGS}" \
    \${CMD}
done
EOF

chmod +x "${TMP_DIR}/worker.sh"

tmux new-session -d -s ${SESSION} "${TMP_DIR}/worker.sh ${GPUS[0]}"

for ((i=1; i<${#GPUS[@]}; i++)); do
  tmux split-window -t ${SESSION} "${TMP_DIR}/worker.sh ${GPUS[$i]}"
  tmux select-layout -t ${SESSION} tiled
done

tmux attach -t ${SESSION}
