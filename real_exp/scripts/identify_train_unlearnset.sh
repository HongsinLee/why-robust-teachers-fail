#!/bin/bash
set -e

ARCH=${1:-res18}
DATASET=${DATASET:-cifar10}
SESSION=unlearnset_${DATASET}_${ARCH}

tmux has-session -t ${SESSION} 2>/dev/null && tmux kill-session -t ${SESSION}

EPS=${EPS:-8}
BATCH=${BATCH:-128}
EVAL_PGD_STEPS=${EVAL_PGD_STEPS:-10}

# GPU IDs to use. Add more IDs if more GPUs are available.
# e.g., GPUS=(0 1 2 3 4 5 6 7)
GPUS=(0 1 2 3)
NUM_WORKERS=${#GPUS[@]}

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

if [[ ${DATASET} == "cifar10" ]]; then
  SPLIT_GROUPS=(pgd_at trades chen rebuffi bartoldson gowal)
elif [[ ${DATASET} == "cifar100" ]]; then
  SPLIT_GROUPS=(pgd_at trades chen wang28 wang70 gowal)
elif [[ ${DATASET} == "tinyimg" ]]; then
  SPLIT_GROUPS=(pgd_at trades wang)
else
  echo "Unknown dataset: ${DATASET}"
  echo "Use one of: cifar10, cifar100, tinyimg"
  exit 1
fi

SEEDS=(0 1 2 3 4 5 6 7 8 9)

TMP_DIR=$(mktemp -d)
JOB_FILE="${TMP_DIR}/jobs.txt"
LOCK_FILE="${TMP_DIR}/lock"
DONE_DIR="${TMP_DIR}/done"
mkdir -p ${DONE_DIR}

for GROUP_NAME in "${SPLIT_GROUPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "--split_group ${GROUP_NAME} --split_seed ${SEED}" >> ${JOB_FILE}
  done
done

cat > "${TMP_DIR}/worker.sh" <<EOF
#!/bin/bash
GPU=\$1
JOB_FILE="$JOB_FILE"
LOCK_FILE="$LOCK_FILE"
DONE_DIR="$DONE_DIR"

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

  echo "[GPU \${GPU}] \${CMD}"

  CUDA_VISIBLE_DEVICES=\${GPU} python real_exp/identify_train_unlearnset.py \
    --split_mode eval \
    --dataset $DATASET \
    --student $STUDENT \
    --depth $DEPTH \
    --widen_factor $WIDEN \
    --eps $EPS \
    --batch $BATCH \
    --eval_pgd_steps $EVAL_PGD_STEPS \
    \${CMD}
done

touch \${DONE_DIR}/gpu_\${GPU}.done
echo "[GPU \${GPU}] worker finished"
read -p "Press Enter to close this pane..."
EOF

chmod +x "${TMP_DIR}/worker.sh"

tmux new-session -d -s ${SESSION} -n eval "${TMP_DIR}/worker.sh ${GPUS[0]}"

for ((i=1; i<${#GPUS[@]}; i++)); do
  tmux split-window -t ${SESSION}:eval "${TMP_DIR}/worker.sh ${GPUS[$i]}"
  tmux select-layout -t ${SESSION}:eval tiled
done

tmux new-window -t ${SESSION} -n merge "bash -c '
echo \"Waiting for ${NUM_WORKERS} eval workers to finish...\"

while true; do
  DONE_COUNT=\$(ls ${DONE_DIR}/*.done 2>/dev/null | wc -l)
  echo \"Done workers: \${DONE_COUNT}/${NUM_WORKERS}\"

  if [[ \${DONE_COUNT} -ge ${NUM_WORKERS} ]]; then
    break
  fi

  sleep 30
done

echo \"All workers finished. Merging results...\"

python real_exp/identify_train_unlearnset.py \
  --split_mode merge \
  --dataset ${DATASET} \
  --student ${STUDENT} \
  --depth ${DEPTH} \
  --widen_factor ${WIDEN} \
  --eps ${EPS} \
  --batch ${BATCH} \
  --eval_pgd_steps ${EVAL_PGD_STEPS}

rm -rf ${TMP_DIR}
echo \"Done.\"
read -p \"Press Enter to close this pane...\"
'"

tmux select-window -t ${SESSION}:eval
tmux attach -t ${SESSION}
