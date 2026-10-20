#!/usr/bin/env bash
set -euo pipefail
cd /home/eter/桌面/workspace/a3_dual_arm_sim

run=outputs/cookie_multi_left8_40k_v1
dataset=datasets/cookie_multi_left8_v1
if [[ -d "$run/model" ]]; then
  echo "Refusing to overwrite existing training output: $run/model" >&2
  exit 1
fi
mkdir -p "$run"
printf '%s\n' "$$" > "$run/pid"
printf '%s\n' running > "$run/status"
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "%s\n" completed > "$run/status"; else printf "failed exit=%s\n" "$code" > "$run/status"; fi' EXIT

export A3_SMOLVLA_SDPA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

a3-sim train-smolvla \
  --root "$dataset" --repo-id local/cookie-multi-left8-v1 --offline \
  --output "$run/model" --steps 40000 --batch-size 32 \
  --num-workers 2 --save-freq 2000 --lr 0.00005 --seed 1000 --device cuda \
  > "$run/train.log" 2>&1
