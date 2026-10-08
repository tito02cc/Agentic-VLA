#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${ROBOMME_PLANNER_MODEL:-${HOME}/.cache/carve-vla/checkpoints/Qwen3-VL-4B-Instruct}"
ADAPTER_ROOT="${ROBOMME_PLANNER_ADAPTER:-${HOME}/.cache/carve-vla/checkpoints/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200}"
PORT="${ROBOMME_PLANNER_PORT:-18070}"
SWIFT="${ROBOMME_SWIFT:-/home/admin1/miniconda3/envs/g2agent/bin/swift}"
QUANT="${ROBOMME_PLANNER_QUANT:-none}"

required_model_files=(
  config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  tokenizer.json
)
for file in "${required_model_files[@]}"; do
  [[ -f "${MODEL_ROOT}/${file}" ]] || {
    echo "missing Qwen3-VL model file: ${MODEL_ROOT}/${file}" >&2
    exit 2
  }
done
for file in adapter_config.json adapter_model.safetensors; do
  [[ -f "${ADAPTER_ROOT}/${file}" ]] || {
    echo "missing GroundSG adapter file: ${ADAPTER_ROOT}/${file}" >&2
    exit 2
  }
done
[[ -x "${SWIFT}" ]] || {
  echo "swift executable is unavailable: ${SWIFT}" >&2
  exit 2
}

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-64}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

quant_args=()
case "${QUANT}" in
  none)
    ;;
  nf4)
    quant_args=(
      --quant-method bnb
      --quant-bits 4
      --bnb-4bit-quant-type nf4
      --bnb-4bit-use-double-quant true
    )
    ;;
  *)
    echo "unsupported ROBOMME_PLANNER_QUANT=${QUANT}; expected none or nf4" >&2
    exit 2
    ;;
esac

exec "${SWIFT}" deploy \
  --model "${MODEL_ROOT}" \
  --adapters "${ADAPTER_ROOT}" \
  --infer-backend transformers \
  --torch-dtype bfloat16 \
  --attn-impl sdpa \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name robomme-groundsg-qwen3vl4b \
  "${quant_args[@]}" \
  --log-level warning
