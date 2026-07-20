#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venvs/openvla-carve/bin/python"
CHECKPOINT="${ROOT}/checkpoints/openvla-7b-finetuned-libero-10"
OUTPUT_DIR="${ROOT}/results/carve_optimize/openvla_4090_20260718"
BF16="${OUTPUT_DIR}/openvla_eager_bf16_reference.json"
INT8="${OUTPUT_DIR}/openvla_bnb_int8_candidate.json"
NF4="${OUTPUT_DIR}/openvla_bnb_nf4_candidate.json"

mkdir -p "${OUTPUT_DIR}"

COMMON=(
  "${ROOT}/scripts/benchmark_carve_openvla_profile.py"
  --checkpoint-dir "${CHECKPOINT}"
  --device cuda:0
  --warmup-calls 2
  --measured-calls 10
  --max-observations 10
  --deadline-ms 100
  --attn-implementation sdpa
  --record-latency-samples
)

"${PYTHON}" "${COMMON[@]}" \
  --precision bf16 \
  --output-manifest "${BF16}"
bf16_status=$?
if [[ "${bf16_status}" -ne 0 ]]; then
  echo "BF16 reference failed; low-bit comparisons are not valid." >&2
  exit "${bf16_status}"
fi

"${PYTHON}" "${COMMON[@]}" \
  --precision int8 \
  --reference-manifest "${BF16}" \
  --output-manifest "${INT8}"
int8_status=$?

"${PYTHON}" "${COMMON[@]}" \
  --precision nf4 \
  --reference-manifest "${BF16}" \
  --output-manifest "${NF4}"
nf4_status=$?

"${PYTHON}" "${ROOT}/scripts/summarize_openvla_profile_gate.py" \
  --input-dir "${OUTPUT_DIR}"
summary_status=$?

echo "OpenVLA profile gate complete: BF16=${bf16_status}, INT8=${int8_status}, NF4=${nf4_status}, summary=${summary_status}"
exit 0
