#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-${ROOT}/checkpoints/openvla-7b-finetuned-libero-10}"
CACHE="${CHECKPOINT}/.cache/huggingface/download"
BASE="https://hf-mirror.com/openvla/openvla-7b-finetuned-libero-10/resolve/main"

mkdir -p "${CACHE}"

PART1="${CACHE}/IO4xwqmZYzFmxznkwkiNSBwO1H0=.72f7196d8768c104073d7e9104ad8023cb27727191de1a490612886428b18c51.incomplete"
PART2="${CACHE}/t9msAuTjAZjuQnmzGOwTjiptvIU=.92a60a88c7dad2ba1fccc4acbebef98fc7a894891bc49f2e21cfbf1600daf413.incomplete"
PART3="${CACHE}/DaGOU-KRMVrY0aYktrsE34tL0Bs=.12b45ae85535565148907e379d6f35bcba83b9497006ff3b66658fa412ffac85.incomplete"
PART4="${CACHE}/-dFtyT7kcgbTHt1cy9JKqruJCR4=.d3b8e759db56b86709c56f8d156952415dd64e101feccfd01e701487041dd8c1.incomplete"

curl --parallel --parallel-immediate --parallel-max 4 \
  --location --fail --retry 20 --retry-all-errors --continue-at - \
  --output "${PART1}" "${BASE}/model-00001-of-00004.safetensors" \
  --next --location --fail --retry 20 --retry-all-errors --continue-at - \
  --output "${PART2}" "${BASE}/model-00002-of-00004.safetensors" \
  --next --location --fail --retry 20 --retry-all-errors --continue-at - \
  --output "${PART3}" "${BASE}/model-00003-of-00004.safetensors" \
  --next --location --fail --retry 20 --retry-all-errors --continue-at - \
  --output "${PART4}" "${BASE}/model-00004-of-00004.safetensors"

EXPECTED=(4925122448 4947392496 4947417456 262668432)
EXPECTED_SHA256=(
  72f7196d8768c104073d7e9104ad8023cb27727191de1a490612886428b18c51
  92a60a88c7dad2ba1fccc4acbebef98fc7a894891bc49f2e21cfbf1600daf413
  12b45ae85535565148907e379d6f35bcba83b9497006ff3b66658fa412ffac85
  d3b8e759db56b86709c56f8d156952415dd64e101feccfd01e701487041dd8c1
)
PARTS=("${PART1}" "${PART2}" "${PART3}" "${PART4}")
for index in 0 1 2 3; do
  actual="$(stat -c '%s' "${PARTS[${index}]}")"
  if [[ "${actual}" != "${EXPECTED[${index}]}" ]]; then
    echo "Shard $((index + 1)) has ${actual} bytes; expected ${EXPECTED[${index}]}" >&2
    exit 1
  fi
  actual_sha256="$(sha256sum "${PARTS[${index}]}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256[${index}]}" ]]; then
    echo "Shard $((index + 1)) SHA-256 mismatch" >&2
    exit 1
  fi
  target="$(printf '%s/model-%05d-of-00004.safetensors' "${CHECKPOINT}" "$((index + 1))")"
  mv "${PARTS[${index}]}" "${target}"
done

echo "OpenVLA LIBERO-10 checkpoint shards downloaded and size-verified in ${CHECKPOINT}"
