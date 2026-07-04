#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data/nuscenes"
ARCHIVE="${DATA_ROOT}/v1.0-mini.tgz"
DOWNLOAD_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

mkdir -p "${DATA_ROOT}"

if [[ -f "${ARCHIVE}" ]]; then
    echo "Archive already exists; skipping download: ${ARCHIVE}"
else
    echo "Downloading nuScenes mini..."
    curl --fail --location --output "${ARCHIVE}.part" "${DOWNLOAD_URL}"
    mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

echo "Extracting nuScenes mini..."
tar -xzf "${ARCHIVE}" -C "${DATA_ROOT}"

required_directories=(
    "${DATA_ROOT}/v1.0-mini"
    "${DATA_ROOT}/samples"
    "${DATA_ROOT}/sweeps"
    "${DATA_ROOT}/maps"
)

missing_directories=()
for directory in "${required_directories[@]}"; do
    if [[ ! -d "${directory}" ]]; then
        missing_directories+=("${directory}")
    fi
done

if (( ${#missing_directories[@]} > 0 )); then
    echo "Error: nuScenes mini extraction is incomplete. Missing directories:" >&2
    printf '  - %s\n' "${missing_directories[@]}" >&2
    exit 1
fi

echo "nuScenes mini dataset is ready."
