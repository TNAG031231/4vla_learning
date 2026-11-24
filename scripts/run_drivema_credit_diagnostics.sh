#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/run_drivema_credit_diagnostics.sh ORACLE_ARCHIVE RESULT_DIR" >&2
    exit 2
fi
oracle_archive=$1
result_dir=$2
: "${VLA_DERIVED_ROOT:?set VLA_DERIVED_ROOT}" "${NUSCENES_ROOT:?set NUSCENES_ROOT}"

run_dir=$(mktemp -d "$VLA_DERIVED_ROOT/drivema/action_credit_diagnostics_XXXXXXXX")
run_name=${run_dir##*/}
script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run_dir/oracle" "$result_dir"
tar -xzf "$oracle_archive" -C "$run_dir/oracle" metadata.json pairs.jsonl summary.json
echo "Diagnostic outputs: $run_dir"

python -u "$script_dir/run_drivema_action_crossover.py" evaluate \
    --oracle-dir "$run_dir/oracle" --output-dir "$run_dir/crossover"
tar -czf "$result_dir/${run_name}_crossover.tar.gz" -C "$run_dir" crossover
echo "Collected: $result_dir/${run_name}_crossover.tar.gz"

python -u "$script_dir/run_drivema_longitudinal_branches.py" run \
    --output-relative-dir "drivema/$run_name/longitudinal"
tar -czf "$result_dir/${run_name}_longitudinal.tar.gz" -C "$run_dir" longitudinal
echo "Collected: $result_dir/${run_name}_longitudinal.tar.gz"
