#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/workspace/feedbax_runs/40e1911_delayed_timing_hold}"
PYTHONPATH="${PYTHONPATH:-src}"
MAX_PARALLEL_ROWS="${MAX_PARALLEL_ROWS:-6}"
ROW_LAUNCH_STAGGER_SECONDS="${ROW_LAUNCH_STAGGER_SECONDS:-10}"
export PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

COMMON_FLAGS=(
  --full-train
  --seed 42
  --n-train-batches 12000
  --batch-size 64
  --controller-lr 0.003
  --lr-warmup-batches 500
  --lr-cosine-alpha 0.1
  --n-replicates 5
  --hidden-size 180
  --gradient-clip-norm 5
  --checkpoint-interval-batches 500
  --target-relative-multitarget
  --delayed-reach
  --delayed-reach-p-catch-trial 0.5
  --force-filter-feedback
  --loss-objective full_analytical_qrf
  --perturbation-training
  --perturbation-calibrated-timing
  --perturbation-movement-age-timing
  --perturbation-physical-level small
  --perturbation-nominal-fraction 0.45
  --perturbation-single-fraction 0.45
  --perturbation-combined-fraction 0.10
  --disable-progress
)

run_row() {
  local issue=$1
  local row=$2
  local go_min=$3
  local go_max=$4
  local nn_output_pre_go=$5
  local force_filter_hold=$6
  local start_pos_hold=$7
  local zero_vel_hold=$8

  local row_root="$RUN_ROOT/$issue/$row"
  mkdir -p "$row_root"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$issue" "$row" "start" \
    >> "$RUN_ROOT/row_status.tsv"

  uv run --no-sync python scripts/train_cs_nominal_gru.py \
    --issue "$issue" \
    --output-dir "$row_root/artifacts" \
    --spec-dir "$row_root/spec" \
    "${COMMON_FLAGS[@]}" \
    --delayed-reach-go-cue-min-step "$go_min" \
    --delayed-reach-go-cue-max-step "$go_max" \
    --nn-output-pre-go "$nn_output_pre_go" \
    --delayed-pre-go-force-filter-hold "$force_filter_hold" \
    --delayed-pre-go-start-pos-hold "$start_pos_hold" \
    --delayed-pre-go-zero-vel-hold "$zero_vel_hold"

  printf '%s\t%s\t%s\t%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$issue" "$row" "done" \
    >> "$RUN_ROOT/row_status.tsv"
}

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/rows.tsv" <<'ROWS'
issue	row	go_min	go_max	nn_output_pre_go	force_filter_hold	start_pos_hold	zero_vel_hold
6c36536	baseline__delayed_repeat	10	30	100000	0	0	0
bf71d86	timing__fixed_go10	10	10	100000	0	0	0
bf71d86	timing__fixed_go20	20	20	100000	0	0	0
bf71d86	timing__go10_15	10	15	100000	0	0	0
ef9c882	hold__force_filter	10	30	0	100000	0	0
ef9c882	hold__start_pos_zero_vel	10	30	0	0	1000000	100000
ROWS

declare -a ROW_PIDS=()
launched_rows=0

wait_for_slot() {
  while [ "${#ROW_PIDS[@]}" -ge "$MAX_PARALLEL_ROWS" ]; do
    local next_pid="${ROW_PIDS[0]}"
    wait "$next_pid"
    ROW_PIDS=("${ROW_PIDS[@]:1}")
  done
}

while IFS=$'\t' read -r issue row go_min go_max nn_output_pre_go force_filter_hold start_pos_hold zero_vel_hold; do
    wait_for_slot
    mkdir -p "$RUN_ROOT/logs"
    run_row \
      "$issue" \
      "$row" \
      "$go_min" \
      "$go_max" \
      "$nn_output_pre_go" \
      "$force_filter_hold" \
      "$start_pos_hold" \
      "$zero_vel_hold" \
      > "$RUN_ROOT/logs/${issue}__${row}.log" \
      2>&1 &
    ROW_PIDS+=("$!")
    launched_rows=$((launched_rows + 1))
    if [ "$ROW_LAUNCH_STAGGER_SECONDS" -gt 0 ] && [ "$launched_rows" -lt 6 ]; then
      sleep "$ROW_LAUNCH_STAGGER_SECONDS"
    fi
  done < <(tail -n +2 "$RUN_ROOT/rows.tsv")

for pid in "${ROW_PIDS[@]}"; do
  wait "$pid"
done
