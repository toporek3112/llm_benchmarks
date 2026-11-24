#!/usr/bin/env bash
set -euo pipefail

# Run pinned lm-evaluation-harness tasks against an already-running vLLM server.
# Invoke from the llm_benchmarks repository root.
# Options after '--' are forwarded unchanged to lm-eval.

usage() {
  cat <<'USAGE'
Usage:
  run_benchmark.sh --tasks TASK --model MODEL --output-dir DIR [options] -- [lm-eval options]

Required:
  --tasks TASK                 e.g. mmlu_pro, ifeval_pinned, minerva_math500
  --model MODEL                exact name returned by vLLM /v1/models
  --output-dir DIR             dedicated directory per benchmark and model

Model/API:
  --tokenizer ID               default: --model
  --revision SHA               tokenizer revision used by lm-eval (NOT vLLM weights)
  --tokenizer-revision SHA     alias for --revision
  --base-url URL               default: http://127.0.0.1:8000/v1/completions
  --max-length N               default: 4096
  --client-seed N              default: 1234
  --tokenizer-backend NAME     default: huggingface
  --tokenized-requests BOOL    default: False
  --model-args KEY=VAL,...     additional lm-eval model arguments

Runs:
  --concurrencies 8,32,64,128,256  default: 8,32,64,128,256
  --repetitions N              number of runs per concurrency (default: 3)
  --start-run N                first run number for the generated grid (default: 1)
  --schedule '8:1 32:1 8:2'   explicit order and run numbers (instead of grid)
  --grace-seconds N            wait between executed runs (default: 0)
  --skip-completed             skip runs with matching command and valid result
  --overwrite                 replace selected existing run(s); archive old files
  --dry-run                   display commands without executing or writing files

Everything after '--' is passed to lm-eval, e.g.:
  -- --include_path capability/tasks --num_fewshot 5 --batch_size 1 --log_samples

Outputs:
  DIR/c<concurrency>/run-<number>/    command.txt, run.log, date.txt, time.txt,
                                     .completed and lm-eval outputs
  DIR/run_windows.tsv                append-only per-run timestamps for Prometheus
  DIR/.replaced/c<concurrency>/...   archived runs replaced via --overwrite

The timestamps in run_windows.tsv use UTC ISO-8601 and Unix epoch seconds
(with millisecond precision). Start/end delimit the complete lm-eval CLI process,
including setup and result writing, but EXCLUDE the inter-run grace period.
Do not query the Prometheus metrics for a run until its status is 'completed'.

IMPORTANT: vLLM is started separately. --revision pins only the local tokenizer.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

# The pinned harness writes results_<timestamp>.json under a model directory.
has_result_file() {
  [[ -n $(find "$1" -type f -name 'results_*.json' -print -quit) ]]
}

# Write UTC and Unix timestamps derived from the SAME clock reading.
# Returns e.g. 2026-11-23T14:05:04.123Z and 1795442704.123.
timestamp_from_ns() {
  local ns=$1
  local epoch_s=$((ns / 1000000000))
  local millis=$(((ns % 1000000000) / 1000000))
  printf -v timestamp_epoch_s '%s.%03d' "$epoch_s" "$millis"
  timestamp_utc=$(date -u -d "@$timestamp_epoch_s" +'%Y-%m-%dT%H:%M:%S.%3NZ')
}

# Appends one finished attempt; old records are retained after --overwrite.
append_time_index() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tasks" "$model" "$concurrency" "$repetition" "$status" "$exit_code" \
    "$started_utc" "$finished_utc" "$start_epoch_s" "$end_epoch_s" \
    "$elapsed_seconds" "$output" >> "$time_index"
}

tasks='' model='' tokenizer='' revision='' output_dir=''
base_url='http://127.0.0.1:8000/v1/completions'
max_length=4096 client_seed=1234
tokenizer_backend='huggingface' tokenized_requests='False' extra_model_args=''
concurrencies='8,32,64,128,256' repetitions=3 start_run=1
schedule_spec='' schedule_supplied=false grid_supplied=false
grace_seconds=0 skip_completed=false overwrite=false dry_run=false
harness_args=()

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift; continue ;;
    --skip-completed) skip_completed=true; shift; continue ;;
    --overwrite) overwrite=true; shift; continue ;;
    --) shift; harness_args=("$@"); break ;;
  esac

  (( $# >= 2 )) || fail "Missing value for $1"
  option=$1
  value=$2
  shift 2

  case "$option" in
    --tasks) tasks=$value ;;
    --model) model=$value ;;
    --tokenizer) tokenizer=$value ;;
    --revision|--tokenizer-revision) revision=$value ;;
    --base-url) base_url=$value ;;
    --max-length) max_length=$value ;;
    --client-seed) client_seed=$value ;;
    --tokenizer-backend) tokenizer_backend=$value ;;
    --tokenized-requests) tokenized_requests=$value ;;
    --model-args) extra_model_args=$value ;;
    --concurrencies) concurrencies=$value; grid_supplied=true ;;
    --repetitions) repetitions=$value; grid_supplied=true ;;
    --start-run) start_run=$value; grid_supplied=true ;;
    --schedule) schedule_spec=$value; schedule_supplied=true ;;
    --grace-seconds) grace_seconds=$value ;;
    --output-dir) output_dir=$value ;;
    *) fail "Unknown option: $option (put native lm-eval options after --)" ;;
  esac
done

[[ -n $tasks && -n $model && -n $output_dir ]] || fail '--tasks, --model and --output-dir are required'
tokenizer=${tokenizer:-$model}

[[ $base_url == http://* || $base_url == https://* ]] || fail '--base-url must be an unescaped HTTP(S) URL'
[[ $max_length =~ ^[1-9][0-9]*$ ]] || fail '--max-length must be positive'
[[ $client_seed =~ ^[0-9]+$ ]] || fail '--client-seed must be non-negative'
[[ $tokenized_requests == True || $tokenized_requests == False ]] || fail '--tokenized-requests must be True or False'
[[ $grace_seconds =~ ^(0|[1-9][0-9]*)$ ]] || fail '--grace-seconds must be a non-negative integer'
[[ $start_run =~ ^[1-9][0-9]*$ ]] || fail '--start-run must be positive'

for value in "$tasks" "$model" "$tokenizer" "$revision" "$base_url" "$tokenizer_backend"; do
  [[ $value != *','* ]] || fail 'Comma in model argument is unsupported'
  [[ $value != *$'\n'* && $value != *$'\t'* ]] || fail 'Newlines and tabs in parameters are unsupported'
done
[[ $output_dir != *$'\n'* && $output_dir != *$'\t'* ]] || fail 'Newlines and tabs in output directory are unsupported' 

if [[ -n $extra_model_args ]] && [[ ,$extra_model_args =~ ,(model|base_url|tokenizer|revision|tokenizer_backend|tokenized_requests|max_length|num_concurrent|seed)= ]]; then
  fail '--model-args must not override a dedicated flag'
fi
for option in "${harness_args[@]}"; do
  case "$option" in
    --model|--model=*|--model_args|--model_args=*|--tasks|--tasks=*|--output_path|--output_path=*)
      fail "Option $option is managed by the wrapper; do not repeat after --" ;;
  esac
done
if "$skip_completed" && "$overwrite"; then
  fail '--skip-completed and --overwrite are mutually exclusive'
fi

schedule=()
if "$schedule_supplied"; then
  ! "$grid_supplied" || fail '--schedule cannot be combined with --concurrencies, --repetitions or --start-run'
  read -r -a schedule <<< "${schedule_spec//,/ }"
else
  [[ $repetitions =~ ^[1-9][0-9]*$ ]] || fail '--repetitions must be positive'
  read -r -a concurrency_values <<< "${concurrencies//,/ }"
  for concurrency in "${concurrency_values[@]}"; do
    [[ $concurrency =~ ^[1-9][0-9]*$ ]] || fail "Invalid concurrency: $concurrency"
    for ((repetition=start_run; repetition<start_run+repetitions; repetition++)); do
      schedule+=("${concurrency}:${repetition}")
    done
  done
fi

(( ${#schedule[@]} > 0 )) || fail 'Run schedule cannot be empty'
declare -A seen_entries=()
for entry in "${schedule[@]}"; do
  [[ $entry =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]] || fail "Invalid schedule entry: $entry"
  [[ ! -v seen_entries[$entry] ]] || fail "Duplicate run in schedule: $entry"
  seen_entries[$entry]=1
done

model_args="model=${model},base_url=${base_url},tokenizer=${tokenizer}"
[[ -z $revision ]] || model_args+=",revision=${revision}"
model_args+=",tokenizer_backend=${tokenizer_backend},tokenized_requests=${tokenized_requests},max_length=${max_length}"

if ! "$dry_run"; then
  command -v lm-eval >/dev/null || fail 'lm-eval is not on PATH; activate the evaluation environment'
fi

index_header=$'tasks\tmodel\tconcurrency\trepetition\tstatus\texit_code\tstarted_utc\tfinished_utc\tstart_epoch_s\tend_epoch_s\telapsed_seconds\toutput_dir'
time_index="$output_dir/run_windows.tsv"
if ! "$dry_run"; then
  mkdir -p -- "$output_dir"
  if [[ -L $time_index ]]; then
    fail "Refusing symlinked time index: $time_index"
  fi
  if [[ -e $time_index ]]; then
    [[ -f $time_index ]] || fail "Not a regular time index: $time_index"
    [[ $(head -n 1 -- "$time_index") == "$index_header" ]] || fail "Unexpected index header: $time_index"
  else
    printf '%s\n' "$index_header" > "$time_index"
  fi
fi

previous_executed=false
for entry in "${schedule[@]}"; do
  IFS=: read -r concurrency repetition <<< "$entry"
  output="${output_dir}/c${concurrency}/run-${repetition}"
  run_model_args="${model_args},num_concurrent=${concurrency},seed=${client_seed}"
  [[ -z $extra_model_args ]] || run_model_args+=",${extra_model_args}"

  command=(
    lm-eval run
    --model local-completions
    --model_args "$run_model_args"
    --tasks "$tasks"
    --output_path "$output"
    "${harness_args[@]}"
  )
  printf -v printable_command '%q ' "${command[@]}"
  printable_command=${printable_command% }

  if "$dry_run"; then
    printf '%s\n' "$printable_command"
    continue
  fi

  if [[ -e $output || -L $output ]]; then
    if "$skip_completed" && [[ -d $output && ! -L $output && -f $output/.completed && -f $output/command.txt ]] \
      && [[ $(< "$output/command.txt") == "$printable_command" ]] \
      && has_result_file "$output"; then
      printf 'SKIP completed concurrency=%s run=%s: %s\n' "$concurrency" "$repetition" "$output"
      continue
    fi
    if ! "$overwrite"; then
      fail "Output exists: $output (use --skip-completed, --start-run or --overwrite)"
    fi
    [[ -d $output && ! -L $output ]] || fail "Cannot replace non-directory or symlink: $output"
    archive_root="${output_dir}/.replaced/c${concurrency}"
    mkdir -p -- "$archive_root"
    archive_path="${archive_root}/run-${repetition}_$(date -u +%Y%m%dT%H%M%S%NZ)"
    [[ ! -e $archive_path && ! -L $archive_path ]] || fail "Archive collision: $archive_path"
    mv -- "$output" "$archive_path"
    printf 'ARCHIVED previous run: %s\n' "$archive_path"
  fi

  # Only pause before a subsequent EXECUTED run; never before the first
  # or after the last, and skipped runs do not reset the waiting period.
  if "$previous_executed" && (( grace_seconds > 0 )); then
    printf 'GRACE waiting %ss before concurrency=%s repetition=%s\n' \
      "$grace_seconds" "$concurrency" "$repetition"
    sleep "$grace_seconds"
  fi

  mkdir -p -- "${output%/*}"
  mkdir -- "$output"
  printf '%s\n' "$printable_command" > "$output/command.txt"

  start_ns=$(date +%s%N)
  timestamp_from_ns "$start_ns"
  started_utc=$timestamp_utc
  start_epoch_s=$timestamp_epoch_s
  printf 'started_utc=%s\nstart_epoch_s=%s\n' "$started_utc" "$start_epoch_s" > "$output/date.txt"
  printf 'START concurrency=%s repetition=%s UTC=%s\n' "$concurrency" "$repetition" "$started_utc"

  # pipefail propagates a harness error even though output is piped to tee.
  if "${command[@]}" 2>&1 | tee -- "$output/run.log"; then
    exit_code=0
  else
    exit_code=$?
  fi

  # Exit code 0 without a saved result is not a successful benchmark.
  if (( exit_code == 0 )) && ! has_result_file "$output"; then
    printf 'ERROR: lm-eval exited 0 but no results_*.json was saved: %s\n' "$output" \
      | tee -a "$output/run.log" >&2
    exit_code=1
  fi

  end_ns=$(date +%s%N)
  timestamp_from_ns "$end_ns"
  finished_utc=$timestamp_utc
  end_epoch_s=$timestamp_epoch_s
  elapsed_ns=$((end_ns - start_ns))
  printf -v elapsed_seconds '%d.%03d' \
    "$((elapsed_ns / 1000000000))" "$(((elapsed_ns % 1000000000) / 1000000))"
  status=completed
  (( exit_code == 0 )) || status=failed

  {
    printf 'finished_utc=%s\n' "$finished_utc"
    printf 'end_epoch_s=%s\n' "$end_epoch_s"
    printf 'elapsed_seconds=%s\n' "$elapsed_seconds"
    printf 'status=%s\n' "$status"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'grace_seconds=%s\n' "$grace_seconds"
  } >> "$output/date.txt"
  printf '%s\n' "$elapsed_seconds" > "$output/time.txt"
  append_time_index
  previous_executed=true

  if (( exit_code != 0 )); then
    printf 'FAILED concurrency=%s repetition=%s exit_code=%s (output retained: %s)\n' \
      "$concurrency" "$repetition" "$exit_code" "$output" >&2
    exit "$exit_code"
  fi

  printf '%s\n' "$finished_utc" > "$output/.completed"
  printf 'COMPLETED concurrency=%s repetition=%s elapsed=%ss\n' \
    "$concurrency" "$repetition" "$elapsed_seconds"
done
