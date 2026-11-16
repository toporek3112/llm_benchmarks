#!/usr/bin/env bash
set -euo pipefail

# Execute from the llm_benchmarks repository root.
export HF_HOME="$HOME/.cache/hf-mmlu"

# Same concurrency settings and repetition count as MMLU-Pro.
# Interleaved execution reduces potential time-dependent confounding.
schedule=(
  "8:1" "32:1" "64:1" "128:1" "256:1"
  "256:2" "128:2" "64:2" "32:2" "8:2"
  "64:3" "256:3" "8:3" "128:3" "32:3"
)

for entry in "${schedule[@]}"; do
  IFS=: read -r concurrency repetition <<<"$entry"

  # Preserve the original results and logs for each run.
  output="capability/results/ifeval-repro/c${concurrency}/run-${repetition}"

  if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 1
  fi

  mkdir -p "$output"

  echo "Starting concurrency=${concurrency}, repetition=${repetition}"

  start=$(date +%s)

  # Use the same pinned model and evaluation configuration.
  lm-eval run \
    --model local-completions \
    --model_args "model=google/gemma-2b,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=google/gemma-2b,revision=9cf48e52b224239de00d483ec8eb84fb8d0f3a3a,tokenizer_backend=huggingface,tokenized_requests=False,max_length=4096,num_concurrent=${concurrency},seed=1234" \
    --include_path capability/tasks \
    --tasks ifeval_pinned \
    --batch_size 1 \
    --output_path "$output" \
    --log_samples \
    2>&1 | tee "$output/run.log"

  end=$(date +%s)

  # Record wall-clock runtime in seconds.
  echo "$((end - start))" >"$output/time.txt"

  echo "Completed concurrency=${concurrency}, repetition=${repetition}"
done
