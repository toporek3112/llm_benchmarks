# Pinned MMLU-Pro task bundle

## Provenance

- Source: `EleutherAI/lm-evaluation-harness/lm_eval/tasks/mmlu_pro/`.
- Source commit: `d6de81643928d653435c431bae19945d41d32520`.
- Dataset: `TIGER-Lab/MMLU-Pro`.
- Dataset revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
- Both revisions are confirmed in the previous Gemma-2B evaluation notebook.
- Changes to upstream: added dataset revision; renamed group and task identifiers with `_pinned` to avoid built-in task collisions. The subject descriptions, question formatting, per-subject filtering, few-shot selection, generation kwargs, answer extraction, scoring and group weighted aggregation are otherwise unchanged.
- Original methodology: 14 subject tasks, test set 12032 questions, 5-shot CoT, `max_gen_toks: 2048`, `temperature: 0.0`, `do_sample: false`, stop sequence `Question:`; regex extracts first `answer is (X)` occurrence for X in A-J.

## Installation

Extract this bundle into `capability/tasks/` from the repository root, such that
`capability/tasks/mmlu_pro_pinned/_mmlu_pro_pinned.yaml` exists.

## Validate (on the benchmark VM)

```bash
lm-eval validate --tasks mmlu_pro_pinned --include_path capability/tasks
```

Verify the task registration, version and local code within the pinned
harness checkout. Do not infer that matching configurations mean the
original model outputs must match bit-for-bit.

## Run with the generic wrapper
First run vllm
```bash
 CUDA_VISIBLE_DEVICES=0 HF_HOME="$HOME/.cache/hf-mmlu" vllm serve google/gemma-2b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --seed 0 \
  --generation-config vllm \
  --revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a \
  --tokenizer-revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a \
  --trust-remote-code
```

and then the benchmark itself
```bash
HF_HOME="$HOME/.cache/hf-mmlu" ./run_benchmark.sh \
  --tasks mmlu_pro_pinned \
  --model google/gemma-2b \
  --tokenizer google/gemma-2b \
  --revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a \
  --output-dir capability/results/mmlu_pro/gemma-2b-rerun \
  --concurrencies 8,32,64,128,256 \
  --repetitions 5 \
  --grace-seconds 120 \
  --overwrite \
  -- \
  --include_path capability/tasks \
  --num_fewshot 5 \
  --batch_size 1 \
  --log_samples
```

This assumes vLLM serves the original Gemma-2B model snapshot with the
original vLLM launch configuration. `--revision` in the wrapper pins the
client's tokenizer, not vLLM's model weights.

The result task names are `mmlu_pro_pinned_<subject>` rather than the
built-in `mmlu_pro_<subject>`. Preserve historical outputs.

The bundle pins source logic and dataset revision, but fully reproducible
execution also requires a pinned Harness checkout, Python dependencies,
model/server revision, and environment.
