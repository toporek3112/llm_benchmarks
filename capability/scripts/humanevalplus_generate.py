"""Generate HumanEval+ solutions through an OpenAI-compatible completions endpoint.

Run from the llm_benchmarks repository root:
  python capability/scripts/humanevalplus_generate.py --concurrency 8 --run 1
  python capability/scripts/humanevalplus_generate.py --concurrency 32 --run 1

Only model inference happens here. This script NEVER executes generated code.
Results: capability/results/humanevalplus/generated/c8/google--gemma-2b_run-1.jsonl
"""

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError

DATASET_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"
DATASET_VERSION = "v0.1.10"
EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
DEFAULT_DATASET = Path("capability/datasets/HumanEvalPlus-v0.1.10.jsonl.gz")
DEFAULT_RESULTS = Path("capability/results/humanevalplus")

# As in the earlier successful pilot, truncate locally instead of sending
# server-side stop sequences. Keep the same policy across all measured runs.
STOP = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\ndef ",
    "\nclass ",
    "\nimport ",
    "\nfrom ",
    "\nassert ",
]


def model_slug(model):
    # Preserve the complete repo-qualified name without creating subdirectories.
    if not model or model.startswith("/") or ".." in model.split("/"):
        raise ValueError("Invalid model name")
    slug = model.replace("/", "--")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", slug):
        raise ValueError("Model name cannot be represented safely as a filename")
    return slug


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--model", default="google/gemma-2b")
    _ = parser.add_argument("--concurrency", type=int, required=True)
    _ = parser.add_argument("--run", type=int, required=True)
    _ = parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/completions")
    _ = parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    _ = parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    _ = parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if not 1 <= args.concurrency <= 256:
        parser.error("--concurrency must be between 1 and 256")

    if args.run < 1 or args.timeout < 1:
        parser.error("--run and --timeout must be positive")

    try:
        model_slug(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def tasks_from_dataset(path):
    raw = gzip.decompress(path.read_bytes())

    if hashlib.md5(raw).hexdigest() != DATASET_MD5:
        raise ValueError("Dataset does not match pinned HumanEval+ v0.1.10 MD5")

    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [task["task_id"] for task in tasks]

    if len(tasks) != 164 or set(ids) != {f"HumanEval/{i}" for i in range(164)}:
        raise ValueError("Expected exactly 164 distinct HumanEval tasks")

    return tasks, ids


def atomic_json(path, doc):
    pending = path.with_name(path.name + ".tmp")

    with pending.open("w", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    pending.replace(path)


def atomic_text(path, content):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(content, encoding="utf-8")
    pending.replace(path)


def read_completed(path, expected_ids):
    seen = set()

    if not path.exists():
        return seen

    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                task_id = doc["task_id"]

                assert task_id in expected_ids and task_id not in seen
                assert isinstance(doc["completion"], str)

            except (KeyError, ValueError, AssertionError, TypeError) as exc:
                raise ValueError(f"Bad existing JSONL at line {n} ({path})") from exc

            seen.add(task_id)

    return seen


def request_completion(task, args):
    payload = {
        "model": args.model,
        "prompt": task["prompt"],
        "temperature": 0,
        "max_tokens": 768,
        "seed": 0,
    }

    req = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            answer = json.load(response)

    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{task['task_id']}: vLLM HTTP {exc.code}: {detail}") from exc

    except URLError as exc:
        raise RuntimeError(f"{task['task_id']}: vLLM connectivity failure: {exc}") from exc

    choice = answer["choices"][0]
    generated = choice["text"]

    positions = [pos for stop in STOP if (pos := generated.find(stop)) >= 0]

    completion = generated[: min(positions)] if positions else generated

    return {
        "task_id": task["task_id"],
        "completion": completion.replace("\t", "    "),
        "finish_reason": choice.get("finish_reason"),
        "usage": answer.get("usage"),
    }


def main():
    args = args_parser()
    tasks, ids = tasks_from_dataset(args.dataset)
    expected_ids = set(ids)

    name = f"{model_slug(args.model)}_run-{args.run}"

    directory = args.results_dir / "generated" / f"c{args.concurrency}"
    directory.mkdir(parents=True, exist_ok=True)

    output = directory / f"{name}.jsonl"
    protocol_path = directory / f"{name}.protocol.json"
    manifest_path = directory / f"{name}.manifest.json"
    timing_path = directory / f"{name}.date.txt"

    protocol = {
        "model": args.model,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "run": args.run,
        "dataset_version": DATASET_VERSION,
        "dataset_md5": DATASET_MD5,
        "evalplus_commit": EVALPLUS_COMMIT,
        "generation": {
            "prompt": "exact HumanEval+ dataset prompt",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 768,
            "stop": STOP,
            "stop_handling": "local truncation after generation",
        },
    }

    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

        if existing_protocol != protocol:
            raise RuntimeError(f"Conflicting protocol: {protocol_path}. Refusing to mix generations.")

    elif output.exists() and output.stat().st_size:
        raise RuntimeError("Existing output has no protocol record; refusing unsafe resume")

    else:
        atomic_json(protocol_path, protocol)

    completed = read_completed(output, expected_ids)
    initial_completed = len(completed)

    remaining = [task for task in tasks if task["task_id"] not in completed]

    print(f"{name}: {initial_completed}/164 existing; concurrency={args.concurrency}", flush=True)

    timing = None

    if remaining:
        # Measure only active generation and JSONL writing.
        # Dataset loading and protocol validation are excluded.
        started_at = datetime.now(timezone.utc)
        start_clock = perf_counter()

        # Threads perform HTTP only. Generated Python is NEVER executed on the host.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(request_completion, task, args): task["task_id"] for task in remaining}

            for future in as_completed(futures):
                record = future.result()

                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

                completed.add(record["task_id"])

                print(
                    f"{len(completed)}/164 {record['task_id']}",
                    flush=True,
                )

        elapsed_seconds = perf_counter() - start_clock
        finished_at = datetime.now(timezone.utc)

        # A resumed invocation does not measure the entire benchmark run.
        timing_scope = "full_run" if initial_completed == 0 else "resume_segment"

        timing = {
            "started_utc": started_at.isoformat(),
            "finished_utc": finished_at.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "timing_scope": timing_scope,
            "samples_present_before": initial_completed,
            "samples_generated": len(remaining),
        }

    # Validate all 164 distinct tasks before finalizing.
    if read_completed(output, expected_ids) != expected_ids:
        raise RuntimeError("Incomplete generation; rerun with the same arguments to resume")

    generated_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    # Preserve previously finalized metadata when the run is repeated.
    if not remaining and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if existing_manifest.get("generated_sha256") != generated_sha256:
            raise RuntimeError("Existing manifest checksum does not match generated JSONL")

        print("Run already completed. Existing metadata preserved.")
        print(f"Completed: {output}")
        return

    # Save timing only after successful generation and validation.
    if timing is not None:
        timing_text = f"started_utc={timing['started_utc']}\nfinished_utc={timing['finished_utc']}\nelapsed_seconds={timing['elapsed_seconds']:.3f}\ntiming_scope={timing['timing_scope']}\nsamples_present_before={timing['samples_present_before']}\nsamples_generated={timing['samples_generated']}\n"

        atomic_text(timing_path, timing_text)

        print(f"Generation duration: {timing['elapsed_seconds']:.3f}s")

        if timing["timing_scope"] != "full_run":
            print("WARNING: Resumed run. Duration covers only this invocation, not all 164 samples.")

    else:
        print("No generation performed; full-run duration unavailable.")

    manifest = {
        **protocol,
        "task_count": 164,
        "generated_sha256": generated_sha256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }

    if timing is not None:
        manifest["timing"] = timing

    atomic_json(manifest_path, manifest)

    print(f"Completed: {output}")
    print(f"Protocol: {protocol_path}")
    print(f"Manifest: {manifest_path}")

    if timing is not None:
        print(f"Timing: {timing_path}")


if __name__ == "__main__":
    main()
