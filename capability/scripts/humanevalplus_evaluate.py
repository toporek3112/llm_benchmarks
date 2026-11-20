"""Score generated HumanEval+ completions inside a disposable, offline QEMU VM.

Run from the llm_benchmarks repository root:
  python capability/scripts/humanevalplus_evaluate.py --concurrency 8 --run 1 --model google/gemma-2b --yes-reset
  python capability/scripts/humanevalplus_evaluate.py --all --yes-reset

Requires: sudo, virsh, virt-copy-in, virt-customize, virt-cat.
The working QCOW2 disk is REPLACED for each run. The baseline is preserved.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RESULTS = Path("capability/results/humanevalplus")
IMAGE_DIRECTORY = Path("/var/lib/libvirt/images")
DEFAULT_BASELINE = IMAGE_DIRECTORY / "humaneval-sandbox-runner-baseline.qcow2"
DEFAULT_WORKING_DISK = IMAGE_DIRECTORY / "humaneval-sandbox.qcow2"
DEFAULT_VM = "humaneval-sandbox"
LIBVIRT_CONNECTION = "qemu:///system"
DATASET_MD5 = "fe585eb4df8c88d844eeb463ea4d0302"
EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
GUEST_BENCH_DIRECTORY = "/home/ubuntu/humaneval-bench"
GUEST_RESULT_PATH = f"{GUEST_BENCH_DIRECTORY}/samples_eval_results.json"
RUN_NAME_PATTERN = re.compile(r"^(?P<slug>[A-Za-z0-9._-]+)_run-(?P<run>[1-9]\d*)\.jsonl$")
CONCURRENCY_PATTERN = re.compile(r"c([1-9]\d*)")
EXPECTED_TASK_IDS = {f"HumanEval/{i}" for i in range(164)}


def model_slug(model):
    # Match the generation script's validation and filename convention.
    if not model or model.startswith("/") or ".." in model.split("/"):
        raise ValueError("Invalid model name")
    slug = model.replace("/", "--")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", slug):
        raise ValueError("Model name cannot be represented safely as a filename")
    return slug


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Evaluate all discovered runs")
    mode.add_argument("--concurrency", type=int, help="Select a single cN directory")
    parser.add_argument("--run", type=int, help="Run number for single-run mode")
    parser.add_argument("--model", help="Model ID (required for single-run; optional --all filter)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--working-disk", type=Path, default=DEFAULT_WORKING_DISK)
    parser.add_argument("--timeout", type=int, default=7800, help="VM runtime limit in seconds")
    parser.add_argument("--shutdown-timeout", type=int, default=120)
    parser.add_argument("--yes-reset", action="store_true", help="Explicitly authorize replacement of the disposable working QCOW2 disk")

    args = parser.parse_args()
    if args.timeout < 1 or args.shutdown_timeout < 1:
        parser.error("Timeouts must be positive")

    if args.all:
        if args.run is not None:
            parser.error("--run is only supported with --concurrency")
    elif not (1 <= args.concurrency <= 256) or args.run is None or args.run < 1 or not args.model:
        parser.error("Single-run mode requires --concurrency (1-256), --run and --model")

    if args.model:
        try:
            model_slug(args.model)
        except ValueError as exc:
            parser.error(str(exc))

    if not args.yes_reset:
        parser.error("This replaces the disposable working disk. Specify --yes-reset")

    return args


def run_command(*args, capture=False):
    return subprocess.run(
        [str(value) for value in args],
        check=True,
        text=True,
        capture_output=capture,
    )


def privileged(*args, capture=False):
    return run_command("sudo", *args, capture=capture)


def virsh(*args, capture=False):
    return privileged("virsh", "-c", LIBVIRT_CONNECTION, *args, capture=capture)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_samples(path, concurrency, run):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input must be a regular JSONL file: {path}")

    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                task_id = document["task_id"]
                if task_id not in EXPECTED_TASK_IDS or task_id in seen:
                    raise ValueError("unknown or duplicate task_id")
                if not isinstance(document.get("completion"), str):
                    raise ValueError("missing or invalid completion")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            seen.add(task_id)

    if seen != EXPECTED_TASK_IDS:
        raise ValueError(f"Expected 164 distinct tasks in {path}; found {len(seen)}")

    generated_sha256 = sha256(path)
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise ValueError(f"Generation manifest must not be a symlink: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generation = manifest.get("generation", {})
        if (
            manifest.get("generated_sha256") != generated_sha256
            or manifest.get("dataset_md5") != DATASET_MD5
            or manifest.get("evalplus_commit") != EVALPLUS_COMMIT
            or manifest.get("task_count") != 164
            or manifest.get("concurrency") != concurrency
            or manifest.get("run") != run
            or model_slug(manifest.get("model")) != path.stem.removesuffix(f"_run-{run}")
            or generation.get("temperature") != 0
        ):
            raise ValueError(f"Generation manifest mismatch: {manifest_path}")
    else:
        print(f"WARNING: {path.name}: no generation manifest (pilot/manual run?)", flush=True)

    return generated_sha256


def validate_eval_bytes(raw):
    document = json.loads(raw)
    evaluations = document.get("eval")
    if document.get("hash") != DATASET_MD5 or not isinstance(evaluations, dict):
        raise ValueError("Incorrect dataset hash or EvalPlus result structure")
    if set(evaluations) != EXPECTED_TASK_IDS:
        raise ValueError("EvalPlus results do not contain all 164 tasks")

    base_passed = 0
    plus_passed = 0
    for task_id, outcomes in evaluations.items():
        if not isinstance(outcomes, list) or len(outcomes) != 1:
            raise ValueError(f"Expected one evaluated completion for {task_id}")
        outcome = outcomes[0]
        if outcome.get("task_id") != task_id:
            raise ValueError(f"Wrong task_id inside evaluation for {task_id}")
        if not all(isinstance(outcome.get(field), str) for field in ("base_status", "plus_status")):
            raise ValueError(f"Missing evaluation status for {task_id}")
        base_passed += outcome["base_status"] == "pass"
        plus_passed += outcome["base_status"] == outcome["plus_status"] == "pass"

    return base_passed, plus_passed


def vm_xml(vm):
    xml_text = virsh("dumpxml", "--inactive", vm, capture=True).stdout
    return ET.fromstring(xml_text)


def verify_working_disk(vm_config, working_disk):
    disk_paths = []
    for disk in vm_config.findall("./devices/disk"):
        if disk.get("device") != "disk":
            continue
        source = disk.find("source")
        if source is not None and source.get("file"):
            disk_paths.append(Path(source.get("file")))
    if disk_paths != [working_disk]:
        raise RuntimeError(f"VM must use exactly one expected disk ({working_disk}); found: {disk_paths}")


def detach_network(vm):
    vm_config = vm_xml(vm)
    for interface in vm_config.findall("./devices/interface"):
        interface_type = interface.get("type")
        mac = interface.find("mac")
        if interface_type != "network" or mac is None or not mac.get("address"):
            raise RuntimeError("Unexpected VM network interface; refusing automatic modification")
        virsh("detach-interface", vm, "network", "--mac", mac.get("address"), "--config")

    vm_config = vm_xml(vm)
    if vm_config.findall("./devices/interface"):
        raise RuntimeError("VM still has a network interface in its inactive configuration")
    for device in ("hostdev", "filesystem", "vsock", "redirdev"):
        if vm_config.findall(f"./devices/{device}"):
            raise RuntimeError(f"Unexpected VM {device} device; verify isolation before boot")

    print("VM network isolation verified in persistent configuration", flush=True)


def vm_state(vm):
    return virsh("domstate", vm, capture=True).stdout.strip()


def wait_for_shutdown(vm, timeout, force_at_timeout=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if vm_state(vm) == "shut off":
            return
        time.sleep(2)

    if force_at_timeout:
        print("ERROR: VM runtime limit reached; forcing the sandbox off", flush=True)
        virsh("destroy", vm)
        raise TimeoutError("Benchmark VM exceeded its runtime limit")
    raise TimeoutError(f"VM {vm} did not shut down within {timeout} seconds")


def ensure_vm_stopped(vm, timeout):
    if vm_state(vm) == "shut off":
        return
    print("VM is running: requesting graceful shutdown before disk reset", flush=True)
    virsh("shutdown", vm)
    wait_for_shutdown(vm, timeout)


def jobs_from_args(args):
    generated_dir = args.results_dir / "generated"
    if args.all:
        candidates = sorted(
            generated_dir.glob("c*/*.jsonl"),
            key=lambda path: (
                int(path.parent.name[1:]) if CONCURRENCY_PATTERN.fullmatch(path.parent.name) else -1,
                path.name,
            ),
        )
    else:
        name = f"{model_slug(args.model)}_run-{args.run}.jsonl"
        candidates = [generated_dir / f"c{args.concurrency}" / name]

    jobs = []
    for generated_path in candidates:
        match = RUN_NAME_PATTERN.fullmatch(generated_path.name)
        concurrency_match = CONCURRENCY_PATTERN.fullmatch(generated_path.parent.name)
        if match is None or concurrency_match is None:
            raise ValueError(f"Unexpected run path: {generated_path}")
        if args.model and model_slug(args.model) != match.group("slug"):
            continue
        if not generated_path.is_file():
            raise FileNotFoundError(f"Missing generated file: {generated_path}")

        evaluated_path = args.results_dir / "evaluated" / generated_path.parent.name / f"{generated_path.stem}_eval_results.json"
        jobs.append((generated_path, evaluated_path, int(concurrency_match.group(1)), int(match.group("run"))))

    if not jobs:
        raise ValueError(f"No generated runs found under {generated_dir}")
    return jobs


def prepare_working_disk(args, generated_path, temporary_dir):
    working_disk = args.working_disk
    baseline = args.baseline
    if baseline == working_disk:
        raise RuntimeError("Baseline and working disk must differ")
    if baseline.is_symlink() or not baseline.is_file():
        raise RuntimeError("Baseline is missing or is a symlink")
    if working_disk.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked working disk")
    if working_disk.exists() and not working_disk.is_file():
        raise RuntimeError("Working disk exists but is not a regular file")

    verify_working_disk(vm_xml(args.vm), working_disk)
    if vm_state(args.vm) != "shut off":
        raise RuntimeError("VM must be shut off before manipulating its disk")

    print("Restoring the disposable working disk from the read-only baseline", flush=True)
    privileged("rm", "-f", "--", working_disk)
    privileged("cp", "--reflink=auto", "--", baseline, working_disk)
    privileged("chmod", "600", working_disk)

    # Transfer data into the powered-off guest, never execute it on the host.
    transfer_path = Path(temporary_dir) / "samples.jsonl"
    shutil.copyfile(generated_path, transfer_path)
    transfer_path.chmod(0o644)
    privileged(
        "virt-copy-in",
        "--format=qcow2",
        "-a",
        working_disk,
        transfer_path,
        GUEST_BENCH_DIRECTORY,
    )

    # Automatically shut down the guest after the benchmark service stops.
    # ExecStopPost= runs after success or failure; '+' grants only this trusted
    # systemctl operation the privileges required for a clean shutdown.
    dropin_path = Path(temporary_dir) / "90-auto-poweroff.conf"
    dropin_path.write_text(
        "[Service]\nExecStopPost=+/usr/bin/systemctl --no-block poweroff\n",
        encoding="utf-8",
    )
    dropin_path.chmod(0o644)
    guest_dropin_dir = "/etc/systemd/system/humaneval-bench.service.d"
    guest_dropin_path = f"{guest_dropin_dir}/{dropin_path.name}"
    privileged(
        "virt-customize",
        "--no-network",
        "--format",
        "qcow2",
        "-a",
        working_disk,
        "--mkdir",
        guest_dropin_dir,
        "--copy-in",
        f"{dropin_path}:{guest_dropin_dir}",
        "--chown",
        f"0:0:{guest_dropin_path}",
        "--chmod",
        f"0644:{guest_dropin_path}",
    )


def evaluation_manifest_path(evaluated_path):
    name = evaluated_path.stem.removesuffix("_eval_results")
    return evaluated_path.with_name(f"{name}.evaluation_manifest.json")


def evaluate_job(args, generated_path, evaluated_path, concurrency, run):
    generated_sha256 = validate_samples(generated_path, concurrency, run)
    manifest_path = evaluation_manifest_path(evaluated_path)

    if evaluated_path.exists():
        validate_eval_bytes(evaluated_path.read_bytes())
        if not manifest_path.is_file():
            raise RuntimeError(f"Existing results have no evaluation manifest: {evaluated_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generated_sha256") != generated_sha256:
            raise RuntimeError(f"Existing result belongs to a different input: {evaluated_path}")
        print(f"Skipping already evaluated run: {evaluated_path}", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="humanevalplus-eval-") as temporary_dir:
        ensure_vm_stopped(args.vm, args.shutdown_timeout)
        prepare_working_disk(args, generated_path, temporary_dir)
        detach_network(args.vm)

        print(f"Booting offline VM to evaluate {generated_path.name}", flush=True)
        print(f"Monitor: sudo virsh -c {LIBVIRT_CONNECTION} console {args.vm}", flush=True)
        started = time.monotonic()
        virsh("start", args.vm)
        wait_for_shutdown(args.vm, args.timeout, force_at_timeout=True)
        duration = time.monotonic() - started

        # Extract JSON from the *stopped* guest via libguestfs.
        raw = subprocess.run(
            ["sudo", "virt-cat", "--format=qcow2", "-a", str(args.working_disk), GUEST_RESULT_PATH],
            check=True,
            capture_output=True,
        ).stdout
        base_passed, plus_passed = validate_eval_bytes(raw)

        evaluated_path.parent.mkdir(parents=True, exist_ok=True)
        if evaluated_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing result: {evaluated_path}")
        pending_path = evaluated_path.with_name(evaluated_path.name + ".tmp")
        pending_path.write_bytes(raw)
        pending_path.replace(evaluated_path)

        summary = {
            "generated_input": str(generated_path),
            "generated_sha256": generated_sha256,
            "evaluated_output": str(evaluated_path),
            "evaluated_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset_md5": DATASET_MD5,
            "evalplus_commit": EVALPLUS_COMMIT,
            "concurrency": concurrency,
            "run": run,
            "vm": args.vm,
            "baseline": str(args.baseline),
            "network": "no libvirt network interface at boot",
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "total_tasks": 164,
            "vm_runtime_seconds": round(duration, 2),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        pending_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        pending_manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        pending_manifest.replace(manifest_path)
        print(f"RESULT: HumanEval={base_passed}/164, HumanEval+={plus_passed}/164", flush=True)
        print(f"Saved: {evaluated_path}", flush=True)


def main():
    args = args_parser()
    for tool_name in ("sudo", "virsh", "virt-copy-in", "virt-customize", "virt-cat"):
        if shutil.which(tool_name) is None:
            raise RuntimeError(f"Required tool is missing: {tool_name}")
    if args.baseline == args.working_disk:
        raise RuntimeError("Never use the baseline as the VM working disk")

    jobs = jobs_from_args(args)
    print(f"Preparing {len(jobs)} evaluation run(s)", flush=True)
    run_command("sudo", "-v")
    for number, (generated_path, evaluated_path, concurrency, run) in enumerate(jobs, 1):
        print(f"\n[{number}/{len(jobs)}] {generated_path}", flush=True)
        evaluate_job(args, generated_path, evaluated_path, concurrency, run)
    print("Finished. The working VM remains stopped and the baseline is unchanged.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, TimeoutError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
