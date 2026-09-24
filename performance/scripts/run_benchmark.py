#!/usr/bin/env python3
"""Run an Ollama serving benchmark and collect Shelly power from Prometheus.

Use benchmark for one run, campaign for repeated model blocks, or ls to inspect models.
Use --help on a command for its flags and defaults.
Requires Python 3.10+, pydantic 2, Typer, and a separate vllm[bench] installation.
"""

from __future__ import annotations

import codecs
import datetime as dt
import itertools
import json
import math
import os
import re
import selectors
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence, cast

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


JsonObject = dict[str, Any]  # External API responses and on-disk JSON are dynamic.
WorkloadProfile = tuple[float, int, int, int]  # Request rate, concurrency, input tokens, output tokens.


def now() -> float:
  """Return the current Unix timestamp in seconds."""
  return time.time()


def path_label(value: str) -> str:
  """Turn a model or hardware label into one safe path component."""
  return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def quantization_label(value: str | None) -> str:
  """Preserve Ollama's familiar spelling, such as Q4_K_M, in folder names."""
  return re.sub(r"[^A-Za-z0-9_-]+", "-", value or "").strip("-_ ") or "quant-unknown"


def number_label(value: float) -> str:
  """Keep a float readable in a folder name without a decimal point."""
  return repr(value).replace(".", "p").replace("+", "").replace("-", "m")


def installed_quantization(installedModel: JsonObject) -> str | None:
  """Read Ollama's reported model-weight quantization, when available."""
  details: object = installedModel.get("details")
  quantization: object = details.get("quantization_level") if isinstance(details, dict) else None
  return quantization.strip() if isinstance(quantization, str) and quantization.strip() else None


def save_json(path: Path, data: JsonObject) -> None:
  """Write JSON through a temporary file so readers never see partial data."""
  temporary: Path = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  temporary.replace(path)


def get_json(url: str, params: Mapping[str, str] | None = None) -> JsonObject:
  """GET one JSON object from an HTTP API, optionally with query parameters."""
  if params:
    url += "?" + urllib.parse.urlencode(params)

  request: urllib.request.Request = urllib.request.Request(url, headers={"Accept": "application/json"})
  with urllib.request.urlopen(request, timeout=20) as response:
    data: object = json.load(response)

  if not isinstance(data, dict):
    raise RuntimeError(f"Expected a JSON object from {url}")

  return cast(JsonObject, data)


def post_json(url: str, body: Mapping[str, str | bool | int], timeout: int) -> JsonObject:
  """POST a JSON object and return the JSON object in the response."""
  request: urllib.request.Request = urllib.request.Request(
    url,
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json", "Accept": "application/json"},
  )

  with urllib.request.urlopen(request, timeout=timeout) as response:
    data: object = json.load(response)

  if not isinstance(data, dict):
    raise RuntimeError(f"Expected a JSON object from {url}")

  return cast(JsonObject, data)


def detect_quantization(llm_url: str, model_name: str, installedModel: JsonObject) -> tuple[str | None, str]:
  """Prefer Ollama's tag metadata, then try /api/show if it lacks quantization."""
  quantization: str | None = installed_quantization(installedModel)
  if quantization is not None:
    return quantization, "tags"

  try:
    shown: JsonObject = post_json(llm_url + "/api/show", {"model": model_name}, timeout=20)
  except (OSError, RuntimeError, ValueError):
    return None, "unavailable"

  quantization = installed_quantization(shown)
  return quantization, "show" if quantization is not None else "unavailable"


class ValidatedConfig(BaseModel):
  """Share Pydantic validation rules across typed configuration groups."""

  model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class LLMConfig(ValidatedConfig):
  """Settings for the served model and its Ollama cold/warm preparation."""

  llm_url: str = Field(min_length=1, description="Ollama API base URL, including port.")
  llm_model: str = Field(min_length=1, description="Served model name, for example llama3.2:3b.")
  llm_backend: Literal["ollama"] = Field(default="ollama", description="Model control API; currently only ollama is supported.")
  llm_start_mode: Literal["cold", "warm"] = Field(description="cold unloads and measures a separate load; warm preloads before timing.")
  llm_keep_alive: str = Field(default="30m", min_length=1, description="Ollama residency after loading.")
  llm_load_timeout_seconds: int = Field(default=300, ge=1, description="Timeout for an Ollama load request.")
  llm_state_timeout_seconds: int = Field(default=30, ge=1, description="Timeout when checking loaded/unloaded state.")

  @field_validator("llm_url")
  @classmethod
  def validate_url(cls, value: str) -> str:
    """Require an HTTP base URL and remove a trailing slash."""
    parts: urllib.parse.SplitResult = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
      raise ValueError("--llm-url must be an http(s) base URL")

    return value.rstrip("/")


class PrometheusConfig(ValidatedConfig):
  """Settings for sensor freshness checks and bounded power queries."""

  prometheus_url: str = Field(min_length=1, description="Prometheus API base URL.")
  prometheus_sensors: tuple[str, ...] = Field(
    default=("shelly-gpu", "shelly-gpu-node"),
    description="Sensor labels to require and collect.",
  )
  prometheus_step_seconds: int = Field(default=10, ge=1, description="PromQL evaluation step in seconds; scrapes are stored separately.")
  prometheus_pre_padding_seconds: int = Field(default=10, ge=0, description="Seconds of power data before model preparation or the benchmark.")
  prometheus_post_padding_seconds: int = Field(default=10, ge=0, description="Seconds of power data after the benchmark.")
  prometheus_settle_seconds: int = Field(default=20, ge=0, description="Wait after benchmark for scrapes to arrive; must be >= step + post padding.")
  prometheus_preflight_seconds: int = Field(default=600, ge=1, description="Lookback window for recent sensor checks.")
  prometheus_max_sample_age_seconds: int = Field(default=30, ge=1, description="Maximum age of each sensor's newest scrape.")

  @field_validator("prometheus_url")
  @classmethod
  def validate_url(cls, value: str) -> str:
    """Require an HTTP base URL and remove a trailing slash."""
    parts: urllib.parse.SplitResult = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
      raise ValueError("--prometheus-url must be an http(s) base URL")

    return value.rstrip("/")

  @field_validator("prometheus_sensors")
  @classmethod
  def validate_sensors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
    """Reject empty or repeated sensor labels."""
    if not value or any(not sensor for sensor in value) or len(value) != len(set(value)):
      raise ValueError("sensor names must be nonempty and distinct")

    return value

  @model_validator(mode="after")
  def validate_settle_time(self) -> PrometheusConfig:
    """Allow enough time for the final scrape and query padding to arrive."""
    if self.prometheus_settle_seconds < self.prometheus_step_seconds + self.prometheus_post_padding_seconds:
      raise ValueError("--power-settle must be >= --power-step + --power-post-padding")

    return self


class VLLMConfig(ValidatedConfig):
  """Settings for the vLLM benchmark CLI and its generated workload."""

  vllm_bin: str | None = Field(default=None, description="Path to vllm; otherwise use PATH or vllm-bench/.venv/bin/vllm.")
  vllm_model: str = Field(min_length=1, description="Model identifier used by vLLM to prepare prompts.")
  vllm_tokenizer: str = Field(min_length=1, description="Tokenizer identifier used to count and generate prompt tokens.")
  vllm_num_prompts: int = Field(default=10, ge=1, description="Number of measured requests (probe and warmups are extra).")
  vllm_request_rate: float = Field(default=0.5, gt=0, description="Average request arrivals per second (Poisson).")
  vllm_max_concurrency: int = Field(default=1, ge=1, description="Maximum simultaneous requests.")
  vllm_num_warmups: int = Field(default=2, ge=0, description="Warmup requests before the measured run.")
  vllm_random_input_len: int = Field(default=512, ge=1, description="Target random prompt length in tokens.")
  vllm_random_output_len: int = Field(default=128, ge=1, description="Target/cap output tokens per request.")
  vllm_random_range_ratio: float = Field(default=0.0, ge=0, description="Random length variation (0 gives fixed target lengths).")
  vllm_temperature: float = Field(default=0.0, ge=0, description="Sampling temperature; 0 is deterministic decoding.")
  vllm_seed: int = Field(default=42, description="Random prompt and arrival seed.")
  vllm_show_output: bool = Field(default=False, description="Mirror vLLM output to the terminal; benchmark.log is always saved.")

  @field_validator("vllm_bin")
  @classmethod
  def validate_bin(cls, value: str | None) -> str | None:
    """Reject an explicitly empty vLLM executable path."""
    if value == "":
      raise ValueError("--vllm-bin must be a nonempty path when set")

    return value

  def find_bin(self) -> str:
    """Locate the executable and detect an invalid venv interpreter path."""
    project_root: Path = Path(__file__).resolve().parent.parent
    candidate: str = self.vllm_bin or shutil.which("vllm") or str(project_root / "vllm-bench" / ".venv" / "bin" / "vllm")
    binary: Path = Path(candidate).expanduser()

    if not binary.is_file() or not os.access(binary, os.X_OK):
      raise RuntimeError("vllm not found; activate its environment or pass --vllm-bin")

    with binary.open("rb") as handle:
      shebang: str = handle.readline(4096).decode("utf-8", errors="replace").strip()

    if shebang.startswith("#!"):
      interpreter: str = shebang[2:].split(maxsplit=1)[0]
      if interpreter.startswith("/") and not Path(interpreter).exists():
        raise RuntimeError(f"vllm points to a missing interpreter: {interpreter}; recreate the moved venv")

    return str(binary)

  def command(self, llmConfig: LLMConfig, run_dir: Path) -> list[str]:
    """Build the vllm bench serve command without executing it."""
    return [
      self.find_bin(),
      "bench",
      "serve",
      "--backend",
      "openai-chat",
      "--base-url",
      llmConfig.llm_url,
      "--endpoint",
      "/v1/chat/completions",
      "--model",
      self.vllm_model,
      "--served-model-name",
      llmConfig.llm_model,
      "--tokenizer",
      self.vllm_tokenizer,
      "--dataset-name",
      "random",
      "--random-input-len",
      str(self.vllm_random_input_len),
      "--random-output-len",
      str(self.vllm_random_output_len),
      "--random-range-ratio",
      str(self.vllm_random_range_ratio),
      "--num-prompts",
      str(self.vllm_num_prompts),
      "--request-rate",
      str(self.vllm_request_rate),
      "--max-concurrency",
      str(self.vllm_max_concurrency),
      "--num-warmups",
      str(self.vllm_num_warmups),
      "--temperature",
      str(self.vllm_temperature),
      "--seed",
      str(self.vllm_seed),
      "--extra-body",
      json.dumps({"max_tokens": self.vllm_random_output_len}),
      "--percentile-metrics",
      "ttft,tpot,itl,e2el",
      "--metric-percentiles",
      "50,95,99",
      "--save-result",
      "--save-detailed",
      "--result-dir",
      str(run_dir),
      "--result-filename",
      "benchmark.json",
    ]


class BenchConfig(ValidatedConfig):
  """Labels and result location used to identify a benchmark run."""

  bench_deployment: str = Field(min_length=1, description="Deployment label, e.g. docker or kubernetes.")
  bench_hardware: str = Field(min_length=1, description="Hardware label, e.g. gtx-1070.")
  bench_results_dir: Path = Field(default=Path("results"), description="Parent directory for model/run folders.")
  bench_campaign_id: str | None = Field(default=None, min_length=1, description="Optional experiment/campaign identifier.")
  bench_replicate: int | None = Field(default=None, ge=1, description="Optional positive repetition number.")
  bench_scenario: str | None = Field(default=None, min_length=1, description="Optional workload label.")

  @field_validator("bench_results_dir", mode="before")
  @classmethod
  def validate_results_dir(cls, value: object) -> object:
    """Require a nonempty results path when one is supplied."""
    if value == "":
      raise ValueError("--results-dir must be nonempty")

    return value


@dataclass(frozen=True)
class Config:
  """Group the independently validated settings used by the run coordinator."""

  llm: LLMConfig
  prometheus: PrometheusConfig
  vllm: VLLMConfig
  bench: BenchConfig


def run_folder_name(runConfig: Config, quantization: str | None, run_id: str) -> str:
  """Identify the experiment at a glance and keep the timestamp for uniqueness."""
  llmConfig: LLMConfig = runConfig.llm
  vllmConfig: VLLMConfig = runConfig.vllm
  benchConfig: BenchConfig = runConfig.bench

  return "_".join(
    (
      quantization_label(quantization),
      f"gpu-{path_label(benchConfig.bench_hardware)}",
      path_label(benchConfig.bench_deployment),
      llmConfig.llm_start_mode,
      f"i{vllmConfig.vllm_random_input_len}",
      f"o{vllmConfig.vllm_random_output_len}",
      f"rps{number_label(vllmConfig.vllm_request_rate)}",
      f"c{vllmConfig.vllm_max_concurrency}",
      f"n{vllmConfig.vllm_num_prompts}",
      f"w{vllmConfig.vllm_num_warmups}",
      f"range{number_label(vllmConfig.vllm_random_range_ratio)}",
      f"temp{number_label(vllmConfig.vllm_temperature)}",
      f"seed{vllmConfig.vllm_seed}",
      run_id,
    )
  )


class ModelSpec(ValidatedConfig):
  """Benchmark identity and optional expected quantization for one Ollama tag."""

  name: str = Field(min_length=1, description="Exact Ollama model tag.")
  vllm_model: str = Field(min_length=1, description="Model for vLLM prompt generation.")
  tokenizer: str = Field(min_length=1, description="Tokenizer for vLLM prompt generation.")
  quantization: str | None = Field(default=None, min_length=1, description="Expected Ollama quantization, if pinned; live value comes from /api/tags.")
  description: str = Field(default="", description="Human-readable model label.")


class ModelRegistry(ValidatedConfig):
  """Load and validate the ordered model catalog from metadata/models.json."""

  schema_version: Literal[1]
  models: tuple[ModelSpec, ...] = Field(min_length=1)

  @model_validator(mode="after")
  def validate_names(self) -> ModelRegistry:
    """Reject duplicate model tags before any benchmark starts."""
    names: list[str] = [model.name for model in self.models]
    if len(names) != len(set(names)):
      raise ValueError("model names in the registry must be distinct")

    return self

  @classmethod
  def from_file(cls, path: Path) -> ModelRegistry:
    """Read a JSON model registry with a useful path in validation errors."""
    try:
      return cls.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
      raise ValueError(f"Cannot load model registry {path}: {error}") from error

  def by_name(self) -> dict[str, ModelSpec]:
    """Return catalog entries keyed by exact Ollama tag, preserving order."""
    return {model.name: model for model in self.models}

  def require(self, name: str) -> ModelSpec:
    """Get a registered model or list valid names in the error."""
    models: dict[str, ModelSpec] = self.by_name()
    if name not in models:
      raise ValueError(f"Unknown model {name!r}; choose from: {', '.join(models)}")

    return models[name]


def load_registry(models_file: Path) -> ModelRegistry:
  """Resolve --models-file relative to the project root by default."""
  project_root: Path = Path(__file__).resolve().parent.parent
  configured_path: Path = models_file.expanduser()
  if not configured_path.is_absolute():
    configured_path = project_root / configured_path

  return ModelRegistry.from_file(configured_path)


class CampaignConfig(ValidatedConfig):
  """Settings for repeated model-block campaigns; single runs ignore these."""

  bench_model_order: tuple[str, ...] | None = Field(
    default=None,
    description="Model blocks in order; default is every model in the JSON catalog.",
  )
  bench_repetitions: int = Field(default=5, ge=1, description="Repetitions of every warm workload profile per model.")
  bench_prompts: int = Field(default=120, ge=1, description="Measured requests per warm case.")
  bench_rest_seconds: int = Field(default=30, ge=0, description="Pause between consecutive cases without unloading the model.")
  bench_include_cold: bool = Field(default=False, description="Also measure separate cold loads after each warm model block.")
  bench_request_rates: tuple[float, ...] = Field(default=(0.25, 0.5, 1.0), min_length=1, description="Warm arrival rates in requests per second.")
  bench_max_concurrencies: tuple[int, ...] = Field(default=(4,), min_length=1, description="Warm request concurrency limits.")
  bench_input_lengths: tuple[int, ...] = Field(default=(512,), min_length=1, description="Warm random input lengths in tokens.")
  bench_output_lengths: tuple[int, ...] = Field(default=(128,), min_length=1, description="Warm random output targets/caps in tokens.")
  bench_cold_prompts: int = Field(default=10, ge=1, description="Measured requests in each separate cold-load case.")
  bench_cold_request_rate: float = Field(default=0.5, gt=0, description="Cold case request arrival rate.")
  bench_cold_concurrency: int = Field(default=1, ge=1, description="Cold case concurrency limit.")
  bench_cold_input_len: int = Field(default=512, ge=1, description="Input length for separate cold-load cases.")
  bench_cold_output_len: int = Field(default=128, ge=1, description="Output target/cap for separate cold-load cases.")

  @field_validator("bench_model_order")
  @classmethod
  def validate_model_order(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Require distinct, nonempty model tags when an order is supplied."""
    if value is not None and (not value or len(value) != len(set(value)) or any(not name for name in value)):
      raise ValueError("--model needs distinct, nonempty model names")

    return value

  @field_validator("bench_request_rates")
  @classmethod
  def validate_rates(cls, value: tuple[float, ...]) -> tuple[float, ...]:
    """Reject invalid or repeated campaign request rates."""
    if not value or len(value) != len(set(value)) or any(rate <= 0 for rate in value):
      raise ValueError("--request-rate needs distinct positive values")

    return value

  @field_validator("bench_max_concurrencies", "bench_input_lengths", "bench_output_lengths")
  @classmethod
  def validate_positive_axis(cls, value: tuple[int, ...]) -> tuple[int, ...]:
    """Avoid duplicate profiles and invalid concurrency or token lengths."""
    if not value or len(value) != len(set(value)) or any(item <= 0 for item in value):
      raise ValueError("workload grid values must be distinct positive integers")

    return value


class PrometheusClient:
  """Query Shelly power as both stepped evaluations and raw scrape samples."""

  def __init__(self, config: PrometheusConfig) -> None:
    """Build a selector for exactly the configured Shelly sensor labels."""
    self.config: PrometheusConfig = config
    names: str = "|".join(re.escape(name).replace(r"\-", "-") for name in config.prometheus_sensors)
    self.selector: str = f"shelly_power_watts{{sensor=~{json.dumps(names)}}}"

  def query(self, endpoint: str, params: Mapping[str, str]) -> JsonObject:
    """Fetch a Prometheus matrix response or raise on an API error."""
    response: JsonObject = get_json(self.config.prometheus_url + endpoint, params)
    if response.get("status") != "success" or response.get("data", {}).get("resultType") != "matrix":
      raise RuntimeError(f"Prometheus returned an unexpected result: {response.get('error', response.get('status'))}")

    return response

  def check_freshness(self, at: float, minimum_timestamp: float | None = None) -> dict[str, float]:
    """Return each sensor's latest scrape epoch, rejecting missing or stale data.

    With minimum_timestamp, also require a scrape at or after the run end.
    The values are timestamps in seconds, not power measurements in watts.
    """
    expression: str = f"{self.selector}[{self.config.prometheus_preflight_seconds}s]"
    response: JsonObject = self.query("/api/v1/query", {"query": expression, "time": f"{at:.3f}"})
    latest: dict[str, float] = {}

    for series in response["data"]["result"]:
      samples: list[list[float | str]] = series.get("values", [])
      sensor: str | None = series["metric"].get("sensor")
      if sensor in self.config.prometheus_sensors and samples:
        assert sensor is not None
        latest[sensor] = max(latest.get(sensor, 0.0), float(samples[-1][0]))

    missing: set[str] = set(self.config.prometheus_sensors) - set(latest)
    stale: dict[str, float] = {sensor: round(at - latest[sensor], 1) for sensor in latest if at - latest[sensor] > self.config.prometheus_max_sample_age_seconds}
    incomplete: list[str] = [sensor for sensor in latest if minimum_timestamp is not None and latest[sensor] < minimum_timestamp]

    if missing or stale or incomplete:
      returned: list[str] = sorted({series["metric"].get("sensor", "<none>") for series in response["data"]["result"]})
      raise RuntimeError(f"Power samples missing={sorted(missing)}, stale_ages_seconds={stale}, not_yet_through_run_end={incomplete}; query={expression}; returned_sensors={returned}")

    return latest

  def fetch_power(self, start: float, end: float) -> tuple[JsonObject, JsonObject]:
    """Query evaluated power at the configured step within the run window."""
    # query_range timestamps are evaluation times, not necessarily scrape times.
    expression: str = f"max by (sensor) ({self.selector})"
    params: dict[str, str] = {
      "query": expression,
      "start": f"{start:.3f}",
      "end": f"{end:.3f}",
      "step": str(self.config.prometheus_step_seconds),
    }
    response: JsonObject = self.query("/api/v1/query_range", params)
    counts: dict[str, int] = {sensor: 0 for sensor in self.config.prometheus_sensors}

    for series in response["data"]["result"]:
      sensor: str | None = series["metric"].get("sensor")
      if sensor in counts:
        assert sensor is not None
        counts[sensor] += len(series.get("values", []))

    details: JsonObject = {
      "expression": expression,
      "range_start_epoch": start,
      "range_end_epoch": end,
      "step_seconds": self.config.prometheus_step_seconds,
      "sample_counts": counts,
      "timestamps_are_evaluation_times": True,
    }
    return response, details

  def fetch_raw_power(self, start: float, end: float) -> tuple[JsonObject, JsonObject]:
    """Fetch original scrape timestamps and power values in the run window."""
    # An instant range-vector query returns original scrape timestamps/labels.
    lookback: int = math.ceil(end - start) + 1
    expression: str = f"{self.selector}[{lookback}s]"
    response: JsonObject = self.query("/api/v1/query", {"query": expression, "time": f"{end:.3f}"})
    counts: dict[str, int] = {sensor: 0 for sensor in self.config.prometheus_sensors}
    series_counts: dict[str, int] = dict.fromkeys(self.config.prometheus_sensors, 0)

    for series in response["data"]["result"]:
      sensor: str | None = series.get("metric", {}).get("sensor")
      if sensor in counts:
        assert sensor is not None
        series_counts[sensor] += 1
        counts[sensor] += sum(start <= float(value[0]) <= end for value in series.get("values", []))

    details: JsonObject = {
      "expression": expression,
      "range_start_epoch": start,
      "range_end_epoch": end,
      "sample_counts": counts,
      "series_counts": series_counts,
      "timestamps_are_scrape_times": True,
    }
    return response, details


class BenchmarkLogger:
  """Run vLLM, save its complete output, and locate approximate phase markers."""

  MARKERS: dict[str, str] = {
    "Starting initial single prompt test run": "initial_probe_marker_epoch",
    "Warming up with": "warmup_marker_epoch",
    "Starting main benchmark run": "main_start_marker_epoch",
    "Serving Benchmark Result": "result_marker_epoch",
  }

  def __init__(self, path: Path, show_output: bool) -> None:
    """Store the log path and choose whether to mirror output to the terminal."""
    self.path: Path = path
    self.show_output: bool = show_output
    self.markers: dict[str, float] = {}

  def run(self, vllmCommand: Sequence[str]) -> tuple[int, float, float, dict[str, float]]:
    """Execute vLLM via Popen; return exit code, start/end epochs, markers.

    Always consume the pipe to write benchmark.log and find phase markers.
    --show-output controls only the optional terminal copy.
    """
    started: float = now()
    decoder: codecs.IncrementalDecoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending: str = ""

    with self.path.open("w", encoding="utf-8") as log:
      with subprocess.Popen(
        vllmCommand,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
      ) as process:
        assert process.stdout is not None

        with selectors.DefaultSelector() as selector:
          selector.register(process.stdout, selectors.EVENT_READ)

          while selector.get_map():
            for key, _ in selector.select(timeout=1):
              chunk: bytes = os.read(key.fd, 65536)
              if not chunk:
                selector.unregister(key.fileobj)
                text: str = decoder.decode(b"", final=True)
              else:
                text = decoder.decode(chunk)

              log.write(text)
              log.flush()
              if self.show_output:
                sys.stdout.write(text)
                sys.stdout.flush()

              pending += text
              lines: list[str] = re.split(r"[\r\n]", pending)
              pending = lines.pop()

              for line in lines:
                for marker, name in self.MARKERS.items():
                  if marker in line:
                    self.markers.setdefault(name, now())

          status: int = process.wait()

    return status, started, now(), self.markers


class OllamaController:
  """Check Ollama residency and prepare a measured cold or warm model load."""

  def __init__(self, config: LLMConfig) -> None:
    """Use only LLM settings for the model state API."""
    self.config: LLMConfig = config

  def state(self) -> JsonObject:
    """Query /api/ps and describe whether the requested model is loaded."""
    models: list[JsonObject] = get_json(self.config.llm_url + "/api/ps").get("models", [])
    matching: list[JsonObject] = [model for model in models if model.get("name") == self.config.llm_model or model.get("model") == self.config.llm_model]
    return {
      "loaded": bool(matching),
      "match": matching,
      "loaded_model_names": [model.get("name", model.get("model")) for model in models],
    }

  def unload_other_models(self, campaign_models: Mapping[str, ModelSpec]) -> None:
    """At a model-block boundary, unload known previous models once.

    Stop if an unrelated model is loaded, so the campaign cannot measure an
    unexpected shared workload or silently evict another user's model.
    """
    names: list[str] = self.state()["loaded_model_names"]
    other_names: list[str] = [name for name in names if name != self.config.llm_model]
    unexpected: list[str] = [name for name in other_names if name not in campaign_models]

    if unexpected:
      raise RuntimeError(f"Other Ollama models loaded: {unexpected}; stop them before benchmarking")

    for name in dict.fromkeys(other_names):
      print(f"Unloading previous model: {name}", flush=True)
      post_json(
        self.config.llm_url + "/api/generate",
        {"model": name, "stream": False, "keep_alive": 0},
        self.config.llm_load_timeout_seconds,
      )
      deadline: float = time.monotonic() + self.config.llm_state_timeout_seconds

      while name in self.state()["loaded_model_names"]:
        if time.monotonic() >= deadline:
          raise RuntimeError(f"Previous model did not unload: {name}")

        time.sleep(0.5)

  def wait_for(self, loaded: bool) -> JsonObject:
    """Poll /api/ps until the model reaches the requested residency state."""
    deadline: float = time.monotonic() + self.config.llm_state_timeout_seconds

    while True:
      state: JsonObject = self.state()
      if state["loaded"] == loaded:
        return state

      if time.monotonic() >= deadline:
        raise RuntimeError(f"Model did not become {'loaded' if loaded else 'unloaded'}: {state}")

      time.sleep(0.5)

  def generate_empty(self, keep_alive: str | int) -> JsonObject:
    """Ask Ollama to load or unload the model with an empty generate request."""
    return post_json(
      self.config.llm_url + "/api/generate",
      {"model": self.config.llm_model, "stream": False, "keep_alive": keep_alive},
      self.config.llm_load_timeout_seconds,
    )

  def prepare(self, before: JsonObject) -> JsonObject:
    """Measure an explicit cold load or ensure a warm model before vLLM starts.

    The vLLM probe and warmups follow this step, so its main serving run is warm
    even when the separate Ollama load measurement uses cold mode.
    """
    if self.config.llm_start_mode == "cold":
      if before["loaded"]:
        self.generate_empty(0)

      unloaded: JsonObject = self.wait_for(False)
      print("Model verified unloaded; measuring cold load", flush=True)
      started_epoch: float = now()
      started_monotonic: float = time.perf_counter()
      response: JsonObject = self.generate_empty(self.config.llm_keep_alive)
      finished_epoch: float = now()
      duration: float = time.perf_counter() - started_monotonic
      loaded: JsonObject = self.wait_for(True)

      return {
        "state_after_unload": unloaded,
        "model_loaded_before_cold_request": False,
        "measurement_start_epoch": started_epoch,
        "cold_load_start_epoch": started_epoch,
        "cold_load_end_epoch": finished_epoch,
        "cold_load_wall_seconds": duration,
        "cold_load_server_duration_ns": response.get("load_duration"),
        "cold_load_response": response,
        "state_before_benchmark": loaded,
      }

    # Refresh residency outside the measured warm-run power window.
    self.generate_empty(self.config.llm_keep_alive)
    loaded = self.wait_for(True)
    print("Model verified loaded; starting warm benchmark", flush=True)
    return {"measurement_start_epoch": now(), "state_before_benchmark": loaded}


def run(runConfig: Config, modelSpec: ModelSpec) -> int:
  """Run one model test, recording benchmark output and corresponding power."""
  llmConfig: LLMConfig = runConfig.llm
  prometheusConfig: PrometheusConfig = runConfig.prometheus
  vllmConfig: VLLMConfig = runConfig.vllm
  benchConfig: BenchConfig = runConfig.bench
  installedModel: JsonObject = check_models_available(llmConfig.llm_url, [modelSpec.name])[modelSpec.name]
  quantization: str | None
  quantizationSource: str
  quantization, quantizationSource = detect_quantization(llmConfig.llm_url, modelSpec.name, installedModel)
  verify_quantization(modelSpec, installedModel, quantization)

  prometheusClient: PrometheusClient = PrometheusClient(prometheusConfig)
  preflightCheckedAtEpoch: float = now()
  preflightSamples: dict[str, float] = prometheusClient.check_freshness(preflightCheckedAtEpoch)
  ollamaController: OllamaController = OllamaController(llmConfig)
  modelStateBeforePreparation: JsonObject = ollamaController.state()

  slug: str = path_label(llmConfig.llm_model)
  run_id: str = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
  run_dir: Path = benchConfig.bench_results_dir / slug / run_folder_name(runConfig, quantization, run_id)
  run_dir.mkdir(parents=True, exist_ok=False)
  vllmCommand: list[str] = vllmConfig.command(llmConfig, run_dir)
  record: JsonObject = {
    "schema_version": 3,
    "run_id": run_id,
    "run_folder": run_dir.name,
    "status": "running",
    "model": vllmConfig.vllm_model,
    "served_model": llmConfig.llm_model,
    "tokenizer": vllmConfig.vllm_tokenizer,
    "model_spec": modelSpec.model_dump(),
    "ollama_installed_model": installedModel,
    "model_quantization": quantization,
    "model_quantization_source": quantizationSource,
    "deployment": benchConfig.bench_deployment,
    "client_host": socket.gethostname(),
    "hardware": benchConfig.bench_hardware,
    "campaign_id": benchConfig.bench_campaign_id,
    "replicate": benchConfig.bench_replicate,
    "scenario": benchConfig.bench_scenario,
    "llm_url": llmConfig.llm_url,
    "llm_backend": llmConfig.llm_backend,
    "prometheus_url": prometheusConfig.prometheus_url,
    "power_preflight_checked_at_epoch": preflightCheckedAtEpoch,
    "power_preflight_latest_raw_samples": preflightSamples,
    "model_before_preparation": modelStateBeforePreparation,
    "start_mode": llmConfig.llm_start_mode,
    "llm_keep_alive": llmConfig.llm_keep_alive,
    "main_benchmark_expected_warm": True,
    "cold_start_note": "Cold mode measures an explicit empty Ollama load separately. The vLLM main run follows its own probe and warmups, so its serving metrics are warm.",
    "power_sensors": list(prometheusConfig.prometheus_sensors),
    "power_step_seconds": prometheusConfig.prometheus_step_seconds,
    "power_pre_padding_seconds": prometheusConfig.prometheus_pre_padding_seconds,
    "power_post_padding_seconds": prometheusConfig.prometheus_post_padding_seconds,
    "power_settle_seconds": prometheusConfig.prometheus_settle_seconds,
    "command": vllmCommand,
    "files": {
      "benchmark": "benchmark.json",
      "power": "shelly-power.json",
      "power_raw": "shelly-power-raw.json",
      "log": "benchmark.log",
    },
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
  }
  save_json(run_dir / "run.json", record)
  print(f"Run directory: {run_dir}", flush=True)
  print(f"Start mode: {llmConfig.llm_start_mode}; model initially loaded: {modelStateBeforePreparation['loaded']}", flush=True)

  try:
    preparation: JsonObject = ollamaController.prepare(modelStateBeforePreparation)
    record["model_preparation"] = preparation
    save_json(run_dir / "run.json", record)

    status: int
    started: float
    finished: float
    markers: dict[str, float]
    print(f"Running vLLM benchmark; full output: {run_dir / 'benchmark.log'}", flush=True)
    status, started, finished, markers = BenchmarkLogger(run_dir / "benchmark.log", vllmConfig.vllm_show_output).run(vllmCommand)
    record.update(
      {
        "benchmark_exit_code": status,
        "invocation_start_epoch": started,
        "invocation_end_epoch": finished,
        "markers": markers,
      }
    )

    if status != 0:
      record["status"] = "benchmark_failed"
    elif not (run_dir / "benchmark.json").is_file():
      record["status"] = "benchmark_output_missing"
    else:
      result: JsonObject = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
      record["completed"] = result.get("completed")
      record["failed"] = result.get("failed")
      record["status"] = "complete" if result.get("completed", 0) > 0 and result.get("failed") == 0 else "benchmark_requests_failed"

    save_json(run_dir / "run.json", record)

    # Wait for new scrapes, then retrieve the bounded measurement window.
    time.sleep(max(0.0, finished + prometheusConfig.prometheus_settle_seconds - now()))
    power_start: float = float(preparation["measurement_start_epoch"]) - prometheusConfig.prometheus_pre_padding_seconds
    power_end: float = finished + prometheusConfig.prometheus_post_padding_seconds
    power: JsonObject
    details: JsonObject
    power, details = prometheusClient.fetch_power(power_start, power_end)
    save_json(run_dir / "shelly-power.json", power)
    record["power_query"] = details

    raw_power: JsonObject
    raw_details: JsonObject
    raw_power, raw_details = prometheusClient.fetch_raw_power(power_start, power_end)
    save_json(run_dir / "shelly-power-raw.json", raw_power)
    record["power_raw_query"] = raw_details

    if any(count == 0 for count in details["sample_counts"].values()):
      record["status"] = "power_samples_missing"

    if any(count == 0 for count in raw_details["sample_counts"].values()):
      record["status"] = "power_samples_missing"

    try:
      record["power_latest_raw_samples"] = prometheusClient.check_freshness(now(), finished)
    except RuntimeError as error:
      record["power_validation_error"] = str(error)
      record["status"] = "power_samples_incomplete"

    try:
      record["model_after"] = ollamaController.state()
    except Exception as error:
      record["model_after_error"] = str(error)

    save_json(run_dir / "run.json", record)
    print(f"Run status: {record['status']} ({run_dir})")
    return 0 if record["status"] == "complete" else 1

  except BaseException as error:
    record["status"] = "collection_failed"
    record["error"] = str(error)
    save_json(run_dir / "run.json", record)
    raise


def installed_models(llm_url: str) -> dict[str, JsonObject]:
  """Return installed Ollama tags with their size, digest and quantization."""
  response: JsonObject = get_json(llm_url + "/api/tags")
  models: object = response.get("models")

  if not isinstance(models, list):
    raise RuntimeError("Ollama /api/tags did not return a models list")

  return {str(model.get("name") or model.get("model")): cast(JsonObject, model) for model in models if isinstance(model, dict) and (model.get("name") or model.get("model"))}


def check_models_available(llm_url: str, names: Sequence[str]) -> dict[str, JsonObject]:
  """Fail before a benchmark if a selected model is absent from Ollama."""
  available: dict[str, JsonObject] = installed_models(llm_url)
  missing: set[str] = set(names) - available.keys()

  if missing:
    raise RuntimeError(f"Missing Ollama models: {', '.join(sorted(missing))}")

  return available


def verify_quantization(
  modelSpec: ModelSpec,
  installedModel: JsonObject,
  detected: str | None = None,
) -> None:
  """Check an explicitly pinned quantization against detected Ollama metadata."""
  if modelSpec.quantization is None:
    return

  actual: str | None = detected or installed_quantization(installedModel)
  if actual != modelSpec.quantization:
    raise RuntimeError(f"{modelSpec.name} quantization is {actual!r}; expected {modelSpec.quantization!r} in metadata/models.json")


def campaign_case_config(
  template: Config,
  modelSpec: ModelSpec,
  mode: Literal["cold", "warm"],
  prompts: int,
  rate: float,
  concurrency: int,
  input_len: int,
  output_len: int,
  seed: int,
  replicate: int,
  scenario: str,
  campaign_id: str,
) -> Config:
  """Create one validated campaign case from shared flags and model metadata."""
  llmConfig: LLMConfig = LLMConfig.model_validate(
    {
      **template.llm.model_dump(),
      "llm_model": modelSpec.name,
      "llm_start_mode": mode,
    }
  )
  vllmConfig: VLLMConfig = VLLMConfig.model_validate(
    {
      **template.vllm.model_dump(),
      "vllm_model": modelSpec.vllm_model,
      "vllm_tokenizer": modelSpec.tokenizer,
      "vllm_num_prompts": prompts,
      "vllm_request_rate": rate,
      "vllm_max_concurrency": concurrency,
      "vllm_random_input_len": input_len,
      "vllm_random_output_len": output_len,
      "vllm_seed": seed,
    }
  )
  benchConfig: BenchConfig = BenchConfig.model_validate(
    {
      **template.bench.model_dump(),
      "bench_campaign_id": campaign_id,
      "bench_replicate": replicate,
      "bench_scenario": scenario,
    }
  )
  return Config(llm=llmConfig, prometheus=template.prometheus, vllm=vllmConfig, bench=benchConfig)


def run_campaign(
  registry: ModelRegistry,
  template: Config,
  campaignConfig: CampaignConfig,
  dry_run: bool = False,
) -> int:
  """Plan a Cartesian workload grid, then run it one model block at a time."""
  names: tuple[str, ...] = campaignConfig.bench_model_order or tuple(registry.by_name())
  selectedModels: dict[str, ModelSpec] = {name: registry.require(name) for name in names}
  campaign_id: str = template.bench.bench_campaign_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
  profiles: tuple[WorkloadProfile, ...] = tuple(
    itertools.product(
      campaignConfig.bench_request_rates,
      campaignConfig.bench_max_concurrencies,
      campaignConfig.bench_input_lengths,
      campaignConfig.bench_output_lengths,
    )
  )
  blocks: list[tuple[str, list[Config]]] = []

  for name, modelSpec in selectedModels.items():
    cases: list[Config] = []

    for replicate in range(1, campaignConfig.bench_repetitions + 1):
      for index in range(len(profiles)):
        profile_index: int = (index + replicate - 1) % len(profiles)
        rate, concurrency, input_len, output_len = profiles[profile_index]
        scenario: str = f"warm-i{input_len}-o{output_len}-rps-{rate:g}-c{concurrency}"
        cases.append(
          campaign_case_config(
            template,
            modelSpec,
            "warm",
            campaignConfig.bench_prompts,
            rate,
            concurrency,
            input_len,
            output_len,
            template.vllm.vllm_seed + (replicate - 1) * len(profiles) + profile_index,
            replicate,
            scenario,
            campaign_id,
          )
        )

    if campaignConfig.bench_include_cold:
      for replicate in range(1, campaignConfig.bench_repetitions + 1):
        cases.append(
          campaign_case_config(
            template,
            modelSpec,
            "cold",
            campaignConfig.bench_cold_prompts,
            campaignConfig.bench_cold_request_rate,
            campaignConfig.bench_cold_concurrency,
            campaignConfig.bench_cold_input_len,
            campaignConfig.bench_cold_output_len,
            template.vllm.vllm_seed + campaignConfig.bench_repetitions * len(profiles) + replicate - 1,
            replicate,
            "cold-load",
            campaign_id,
          )
        )

    blocks.append((name, cases))

  warm_runs: int = len(names) * campaignConfig.bench_repetitions * len(profiles)
  cold_runs: int = len(names) * campaignConfig.bench_repetitions if campaignConfig.bench_include_cold else 0
  print(f"Campaign: {campaign_id} ({warm_runs} warm + {cold_runs} cold = {warm_runs + cold_runs} cases)")
  print(f"Model order: {', '.join(names)}; {len(profiles)} workload profiles x {campaignConfig.bench_repetitions} repetitions")

  if dry_run:
    for rate, concurrency, input_len, output_len in profiles:
      print(f"  warm-i{input_len}-o{output_len}-rps-{rate:g}-c{concurrency}")

    if campaignConfig.bench_include_cold:
      print(f"  cold-load: {campaignConfig.bench_cold_prompts} requests, {campaignConfig.bench_cold_input_len}/{campaignConfig.bench_cold_output_len} tokens, {campaignConfig.bench_cold_request_rate:g} RPS, concurrency {campaignConfig.bench_cold_concurrency}")

    return 0

  first_case: Config = blocks[0][1][0]
  first_case.vllm.find_bin()
  available: dict[str, JsonObject] = check_models_available(first_case.llm.llm_url, names)
  for name, modelSpec in selectedModels.items():
    if modelSpec.quantization is not None:
      detected, _ = detect_quantization(first_case.llm.llm_url, name, available[name])
      verify_quantization(modelSpec, available[name], detected)

  completed_runs: int = 0
  for name, cases in blocks:
    ollamaController: OllamaController = OllamaController(cases[0].llm)
    ollamaController.unload_other_models(registry.by_name())

    for runConfig in cases:
      if completed_runs:
        time.sleep(campaignConfig.bench_rest_seconds)

      print(
        f"{campaign_id} | {name} | replicate {runConfig.bench.bench_replicate} | {runConfig.bench.bench_scenario}",
        flush=True,
      )
      result: int = run(runConfig, selectedModels[name])
      if result != 0:
        print(f"Campaign stopped after {completed_runs} complete runs", file=sys.stderr)
        return result

      completed_runs += 1

  print(f"Campaign complete: {campaign_id} ({completed_runs} runs)")
  return 0


def list_models(registry: ModelRegistry, live: bool, llm_url: str) -> int:
  """Show configured prompt identifiers and optionally inspect installed tags."""
  catalog: dict[str, ModelSpec] = registry.by_name()
  available: dict[str, JsonObject] = {}

  if live:
    llm_url = LLMConfig.validate_url(llm_url)
    available = installed_models(llm_url)
    print(f"Ollama server: {llm_url}\n")

  for modelSpec in catalog.values():
    print(f"{modelSpec.name} — {modelSpec.description or 'Benchmark model'}")
    print(f"  vLLM model:            {modelSpec.vllm_model}")
    print(f"  tokenizer:             {modelSpec.tokenizer}")
    print(f"  expected quantization: {modelSpec.quantization or 'not pinned'}")

    if live:
      installedModel: JsonObject | None = available.get(modelSpec.name)
      if installedModel is None:
        print("  installed:             no")
      else:
        raw_details: object = installedModel.get("details")
        details: JsonObject = cast(JsonObject, raw_details) if isinstance(raw_details, dict) else {}
        size: object = installedModel.get("size")
        size_text: str = f"{size / 1_000_000_000:.2f} GB" if isinstance(size, (int, float)) else "unknown"
        print("  installed:             yes")
        quantization, source = detect_quantization(llm_url, modelSpec.name, installedModel)
        print(f"  actual quantization:   {quantization or 'unknown'} ({source})")
        print(f"  parameters:            {details.get('parameter_size', 'unknown')}")
        print(f"  stored size:           {size_text}")
        print(f"  digest:                {installedModel.get('digest', 'unknown')}")
        verify_quantization(modelSpec, installedModel, quantization)

    print()

  return 0


class StartMode(str, Enum):
  """Choose whether an individual run measures an explicit cold load."""

  warm = "warm"
  cold = "cold"


app: typer.Typer = typer.Typer(
  no_args_is_help=True,
  add_completion=False,
  pretty_exceptions_enable=False,
  help="Benchmark Ollama serving and collect Shelly power from Prometheus.",
)


def execute(operation: Callable[[], int]) -> None:
  """Turn validation and API failures into concise CLI errors and exit codes."""
  try:
    result: int = operation()
  except (ValidationError, ValueError, RuntimeError, urllib.error.URLError) as error:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error

  if result != 0:
    raise typer.Exit(code=result)


@app.command("ls")
def ls_command(
  models_file: Annotated[Path, typer.Option(help="JSON catalog path, relative to the project root.")] = Path("metadata/models.json"),
  live: Annotated[bool, typer.Option("--live", help="Query Ollama for installed quantization, size and digest.")] = False,
  llm_url: Annotated[str, typer.Option(help="Ollama API URL used with --live.")] = "http://localhost:11434",
) -> None:
  """List the registered models and their optional live Ollama specifications."""
  execute(lambda: list_models(load_registry(models_file), live, llm_url))


@app.command("benchmark")
def benchmark_command(
  model: Annotated[str, typer.Option(help="Exact Ollama tag from the model catalog.")],
  llm_url: Annotated[str, typer.Option(help="Ollama API base URL, including port.")],
  prometheus_url: Annotated[str, typer.Option(help="Prometheus HTTP API base URL.")],
  deployment: Annotated[str, typer.Option(help="Deployment label, e.g. docker or kubernetes.")],
  hardware: Annotated[str, typer.Option(help="Hardware label, e.g. gtx-1070.")],
  models_file: Annotated[Path, typer.Option(help="JSON catalog path relative to the project root.")] = Path("metadata/models.json"),
  start_mode: Annotated[StartMode, typer.Option(help="Cold measures a separate load before warm serving metrics.")] = StartMode.warm,
  keep_alive: Annotated[str, typer.Option(help="Ollama model residency period.")] = "30m",
  load_timeout: Annotated[int, typer.Option(help="Ollama load timeout in seconds.")] = 300,
  state_timeout: Annotated[int, typer.Option(help="Model state polling timeout in seconds.")] = 30,
  sensor: Annotated[list[str] | None, typer.Option("--sensor", help="Required Shelly label; repeat for each plug.")] = None,
  power_step: Annotated[int, typer.Option(help="PromQL query step in seconds; raw scrapes are saved separately.")] = 10,
  power_pre_padding: Annotated[int, typer.Option(help="Seconds of power data before model preparation or serving.")] = 10,
  power_post_padding: Annotated[int, typer.Option(help="Seconds of power data after serving.")] = 10,
  power_settle: Annotated[int, typer.Option(help="Wait for post-run scrapes, in seconds.")] = 20,
  power_preflight: Annotated[int, typer.Option(help="Lookback for sensor freshness, in seconds.")] = 600,
  power_max_sample_age: Annotated[int, typer.Option(help="Maximum age of the latest sensor scrape, in seconds.")] = 30,
  vllm_bin: Annotated[Path | None, typer.Option(help="vLLM executable; otherwise find it on PATH or in vllm-bench/.venv.")] = None,
  num_prompts: Annotated[int, typer.Option(help="Measured requests, excluding probe and warmups.")] = 10,
  request_rate: Annotated[float, typer.Option(help="Average Poisson request arrivals per second.")] = 0.5,
  max_concurrency: Annotated[int, typer.Option(help="Maximum simultaneous requests.")] = 1,
  num_warmups: Annotated[int, typer.Option(help="Unmeasured warmup requests.")] = 2,
  input_len: Annotated[int, typer.Option(help="Target random input tokens.")] = 512,
  output_len: Annotated[int, typer.Option(help="Target and cap output tokens.")] = 128,
  random_range_ratio: Annotated[float, typer.Option(help="Random prompt length variation; 0 fixes the target.")] = 0.0,
  temperature: Annotated[float, typer.Option(help="Decoding temperature; 0 is deterministic.")] = 0.0,
  seed: Annotated[int, typer.Option(help="Random prompt and arrival seed.")] = 42,
  show_output: Annotated[bool, typer.Option("--show-output", help="Mirror vLLM output to the terminal.")] = False,
  results_dir: Annotated[Path, typer.Option(help="Parent directory for model/run folders.")] = Path("results"),
  campaign_id: Annotated[str | None, typer.Option(help="Optional experiment identifier on this run.")] = None,
  replicate: Annotated[int | None, typer.Option(help="Optional repetition number on this run.")] = None,
  scenario: Annotated[str | None, typer.Option(help="Optional workload label on this run.")] = None,
) -> None:
  """Run one catalog model and save its serving and power measurements."""

  def benchmark_once() -> int:
    registry: ModelRegistry = load_registry(models_file)
    modelSpec: ModelSpec = registry.require(model)
    runConfig: Config = Config(
      llm=LLMConfig(
        llm_url=llm_url,
        llm_model=modelSpec.name,
        llm_start_mode=start_mode.value,
        llm_keep_alive=keep_alive,
        llm_load_timeout_seconds=load_timeout,
        llm_state_timeout_seconds=state_timeout,
      ),
      prometheus=PrometheusConfig(
        prometheus_url=prometheus_url,
        prometheus_sensors=tuple(sensor) if sensor is not None else ("shelly-gpu", "shelly-gpu-node"),
        prometheus_step_seconds=power_step,
        prometheus_pre_padding_seconds=power_pre_padding,
        prometheus_post_padding_seconds=power_post_padding,
        prometheus_settle_seconds=power_settle,
        prometheus_preflight_seconds=power_preflight,
        prometheus_max_sample_age_seconds=power_max_sample_age,
      ),
      vllm=VLLMConfig(
        vllm_bin=str(vllm_bin) if vllm_bin is not None else None,
        vllm_model=modelSpec.vllm_model,
        vllm_tokenizer=modelSpec.tokenizer,
        vllm_num_prompts=num_prompts,
        vllm_request_rate=request_rate,
        vllm_max_concurrency=max_concurrency,
        vllm_num_warmups=num_warmups,
        vllm_random_input_len=input_len,
        vllm_random_output_len=output_len,
        vllm_random_range_ratio=random_range_ratio,
        vllm_temperature=temperature,
        vllm_seed=seed,
        vllm_show_output=show_output,
      ),
      bench=BenchConfig(
        bench_deployment=deployment,
        bench_hardware=hardware,
        bench_results_dir=results_dir,
        bench_campaign_id=campaign_id,
        bench_replicate=replicate,
        bench_scenario=scenario,
      ),
    )
    runConfig.vllm.find_bin()
    return run(runConfig, modelSpec)

  execute(benchmark_once)


@app.command("campaign")
def campaign_command(
  llm_url: Annotated[str, typer.Option(help="Ollama API base URL, including port.")],
  prometheus_url: Annotated[str, typer.Option(help="Prometheus HTTP API base URL.")],
  deployment: Annotated[str, typer.Option(help="Deployment label, e.g. docker or kubernetes.")],
  hardware: Annotated[str, typer.Option(help="Hardware label, e.g. gtx-1070.")],
  model: Annotated[list[str] | None, typer.Option("--model", help="Model block in order; repeat to select more. Defaults to the catalog order.")] = None,
  models_file: Annotated[Path, typer.Option(help="JSON catalog path relative to the project root.")] = Path("metadata/models.json"),
  repetitions: Annotated[int, typer.Option(help="Repetitions of every workload profile per model.")] = 5,
  prompts: Annotated[int, typer.Option(help="Measured requests in each warm case.")] = 120,
  request_rate: Annotated[list[float] | None, typer.Option("--request-rate", help="Warm requests per second; repeat for more rates. Default: 0.25, 0.5, 1.0.")] = None,
  max_concurrency: Annotated[list[int] | None, typer.Option("--max-concurrency", help="Warm concurrency limit; repeat to cross with rates and lengths. Default: 4.")] = None,
  rest_seconds: Annotated[int, typer.Option(help="Pause between cases without unloading the model.")] = 30,
  include_cold: Annotated[bool, typer.Option("--include-cold", help="Run separate cold-load cases after each warm model block.")] = False,
  cold_prompts: Annotated[int, typer.Option(help="Measured requests in each cold case.")] = 10,
  cold_request_rate: Annotated[float, typer.Option(help="Cold case request arrivals per second.")] = 0.5,
  cold_concurrency: Annotated[int, typer.Option(help="Cold case maximum simultaneous requests.")] = 1,
  cold_input_len: Annotated[int, typer.Option(help="Input tokens for separate cold-load cases; default 512.")] = 512,
  cold_output_len: Annotated[int, typer.Option(help="Output target/cap for separate cold-load cases; default 128.")] = 128,
  keep_alive: Annotated[str, typer.Option(help="Ollama model residency period.")] = "30m",
  load_timeout: Annotated[int, typer.Option(help="Ollama load timeout in seconds.")] = 300,
  state_timeout: Annotated[int, typer.Option(help="Model state polling timeout in seconds.")] = 30,
  sensor: Annotated[list[str] | None, typer.Option("--sensor", help="Required Shelly label; repeat for each plug.")] = None,
  power_step: Annotated[int, typer.Option(help="PromQL query step in seconds; raw scrapes are saved separately.")] = 10,
  power_pre_padding: Annotated[int, typer.Option(help="Seconds of power data before model preparation or serving.")] = 10,
  power_post_padding: Annotated[int, typer.Option(help="Seconds of power data after serving.")] = 10,
  power_settle: Annotated[int, typer.Option(help="Wait for post-run scrapes, in seconds.")] = 20,
  power_preflight: Annotated[int, typer.Option(help="Lookback for sensor freshness, in seconds.")] = 600,
  power_max_sample_age: Annotated[int, typer.Option(help="Maximum age of the latest sensor scrape, in seconds.")] = 30,
  vllm_bin: Annotated[Path | None, typer.Option(help="vLLM executable; otherwise find it on PATH or in vllm-bench/.venv.")] = None,
  num_warmups: Annotated[int, typer.Option(help="Unmeasured warmup requests in each case.")] = 2,
  input_len: Annotated[list[int] | None, typer.Option("--input-len", help="Warm input tokens; repeat to cross with rates, concurrency, output lengths. Default: 512.")] = None,
  output_len: Annotated[list[int] | None, typer.Option("--output-len", help="Warm output target/cap; repeat for more lengths. Default: 128.")] = None,
  random_range_ratio: Annotated[float, typer.Option(help="Random prompt length variation; 0 fixes the target.")] = 0.0,
  temperature: Annotated[float, typer.Option(help="Decoding temperature; 0 is deterministic.")] = 0.0,
  seed: Annotated[int, typer.Option(help="Random prompt and arrival seed.")] = 42,
  show_output: Annotated[bool, typer.Option("--show-output", help="Mirror vLLM output to the terminal.")] = False,
  results_dir: Annotated[Path, typer.Option(help="Parent directory for model/run folders.")] = Path("results"),
  campaign_id: Annotated[str | None, typer.Option(help="Optional stable identifier for the whole campaign.")] = None,
  dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the full grid and case count without network access or writing results.")] = False,
) -> None:
  """Run every rate × concurrency × input × output profile per model."""

  def campaign_once() -> int:
    registry: ModelRegistry = load_registry(models_file)
    campaignConfig: CampaignConfig = CampaignConfig(
      bench_model_order=tuple(model) if model is not None else None,
      bench_repetitions=repetitions,
      bench_prompts=prompts,
      bench_request_rates=tuple(request_rate) if request_rate is not None else (0.25, 0.5, 1.0),
      bench_max_concurrencies=tuple(max_concurrency) if max_concurrency is not None else (4,),
      bench_input_lengths=tuple(input_len) if input_len is not None else (512,),
      bench_output_lengths=tuple(output_len) if output_len is not None else (128,),
      bench_rest_seconds=rest_seconds,
      bench_include_cold=include_cold,
      bench_cold_prompts=cold_prompts,
      bench_cold_request_rate=cold_request_rate,
      bench_cold_concurrency=cold_concurrency,
      bench_cold_input_len=cold_input_len,
      bench_cold_output_len=cold_output_len,
    )
    first_name: str = (campaignConfig.bench_model_order or (registry.models[0].name,))[0]
    firstModel: ModelSpec = registry.require(first_name)
    first_rate: float = campaignConfig.bench_request_rates[0]
    template: Config = Config(
      llm=LLMConfig(
        llm_url=llm_url,
        llm_model=firstModel.name,
        llm_start_mode="warm",
        llm_keep_alive=keep_alive,
        llm_load_timeout_seconds=load_timeout,
        llm_state_timeout_seconds=state_timeout,
      ),
      prometheus=PrometheusConfig(
        prometheus_url=prometheus_url,
        prometheus_sensors=tuple(sensor) if sensor is not None else ("shelly-gpu", "shelly-gpu-node"),
        prometheus_step_seconds=power_step,
        prometheus_pre_padding_seconds=power_pre_padding,
        prometheus_post_padding_seconds=power_post_padding,
        prometheus_settle_seconds=power_settle,
        prometheus_preflight_seconds=power_preflight,
        prometheus_max_sample_age_seconds=power_max_sample_age,
      ),
      vllm=VLLMConfig(
        vllm_bin=str(vllm_bin) if vllm_bin is not None else None,
        vllm_model=firstModel.vllm_model,
        vllm_tokenizer=firstModel.tokenizer,
        vllm_num_prompts=prompts,
        vllm_request_rate=first_rate,
        vllm_max_concurrency=campaignConfig.bench_max_concurrencies[0],
        vllm_num_warmups=num_warmups,
        vllm_random_input_len=campaignConfig.bench_input_lengths[0],
        vllm_random_output_len=campaignConfig.bench_output_lengths[0],
        vllm_random_range_ratio=random_range_ratio,
        vllm_temperature=temperature,
        vllm_seed=seed,
        vllm_show_output=show_output,
      ),
      bench=BenchConfig(
        bench_deployment=deployment,
        bench_hardware=hardware,
        bench_results_dir=results_dir,
        bench_campaign_id=campaign_id,
      ),
    )
    return run_campaign(registry, template, campaignConfig, dry_run=dry_run)

  execute(campaign_once)


if __name__ == "__main__":
  app()
