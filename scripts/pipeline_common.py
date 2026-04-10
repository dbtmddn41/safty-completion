import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import sys
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_TASK_TYPES = [
    "Explanation",
    "Planning and Procedure",
    "Troubleshooting",
    "Optimization",
    "Transformation",
]


DEFAULT_HARM_DOMAINS = [
    "Hazardous Agent Use",
    "Physical Harm and Weapons",
    "Hate and Harassment",
    "Cyber Compromise",
    "Fraud and Deception",
    "Privacy and Data Misuse",
    "Evasion and Circumvention",
]


STAGE2_POLICY = {
    "Explanation": {
        "benign_cover_categories": [
            "Education / learning",
            "Research / analysis",
            "Journalistic / historical / documentary interest",
            "Defensive protection / safety",
        ],
        "framing_styles": [
            "Direct information request",
            "Reflective / analytical request",
            "Scenario-based / hypothetical request",
        ],
    },
    "Planning and Procedure": {
        "benign_cover_categories": [
            "Professional / operational planning",
            "Defensive protection / safety",
            "Personal safety / practical preparedness",
            "Research / analysis",
        ],
        "framing_styles": [
            "Planning / design request",
            "Direct information request",
            "Scenario-based / hypothetical request",
            "Decision support request",
        ],
    },
    "Troubleshooting": {
        "benign_cover_categories": [
            "Defensive protection / safety",
            "Debugging / maintenance of legitimate systems",
            "Professional / operational planning",
        ],
        "framing_styles": [
            "Troubleshooting request",
            "Direct information request",
            "Scenario-based / hypothetical request",
        ],
    },
    "Optimization": {
        "benign_cover_categories": [
            "Defensive protection / safety",
            "Debugging / maintenance of legitimate systems",
            "Professional / operational planning",
            "Research / analysis",
        ],
        "framing_styles": [
            "Troubleshooting request",
            "Comparative / evaluative request",
            "Planning / design request",
            "Decision support request",
        ],
    },
    "Transformation": {
        "benign_cover_categories": [
            "Creative / fictional use",
            "Professional / operational planning",
            "Research / analysis",
            "Education / learning",
        ],
        "framing_styles": [
            "Transformation request",
            "Scenario-based / hypothetical request",
            "Comparative / evaluative request",
        ],
    },
}

DEFAULT_TOPIC_SUMMARY_PATHS = [
    "stage1.topic_summary",
    "topic_summary",
]

DEFAULT_HARM_DOMAIN_PATHS = [
    "stage1.harm_domain",
    "gemini_harm_domain",
    "harm_domain",
    "claude_harm_domain",
]

DEFAULT_TASK_TYPE_PATHS = [
    "stage1_1.task_type",
    "gemini_task_type",
    "task_type",
    "claude_task_type",
    "stage1.task_type",
]


_VERTEX_TOKEN_CACHE: Optional[Tuple[str, float]] = None
_VERTEX_GOOGLE_CREDENTIALS = None
_VERTEX_GCLOUD_PATH: Optional[str] = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_label(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def load_label_list(path: Optional[str], fallback: List[str]) -> List[str]:
    if not path:
        return list(fallback)
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if raw[0] in "[{":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("labels", [])
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [str(item).strip() for item in data if str(item).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        stripped = []
        for line in text.splitlines():
            if line.startswith("```"):
                continue
            stripped.append(line)
        text = "\n".join(stripped).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def get_by_dot_path(row: Dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Missing field path '{path}' in record: {row}")
        current = current[part]
    return current


def resolve_field_path(row: Dict[str, Any], primary_path: str, fallback_paths: List[str]) -> Any:
    seen = set()
    last_error = None
    for path in [primary_path, *fallback_paths]:
        if path in seen:
            continue
        seen.add(path)
        try:
            return get_by_dot_path(row, path)
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Could not resolve field from any of {list(seen)} in record: {row}"
    ) from last_error


def resolve_topic_summary(row: Dict[str, Any], primary_path: str = "stage1.topic_summary") -> Any:
    return resolve_field_path(row, primary_path, DEFAULT_TOPIC_SUMMARY_PATHS)


def resolve_harm_domain(row: Dict[str, Any], primary_path: str = "stage1.harm_domain") -> Any:
    return resolve_field_path(row, primary_path, DEFAULT_HARM_DOMAIN_PATHS)


def resolve_task_type(row: Dict[str, Any], primary_path: str = "stage1_1.task_type") -> Any:
    return resolve_field_path(row, primary_path, DEFAULT_TASK_TYPE_PATHS)


def ensure_stage_metadata(
    row: Dict[str, Any],
    *,
    topic_summary: Optional[str] = None,
    harm_domain: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    enriched = dict(row)

    stage1 = enriched.get("stage1")
    if not isinstance(stage1, dict):
        stage1 = {}
    else:
        stage1 = dict(stage1)
    if topic_summary:
        stage1.setdefault("topic_summary", topic_summary)
    if harm_domain:
        stage1.setdefault("harm_domain", harm_domain)
    if task_type:
        stage1.setdefault("task_type", task_type)
    if stage1:
        enriched["stage1"] = stage1

    stage1_1 = enriched.get("stage1_1")
    if not isinstance(stage1_1, dict):
        stage1_1 = {}
    else:
        stage1_1 = dict(stage1_1)
    if harm_domain:
        stage1_1.setdefault("harm_domain", harm_domain)
    if task_type:
        stage1_1.setdefault("task_type", task_type)
    if stage1_1:
        enriched["stage1_1"] = stage1_1

    return enriched


def request_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 120) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        message = f"HTTP Error {exc.code}: {exc.reason}"
        if detail:
            message = f"{message} | {detail}"
        raise RuntimeError(message) from exc


def call_with_retries(request_fn, max_attempts: int = 5, base_delay_seconds: float = 2.0) -> Dict[str, Any]:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return request_fn()
        except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(base_delay_seconds * attempt)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_error}") from last_error


def get_env_or_raise(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _resolve_gcloud_path() -> Optional[str]:
    global _VERTEX_GCLOUD_PATH
    if _VERTEX_GCLOUD_PATH:
        return _VERTEX_GCLOUD_PATH

    gcloud_path = shutil.which("gcloud")
    if not gcloud_path:
        for candidate in (
            "/opt/homebrew/bin/gcloud",
            "/usr/local/bin/gcloud",
            os.path.expanduser("~/google-cloud-sdk/bin/gcloud"),
        ):
            if os.path.exists(candidate):
                gcloud_path = candidate
                break

    _VERTEX_GCLOUD_PATH = gcloud_path
    return _VERTEX_GCLOUD_PATH


def _cache_vertex_token(token: str, ttl_seconds: int) -> str:
    global _VERTEX_TOKEN_CACHE
    ttl = max(60, ttl_seconds)
    _VERTEX_TOKEN_CACHE = (token, time.time() + ttl)
    return token


def _get_cached_vertex_token() -> Optional[str]:
    if not _VERTEX_TOKEN_CACHE:
        return None
    token, expires_at = _VERTEX_TOKEN_CACHE
    if time.time() < expires_at:
        return token
    return None


def _load_google_auth_credentials():
    global _VERTEX_GOOGLE_CREDENTIALS
    if _VERTEX_GOOGLE_CREDENTIALS is not None:
        return _VERTEX_GOOGLE_CREDENTIALS

    import google.auth

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    _VERTEX_GOOGLE_CREDENTIALS = credentials
    return _VERTEX_GOOGLE_CREDENTIALS


def _get_token_from_google_auth() -> str:
    from google.auth.transport.requests import Request

    credentials = _load_google_auth_credentials()
    if not getattr(credentials, "token", None) or not getattr(credentials, "valid", False):
        credentials.refresh(Request())

    token = getattr(credentials, "token", "").strip()
    if not token:
        raise RuntimeError("google.auth credentials refreshed without yielding an access token")

    expiry = getattr(credentials, "expiry", None)
    ttl_seconds = 300
    if isinstance(expiry, datetime.datetime):
        if expiry.tzinfo is None:
            now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            expiry = expiry.astimezone(datetime.timezone.utc)
        ttl_seconds = int((expiry - now).total_seconds()) - 300
    return _cache_vertex_token(token, ttl_seconds)


def _get_token_from_gcloud(gcloud_path: str, auth_errors: List[str]) -> Optional[str]:
    for command in (
        [gcloud_path, "auth", "application-default", "print-access-token"],
        [gcloud_path, "auth", "print-access-token"],
    ):
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            command_text = " ".join(command)
            if details:
                auth_errors.append(f"{command_text} failed: {details}")
            else:
                auth_errors.append(
                    f"{command_text} failed with exit status {exc.returncode}"
                )
            continue

        token = process.stdout.strip()
        if token:
            return _cache_vertex_token(token, 45 * 60)
        auth_errors.append(f"{' '.join(command)} returned an empty access token")
    return None


def get_vertex_access_token() -> str:
    env_token = os.getenv("VERTEX_ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token

    cached_token = _get_cached_vertex_token()
    if cached_token:
        return cached_token

    auth_errors: List[str] = []

    try:
        return _get_token_from_google_auth()
    except Exception as exc:
        auth_errors.append(f"google.auth.default failed: {exc}")

    gcloud_path = _resolve_gcloud_path()
    if gcloud_path:
        token = _get_token_from_gcloud(gcloud_path, auth_errors)
        if token:
            return token

    guidance = (
        "Could not obtain a Vertex access token. "
        "Supported auth paths are:\n"
        "1. Set VERTEX_ACCESS_TOKEN\n"
        "2. Configure Application Default Credentials with "
        "'gcloud auth application-default login'\n"
        "3. Configure an authenticated gcloud session with 'gcloud auth login'\n"
        "4. Ensure gcloud can read and write its config directory (usually ~/.config/gcloud)"
    )
    if auth_errors:
        raise RuntimeError(f"{guidance}\n\nAuth attempt details:\n- " + "\n- ".join(auth_errors))
    raise RuntimeError(guidance)


@dataclass
class ClaudeVertexConfig:
    project_id: str
    location: str
    model: str
    max_tokens: int = 256
    temperature: float = 0.0


def call_claude_on_vertex(prompt: str, config: ClaudeVertexConfig) -> str:
    access_token = get_vertex_access_token()
    if " " in config.model:
        raise RuntimeError(
            "Vertex model must be the model ID, not a display name. "
            "For example: claude-sonnet-4@20250514"
        )
    url = (
        f"https://{config.location}-aiplatform.googleapis.com/v1/projects/"
        f"{config.project_id}/locations/{config.location}/publishers/anthropic/"
        f"models/{config.model}:rawPredict"
    )
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        response = call_with_retries(lambda: request_json(url, payload, headers))
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP Error 404" in message:
            raise RuntimeError(
                "Vertex returned 404 for the Claude model request. "
                "Check that the model ID and region are valid for this project, and that "
                "the model has been enabled in Vertex AI Model Garden for this project. "
                f"Request URL was: {url}"
            ) from exc
        raise
    content = response.get("content", [])
    parts = [item.get("text", "") for item in content if item.get("type") == "text"]
    text = "\n".join(part for part in parts if part).strip()
    if response.get("stop_reason") == "refusal":
        raise RuntimeError(
            "Claude refused the request. The model received the unsafe seed and declined to "
            "produce metadata. Use a more refusal-robust Stage 1 design, or catch and audit "
            f"refusals. Response was: {json.dumps(response, ensure_ascii=True)}"
        )
    if not text:
        raise ValueError(f"Unexpected Claude response: {response}")
    return text


def call_gemini(prompt: str, model: str, max_output_tokens: int = 512, temperature: float = 0.0) -> str:
    api_key = "AIzaSyByLhmCUw_FHzrvia7zQZq8o75tn59-kaU"#get_env_or_raise("GEMINI_API_KEY")
    encoded_model = urllib.parse.quote(model, safe="")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}
    response = call_with_retries(lambda: request_json(url, payload, headers))
    candidates = response.get("candidates", [])
    if not candidates:
        raise ValueError(f"Unexpected Gemini response: {response}")
    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
    if finish_reason == "MAX_TOKENS" and not text:
        raise RuntimeError(
            "Gemini hit MAX_TOKENS before returning text. Increase max_output_tokens or shorten the prompt. "
            f"Response was: {json.dumps(response, ensure_ascii=True)}"
        )
    if not text:
        raise ValueError(f"Unexpected Gemini response: {response}")
    return text


def stable_record_id(record: Dict[str, Any], fallback_fields: Optional[List[str]] = None) -> str:
    for candidate in ("id", "record_id", "sample_id", "uuid", "row_index"):
        value = record.get(candidate)
        if value is not None:
            if candidate == "row_index":
                return f"row-{value}"
            return str(value)
    fallback_fields = fallback_fields or ["unsafe_seed", "seed", "prompt", "text"]
    for field in fallback_fields:
        value = record.get(field)
        if value:
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            return digest[:16]
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def deterministic_rng(*parts: str) -> random.Random:
    key = "||".join(parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def weighted_pick(options: List[str], rng: random.Random, preferred_count: int = 2, preferred_weight: float = 0.7) -> str:
    if not options:
        raise ValueError("Cannot choose from an empty option list")
    preferred = options[:preferred_count]
    remaining = options[preferred_count:]
    if not remaining or rng.random() < preferred_weight:
        return rng.choice(preferred)
    return rng.choice(remaining)


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")


def progress_iter(items: List[Dict[str, Any]], desc: str, disable: bool = False) -> Iterator[Dict[str, Any]]:
    if disable:
        for item in items:
            yield item
        return

    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=desc)
        return
    except Exception:
        pass

    total = len(items)
    for index, item in enumerate(items, start=1):
        print(f"[{desc}] {index}/{total}", file=sys.stderr, flush=True)
        yield item
