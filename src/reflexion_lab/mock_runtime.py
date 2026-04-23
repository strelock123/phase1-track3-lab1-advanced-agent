from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_user_prompt,
    build_evaluator_user_prompt,
    build_reflector_user_prompt,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class LLMCallResult:
    content: str
    usage: LLMUsage
    latency_ms: int
    raw: dict[str, Any] | None = None


def _backend() -> str:
    explicit = os.getenv("REFLEXION_LAB_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    if _api_key() or os.getenv("REFLEXION_LAB_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL"):
        return "openai_compatible"
    return "mock"


def _api_base_url() -> str:
    default_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("OPENROUTER_BASE_URL") else "https://api.openai.com/v1"
    return os.getenv(
        "REFLEXION_LAB_BASE_URL",
        os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", default_url)),
    ).rstrip("/")


def _api_key() -> str:
    return os.getenv(
        "REFLEXION_LAB_API_KEY",
        os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    ).strip()


def _provider_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", os.getenv("HTTP_REFERER", "")).strip()
    title = os.getenv("OPENROUTER_TITLE", os.getenv("X_OPENROUTER_TITLE", "")).strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _model_name(default: str) -> str:
    return os.getenv("REFLEXION_LAB_MODEL", default).strip()


def _default_model_name() -> str:
    return "openai/gpt-4o-mini" if _api_base_url().startswith("https://openrouter.ai") else "gpt-4o-mini"


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("REFLEXION_LAB_TIMEOUT_SECONDS", "90"))
    except ValueError:
        return 90.0


def _is_local_ollama() -> bool:
    base_url = _api_base_url()
    return "127.0.0.1:11434" in base_url or "localhost:11434" in base_url


def _ollama_root_url() -> str:
    base_url = _api_base_url()
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def _ollama_options(num_predict: int) -> dict[str, int]:
    thread_default = os.cpu_count() or 8
    try:
        num_thread = int(os.getenv("REFLEXION_LAB_OLLAMA_NUM_THREADS", str(thread_default)))
    except ValueError:
        num_thread = thread_default
    return {"num_gpu": 0, "num_thread": max(1, num_thread), "num_predict": max(1, num_predict)}


def _ollama_keep_alive() -> str:
    return os.getenv("REFLEXION_LAB_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"


def _ollama_light_mode() -> bool:
    return os.getenv("REFLEXION_LAB_OLLAMA_LIGHT_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_usage(payload: dict[str, Any] | None) -> LLMUsage:
    if not payload:
        return LLMUsage()
    return LLMUsage(
        prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
        completion_tokens=int(payload.get("completion_tokens", 0) or 0),
        total_tokens=int(payload.get("total_tokens", 0) or 0),
    )


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def _message_to_prompt(system: str, user: str) -> str:
    return f"{system.strip()}\n\nUser:\n{user.strip()}\n\nAssistant:"


def _ollama_generate(prompt: str, *, model: str, num_predict: int, json_mode: bool = False) -> LLMCallResult:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": _ollama_keep_alive(),
        "options": _ollama_options(num_predict),
    }
    if json_mode:
        payload["format"] = "json"

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{_ollama_root_url()}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail or exc.reason}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    payload = json.loads(raw_text)
    usage = LLMUsage(
        prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
        completion_tokens=int(payload.get("eval_count", 0) or 0),
        total_tokens=int(payload.get("prompt_eval_count", 0) or 0) + int(payload.get("eval_count", 0) or 0),
    )
    if payload.get("total_duration"):
        latency_ms = round(int(payload["total_duration"]) / 1_000_000)
    return LLMCallResult(content=str(payload.get("response", "")).strip(), usage=usage, latency_ms=latency_ms, raw=payload)


def _extract_json_block(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        loaded = json.loads(candidate)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_bool_score(score: Any) -> int:
    if isinstance(score, bool):
        return int(score)
    try:
        return 1 if int(score) >= 1 else 0
    except (TypeError, ValueError):
        return 0


def _http_chat_completion(messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> LLMCallResult:
    api_key = _api_key()
    if not api_key and not _api_base_url().startswith("http://localhost") and not _api_base_url().startswith("http://127.0.0.1"):
        raise RuntimeError(
            "Missing API key. Set REFLEXION_LAB_API_KEY or OPENAI_API_KEY, "
            "or point REFLEXION_LAB_BASE_URL to a local OpenAI-compatible server."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{_api_base_url()}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            **_provider_headers(),
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail or exc.reason}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    payload = json.loads(raw_text)
    return LLMCallResult(
        content=_extract_text(payload),
        usage=_parse_usage(payload.get("usage")),
        latency_ms=latency_ms,
        raw=payload,
    )


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.", final_answer=answer)
    if normalize_answer(answer) == "london":
        return JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
            final_answer=answer,
        )
    return JudgeResult(
        score=0,
        reason="The final answer selected the wrong second-hop entity.",
        missing_evidence=["Need to ground the answer in the second paragraph."],
        spurious_claims=[answer],
        final_answer=answer,
    )


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )


def generate_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> LLMCallResult:
    if _backend() == "mock":
        return LLMCallResult(
            content=_mock_actor_answer(example, attempt_id, agent_type, reflection_memory),
            usage=LLMUsage(),
            latency_ms=0,
            raw={"backend": "mock"},
        )
    if _is_local_ollama():
        prompt = _message_to_prompt(
            ACTOR_SYSTEM,
            build_actor_user_prompt(example.question, example.context, reflection_memory),
        )
        return _ollama_generate(prompt, model=_model_name("deepseek-v2:latest"), num_predict=32)
    messages = [
        {"role": "system", "content": ACTOR_SYSTEM.strip()},
        {"role": "user", "content": build_actor_user_prompt(example.question, example.context, reflection_memory)},
    ]
    return _http_chat_completion(messages, model=_model_name(_default_model_name()), temperature=0.0)


def judge_answer(example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallResult]:
    if _backend() == "mock":
        result = _mock_evaluator(example, answer)
        return result, LLMCallResult(content=result.model_dump_json(), usage=LLMUsage(), latency_ms=0, raw={"backend": "mock"})
    if _is_local_ollama():
        if _ollama_light_mode():
            result = _mock_evaluator(example, answer)
            return result, LLMCallResult(content=result.model_dump_json(), usage=LLMUsage(), latency_ms=0, raw={"backend": "ollama_light"})
        prompt = _message_to_prompt(
            EVALUATOR_SYSTEM,
            build_evaluator_user_prompt(example.question, example.gold_answer, answer, example.context),
        )
        call = _ollama_generate(prompt, model=_model_name("deepseek-v2:latest"), num_predict=192, json_mode=True)
        payload = _extract_json_block(call.content)
        if payload is None:
            result = _mock_evaluator(example, answer)
            result.raw_output = call.content
            return result, call
        result = JudgeResult(
            score=_normalize_bool_score(payload.get("score", 0)),
            reason=str(payload.get("reason", "")).strip() or "No reason provided.",
            missing_evidence=[str(item) for item in payload.get("missing_evidence", []) if str(item).strip()],
            spurious_claims=[str(item) for item in payload.get("spurious_claims", []) if str(item).strip()],
            final_answer=str(payload.get("final_answer", answer)).strip() or answer,
            raw_output=call.content,
        )
        return result, call

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM.strip()},
        {"role": "user", "content": build_evaluator_user_prompt(example.question, example.gold_answer, answer, example.context)},
    ]
    call = _http_chat_completion(messages, model=_model_name(_default_model_name()), temperature=0.0)
    payload = _extract_json_block(call.content)
    if payload is None:
        result = _mock_evaluator(example, answer)
        result.raw_output = call.content
        return result, call
    result = JudgeResult(
        score=_normalize_bool_score(payload.get("score", 0)),
        reason=str(payload.get("reason", "")).strip() or "No reason provided.",
        missing_evidence=[str(item) for item in payload.get("missing_evidence", []) if str(item).strip()],
        spurious_claims=[str(item) for item in payload.get("spurious_claims", []) if str(item).strip()],
        final_answer=str(payload.get("final_answer", answer)).strip() or answer,
        raw_output=call.content,
    )
    return result, call


def build_reflection(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> tuple[ReflectionEntry, LLMCallResult]:
    if _backend() == "mock":
        result = _mock_reflector(example, attempt_id, judge)
        return result, LLMCallResult(content=result.model_dump_json(), usage=LLMUsage(), latency_ms=0, raw={"backend": "mock"})
    if _is_local_ollama():
        if _ollama_light_mode():
            result = _mock_reflector(example, attempt_id, judge)
            return result, LLMCallResult(content=result.model_dump_json(), usage=LLMUsage(), latency_ms=0, raw={"backend": "ollama_light"})
        prompt = _message_to_prompt(
            REFLECTOR_SYSTEM,
            build_reflector_user_prompt(
                example.question,
                example.gold_answer,
                answer,
                judge.reason,
                example.context,
                attempt_id,
            ),
        )
        call = _ollama_generate(prompt, model=_model_name("deepseek-v2:latest"), num_predict=192, json_mode=True)
        payload = _extract_json_block(call.content)
        if payload is None:
            result = _mock_reflector(example, attempt_id, judge)
            result.raw_output = call.content
            return result, call
        result = ReflectionEntry(
            attempt_id=int(payload.get("attempt_id", attempt_id) or attempt_id),
            failure_reason=str(payload.get("failure_reason", judge.reason)).strip() or judge.reason,
            lesson=str(payload.get("lesson", "")).strip() or "Focus on the missing hop and avoid premature answers.",
            next_strategy=str(payload.get("next_strategy", "")).strip() or "Re-evaluate the context with the missing hop in mind.",
            raw_output=call.content,
        )
        return result, call

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM.strip()},
        {
            "role": "user",
            "content": build_reflector_user_prompt(
                example.question,
                example.gold_answer,
                answer,
                judge.reason,
                example.context,
                attempt_id,
            ),
        },
    ]
    call = _http_chat_completion(messages, model=_model_name(_default_model_name()), temperature=0.2)
    payload = _extract_json_block(call.content)
    if payload is None:
        result = _mock_reflector(example, attempt_id, judge)
        result.raw_output = call.content
        return result, call

    result = ReflectionEntry(
        attempt_id=int(payload.get("attempt_id", attempt_id) or attempt_id),
        failure_reason=str(payload.get("failure_reason", judge.reason)).strip() or judge.reason,
        lesson=str(payload.get("lesson", "")).strip() or "Focus on the missing hop and avoid premature answers.",
        next_strategy=str(payload.get("next_strategy", "")).strip() or "Re-evaluate the context with the missing hop in mind.",
        raw_output=call.content,
    )
    return result, call


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    return generate_actor_answer(example, attempt_id, agent_type, reflection_memory).content


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return judge_answer(example, answer)[0]


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    return build_reflection(example, attempt_id, answer=judge.final_answer or "", judge=judge)[0]
