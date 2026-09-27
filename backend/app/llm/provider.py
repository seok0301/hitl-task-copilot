"""LLM 프로바이더 추상화.

에이전트는 `call_json()`만 부른다. 프로바이더(litellm / gemini / mock)는 설정으로 바꾸고,
모든 호출은 소요 시간과 토큰 수를 `llm_calls` 테이블에 남긴다.
"""

import json
import re
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.db import SessionLocal
from app.models import LLMCall


class LLMError(RuntimeError):
    pass


@lru_cache
def get_chat_model() -> BaseChatModel | None:
    s = get_settings()
    if s.llm_provider == "litellm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.llm_model,
            base_url=s.litellm_base_url,
            api_key=s.litellm_api_key or "none",
            temperature=s.llm_temperature,
            timeout=180,
            max_retries=2,
            # 게이트웨이(vLLM)의 JSON 모드로 응답 형식을 강제한다.
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    if s.llm_provider == "gemini":
        if not s.allow_paid_api:
            raise LLMError("Gemini는 과금되는 API입니다. 쓰려면 ALLOW_PAID_API=true를 설정하세요.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=s.llm_model,
            google_api_key=s.gemini_api_key,
            temperature=s.llm_temperature,
        )
    if s.llm_provider == "mock":
        return None
    raise LLMError(f"알 수 없는 LLM_PROVIDER: {s.llm_provider}")


def model_name() -> str:
    s = get_settings()
    return "mock" if s.llm_provider == "mock" else s.llm_model


def parse_json(text: str) -> Any:
    """모델 응답에서 JSON을 꺼낸다. 코드 블록으로 감싼 응답도 처리한다."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start < 0 or end < start:
            raise LLMError(f"JSON을 찾지 못했습니다: {text[:200]}") from None
        return json.loads(text[start : end + 1])


def _log(agent: str, project_id: int | None, latency_ms: float, usage: dict) -> None:
    with SessionLocal() as db:
        db.add(
            LLMCall(
                agent=agent,
                model=model_name(),
                project_id=project_id,
                latency_ms=latency_ms,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        )
        db.commit()


def call_json(
    agent: str,
    system: str,
    user: str,
    mock: Callable[[], Any],
    project_id: int | None = None,
    retries: int = 1,
) -> Any:
    """LLM에 JSON 응답을 요청한다. mock 프로바이더에서는 `mock()` 결과를 그대로 돌려준다."""
    model = get_chat_model()
    started = time.perf_counter()
    if model is None:
        result = mock()
        _log(agent, project_id, (time.perf_counter() - started) * 1000, {})
        return result

    messages = [
        SystemMessage(system + "\n\n반드시 JSON만 출력한다."),
        HumanMessage(user),
    ]
    last_error: Exception | None = None
    for _ in range(retries + 1):
        resp = model.invoke(messages)
        usage = dict(resp.usage_metadata or {})
        _log(agent, project_id, (time.perf_counter() - started) * 1000, usage)
        try:
            return parse_json(str(resp.content))
        except (LLMError, json.JSONDecodeError) as e:
            last_error = e
            started = time.perf_counter()
    raise LLMError(f"{agent}: JSON 파싱 실패 ({last_error})")
