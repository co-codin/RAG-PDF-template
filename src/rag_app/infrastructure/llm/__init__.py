"""Chat model providers. Interface: ``langchain_core.language_models.BaseChatModel``.
Selected by LLM_PROVIDER."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_app.core.config import Settings
from rag_app.core.errors import ConfigurationError
from rag_app.infrastructure.llm.fake import FakeChatModel, message_text

__all__ = ["BaseChatModel", "FakeChatModel", "build_chat_model", "message_text"]


def build_chat_model(settings: Settings) -> BaseChatModel:
    match settings.llm_provider:
        case "fake":
            return FakeChatModel()
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs: dict[str, Any] = {
                "model": settings.llm_model,
                "max_tokens": settings.llm_max_tokens,
                "timeout": settings.llm_timeout_s,
                "max_retries": settings.llm_max_retries,
            }
            if settings.llm_temperature is not None:
                kwargs["temperature"] = settings.llm_temperature
            if settings.anthropic_api_key is not None:
                kwargs["api_key"] = settings.anthropic_api_key
            return ChatAnthropic(**kwargs)
    raise ConfigurationError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
