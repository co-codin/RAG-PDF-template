"""Offline, deterministic chat model for tests and keyless demos (LLM_PROVIDER=fake).

It recognises the pipeline's prompts by their *format* (``Follow-up question:`` for
the rewrite prompt, ``Context passages:`` for the answer prompt) so this module does
not import the rag layer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from rag_app.domain.text import tokenize

DONT_KNOW = "I don't know based on the provided documents."


def message_text(msg: BaseMessage) -> str:
    """Plain text of a message/chunk (handles Anthropic content-block lists)."""
    text = msg.text
    return text() if callable(text) else (text or "")


class FakeChatModel(BaseChatModel):
    """* answer prompt  -> the 1-2 context sentences best matching the question, cited [n]
    * rewrite prompt -> the follow-up question unchanged
    * anything else  -> echo of the last message
    Streams word by word so SSE paths are testable without a network."""

    @property
    def _llm_type(self) -> str:
        return "fake-extractive"

    def _respond(self, messages: list[BaseMessage]) -> str:
        last = message_text(messages[-1]) if messages else ""
        if "Follow-up question:" in last:
            m = re.search(r"Follow-up question:\s*(.+?)\n", last + "\n", re.S)
            return m.group(1).strip() if m else last
        if "Context passages:" in last:
            return self._extractive_answer(last)
        return last

    @staticmethod
    def _extractive_answer(prompt: str) -> str:
        q = re.search(r"Question:\s*(.+?)\n", prompt + "\n", re.S)
        q_tokens = set(tokenize(q.group(1) if q else ""))
        passages = re.findall(
            r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\n\[\d+\] \(source:|\n\nQuestion:)", prompt, re.S)
        sentences: list[tuple[str, str, set[str]]] = []
        for n, text in passages:
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for sent in re.split(r"(?<=[.!?])\s+", line):
                    sent = sent.strip().lstrip("-* ").strip()
                    if len(sent) >= 15:
                        sentences.append((n, sent, set(tokenize(sent))))
        if not sentences:
            return DONT_KNOW
        df = {t: sum(1 for *_, toks in sentences if t in toks) for t in q_tokens}
        scored = []
        for order, (n, sent, toks) in enumerate(sentences):
            score = sum(math.log(1 + len(sentences) / df[t]) for t in q_tokens & toks)
            if score > 0:
                scored.append((-score, order, n, sent))
        if not scored:
            return DONT_KNOW
        top = sorted(sorted(scored)[:2], key=lambda x: x[1])
        return " ".join(f"{sent} [{n}]" for _, _, n, sent in top)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._respond(messages)))])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
                ) -> Iterator[ChatGenerationChunk]:
        for token in re.findall(r"\S+\s*", self._respond(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
