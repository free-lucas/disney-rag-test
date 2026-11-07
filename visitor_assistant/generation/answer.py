from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from generation.prompt import (
    CONFIDENTIALITY_REFUSAL_MESSAGE,
    DECISION_ANSWER,
    DECISION_REFUSE,
    PARTIAL_EVIDENCE_NOTE,
    REFUSAL_MESSAGE,
    TIME_SENSITIVE_DISCLAIMER,
    build_messages,
    cited_numbers,
    find_internal_leak,
    is_time_sensitive,
    public_evidence,
)
from models import Answer, Audience, ChunkType, Citation, ImageRef, RetrievalResult


class GenerationError(RuntimeError):
    pass


@dataclass
class GenerationOutput:
    decision: str
    body: str
    raw: str = ''


def parse_generation_response(raw: str) -> GenerationOutput:
    text = str(raw or '').strip()
    if not text:
        return GenerationOutput(DECISION_REFUSE, '', text)
    first_line, separator, rest = text.partition('\n')
    normalized = first_line.strip().upper().rstrip('：:')
    if normalized.startswith(DECISION_REFUSE):
        return GenerationOutput(DECISION_REFUSE, '', text)
    if normalized.startswith(DECISION_ANSWER):
        return GenerationOutput(DECISION_ANSWER, rest.strip(), text)
    return GenerationOutput(DECISION_ANSWER, text, text)


def selected_evidence(
    results: list[RetrievalResult],
    answer_text: str,
) -> list[RetrievalResult]:
    evidence = public_evidence(results)
    referenced = set(cited_numbers(answer_text))
    return [
        result for index, result in enumerate(evidence, start=1)
        if index in referenced
    ] or evidence


def citations_from_results(
    results: list[RetrievalResult],
    answer_text: str,
) -> list[Citation]:
    selected = selected_evidence(results, answer_text)
    citations: list[Citation] = []
    seen: set[tuple[object, ...]] = set()
    for result in selected:
        citation = Citation.from_source_ref(result.source_ref, result.chunk.id)
        key = (
            citation.source_file,
            tuple(citation.section_path),
            citation.page,
            citation.slide,
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def images_from_results(results: list[RetrievalResult], answer_text: str = '') -> list[ImageRef]:
    images: list[ImageRef] = []
    seen: set[str] = set()
    for result in selected_evidence(results, answer_text):
        chunk = result.chunk
        if chunk.audience is not Audience.PUBLIC:
            continue
        if chunk.chunk_type is not ChunkType.IMAGE_CAPTION or not chunk.image_path:
            continue
        if chunk.image_path in seen:
            continue
        seen.add(chunk.image_path)
        images.append(ImageRef(
            image_path=chunk.image_path,
            caption=chunk.caption or chunk.content,
            source_file=chunk.source_file,
            section_path=list(chunk.section_path),
            page=chunk.page,
            slide=chunk.slide,
        ))
    return images


def _append_disclaimer(text: str) -> str:
    if TIME_SENSITIVE_DISCLAIMER in text:
        return text
    return text.rstrip() + '\n\n' + TIME_SENSITIVE_DISCLAIMER


class AnswerGenerator:
    def __init__(
        self,
        *,
        api_key: str = '',
        base_url: str = 'https://api.deepseek.com',
        model: str = 'deepseek-flash',
        timeout_seconds: int = 90,
        max_retries: int = 3,
        max_tokens: int = 1600,
        client: object | None = None,
        sleep_fn: object | None = None,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.max_tokens = max_tokens
        self.client = client
        self.sleep_fn = sleep_fn or time.sleep
        self.retry_base_seconds = retry_base_seconds

    @classmethod
    def from_settings(cls, settings: object, client: object | None = None) -> 'AnswerGenerator':
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
            max_tokens=settings.answer_max_tokens,
            client=client,
        )

    def _get_client(self) -> object:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise GenerationError('缺少 DEEPSEEK_API_KEY')
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_base_seconds * (2 ** attempt)
        self.sleep_fn(delay)

    def complete(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=min(self.max_tokens + attempt * 1000, 8000),
                    timeout=self.timeout_seconds,
                )
                choice = response.choices[0]
                content = str(choice.message.content or '').strip()
                finish_reason = getattr(choice, 'finish_reason', None)
                if content and finish_reason != 'length':
                    return content
                last_error = GenerationError(
                    '模型输出为空' if not content else '模型输出因长度限制被截断'
                )
                if attempt < self.max_retries:
                    self._backoff(attempt)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._backoff(attempt)
        raise GenerationError('生成失败: ' + str(last_error)[:500]) from last_error

    def _stream_once(self, messages: list[dict[str, str]]) -> Iterator[str]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.max_tokens,
            timeout=self.timeout_seconds,
            stream=True,
        )
        for event in response:
            choices = getattr(event, 'choices', None) or []
            if not choices:
                continue
            delta = getattr(choices[0], 'delta', None)
            content = getattr(delta, 'content', None) if delta is not None else None
            if content:
                yield str(content)

    def stream_complete(self, messages: list[dict[str, str]]) -> Iterator[str]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = False
            try:
                for piece in self._stream_once(messages):
                    started = True
                    yield piece
                return
            except Exception as exc:
                last_error = exc
                if started or attempt >= self.max_retries:
                    raise GenerationError('流式生成失败: ' + str(exc)[:500]) from exc
                self._backoff(attempt)
        raise GenerationError('流式生成失败: ' + str(last_error)[:500]) from last_error

    def refusal(self, question: str, rewritten_query: str = '', *, confidential: bool = False) -> Answer:
        return Answer(
            text=CONFIDENTIALITY_REFUSAL_MESSAGE if confidential else REFUSAL_MESSAGE,
            citations=[],
            images=[],
            refused=True,
            disclaimer=False,
            query=str(question),
            rewritten_query=rewritten_query or str(question),
        )

    def finalize(
        self,
        question: str,
        results: list[RetrievalResult],
        raw: str,
        *,
        rewritten_query: str = '',
        decision: str | None = None,
        force_refuse: bool = False,
    ) -> Answer:
        parsed = parse_generation_response(raw)
        effective_decision = decision or parsed.decision
        if force_refuse or effective_decision == DECISION_REFUSE or not parsed.body.strip():
            return self.refusal(question, rewritten_query)

        body = parsed.body.strip()
        leak = find_internal_leak(body)
        if leak is not None:
            return self.refusal(question, rewritten_query, confidential=True)

        answer = Answer(
            text=body,
            citations=citations_from_results(results, body),
            images=images_from_results(results, body),
            refused=False,
            disclaimer=False,
            query=str(question),
            rewritten_query=rewritten_query or str(question),
        )
        if not answer.citations:
            answer.refused = True
            answer.text = REFUSAL_MESSAGE
            answer.images = []
            return answer

        volatile_text = str(rewritten_query or question) + ' ' + body
        if is_time_sensitive(volatile_text):
            answer.text = _append_disclaimer(answer.text)
            answer.disclaimer = True
        return answer

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        history: list[dict[str, str]] | None = None,
        *,
        rewritten_query: str = '',
    ) -> Answer:
        evidence = public_evidence(results)
        if not evidence:
            return self.refusal(question, rewritten_query)
        messages = build_messages(question, evidence, history)
        try:
            raw = self.complete(messages)
        except GenerationError as exc:
            answer = self.refusal(question, rewritten_query)
            answer.error = str(exc)
            return answer
        return self.finalize(question, evidence, raw, rewritten_query=rewritten_query)


@dataclass
class StreamingAnswer:
    question: str
    results: list[RetrievalResult]
    rewritten_query: str = ''
    decision: str = DECISION_ANSWER
    raw_parts: list[str] = field(default_factory=list)
    error: str | None = None
    _final: Answer | None = None
    _iterator: Iterator[str] | None = None
    _generator: 'AnswerGenerator | None' = None

    def __iter__(self) -> Iterator[str]:
        if self._iterator is None:
            return iter(())
        return self._iterator

    @property
    def raw_text(self) -> str:
        return ''.join(self.raw_parts)

    def finalize(self) -> Answer:
        if self._final is not None:
            return self._final
        if self._generator is None:
            raise GenerationError('流式回答没有绑定生成器')
        if self.error:
            answer = self._generator.refusal(self.question, self.rewritten_query)
            answer.error = self.error
            return answer
        return self._generator.finalize(
            self.question,
            self.results,
            self.raw_text,
            rewritten_query=self.rewritten_query,
            decision=self.decision,
            force_refuse=self.decision == DECISION_REFUSE,
        )


def stream_answer(
    generator: AnswerGenerator,
    question: str,
    results: list[RetrievalResult],
    history: list[dict[str, str]] | None = None,
    *,
    rewritten_query: str = '',
) -> StreamingAnswer:
    evidence = public_evidence(results)
    holder = StreamingAnswer(
        question=question,
        results=evidence,
        rewritten_query=rewritten_query,
        _generator=generator,
    )

    def iterate() -> Iterator[str]:
        if not evidence:
            holder.decision = DECISION_REFUSE
            yield REFUSAL_MESSAGE
            return
        messages = build_messages(question, evidence, history)
        prefix = ''
        body_started = False
        try:
            for piece in generator.stream_complete(messages):
                if body_started:
                    holder.raw_parts.append(piece)
                    yield piece
                    continue
                prefix += piece
                if '\n' not in prefix:
                    continue
                first_line, rest = prefix.split('\n', 1)
                normalized = first_line.strip().upper().rstrip('：:')
                if normalized.startswith(DECISION_REFUSE):
                    holder.decision = DECISION_REFUSE
                    yield REFUSAL_MESSAGE
                    return
                body_started = True
                if normalized.startswith(DECISION_ANSWER):
                    if rest:
                        holder.raw_parts.append(rest)
                        yield rest
                else:
                    whole = first_line + '\n' + rest
                    holder.raw_parts.append(whole)
                    yield whole
            if not body_started:
                normalized = prefix.strip().upper()
                if normalized.startswith(DECISION_REFUSE):
                    holder.decision = DECISION_REFUSE
                    yield REFUSAL_MESSAGE
                    return
                if not prefix.strip():
                    fallback = generator.generate(
                        question,
                        evidence,
                        history,
                        rewritten_query=rewritten_query,
                    )
                    holder._final = fallback
                    holder.decision = DECISION_REFUSE if fallback.refused else DECISION_ANSWER
                    holder.raw_parts.append(fallback.text)
                    yield fallback.text
                    return
                body = prefix
                if normalized.startswith(DECISION_ANSWER):
                    body = prefix.split('\n', 1)[-1].strip() if '\n' in prefix else ''
                holder.raw_parts.append(body)
                if body:
                    yield body
        except GenerationError as exc:
            holder.error = str(exc)

    holder._iterator = iterate()
    return holder
