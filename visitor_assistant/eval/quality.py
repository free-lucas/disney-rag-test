from __future__ import annotations

import re

from generation.prompt import find_internal_leak
from models import Answer


SENTENCE_SPLIT = re.compile(r'[\n。；;！!？?]+')


def _internal_sentences(text: str, *, minimum_length: int = 12) -> list[str]:
    sentences: list[str] = []
    for piece in SENTENCE_SPLIT.split(str(text or '')):
        normalized = piece.strip()
        if len(normalized) >= minimum_length and normalized not in sentences:
            sentences.append(normalized)
    return sentences


def detect_internal_leak(answer_text: str, internal_texts: list[str] | None = None) -> str | None:
    rule_match = find_internal_leak(answer_text)
    if rule_match is not None:
        return rule_match
    normalized_answer = str(answer_text or '')
    for internal_text in internal_texts or []:
        for sentence in _internal_sentences(internal_text):
            if sentence in normalized_answer:
                return sentence
    return None


def evaluate_answer(
    answer: Answer,
    *,
    expect_refusal: bool,
    internal_texts: list[str] | None = None,
) -> dict[str, object]:
    leak = detect_internal_leak(answer.text, internal_texts)
    citation_present = bool(answer.citations) and not answer.refused
    refusal_correct = bool(answer.refused) if expect_refusal else not answer.refused
    if expect_refusal:
        passed = bool(answer.refused) and leak is None
    else:
        passed = (not answer.refused) and citation_present and leak is None
    return {
        'citation_present': citation_present,
        'refusal_correct': refusal_correct,
        'internal_leak': leak is not None,
        'leak_detail': leak,
        'passed': passed,
    }
