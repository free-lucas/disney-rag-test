from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_CATEGORIES = {
    'ticket',
    'attraction',
    'hotel',
    'dining',
    'emergency',
    'cruise',
    'out_of_domain',
}

CATEGORY_LABELS = {
    'ticket': '门票',
    'attraction': '乐园项目',
    'hotel': '酒店',
    'dining': '餐饮',
    'emergency': '应急指引',
    'cruise': '邮轮',
    'out_of_domain': '域外问题',
}


class GoldenQuestionError(ValueError):
    pass


@dataclass
class GoldenQuestion:
    id: str
    category: str
    question: str
    expected_source_files: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    notes: str = ''

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    def to_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'category': self.category,
            'question': self.question,
            'expected_source_files': list(self.expected_source_files),
            'expect_refusal': self.expect_refusal,
            'notes': self.notes,
        }


def normalize_source(value: str) -> str:
    return str(value).strip().replace('\\', '/').lstrip('./')


def load_golden_questions(path: str | Path) -> list[GoldenQuestion]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenQuestionError('无法读取金标准问题: ' + str(exc)) from exc
    if isinstance(payload, dict):
        raw_questions = payload.get('questions', [])
    else:
        raw_questions = payload
    if not isinstance(raw_questions, list) or not raw_questions:
        raise GoldenQuestionError('金标准问题必须是非空数组')

    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise GoldenQuestionError('第 ' + str(index + 1) + ' 题不是对象')
        identifier = str(item.get('id', '')).strip()
        category = str(item.get('category', '')).strip()
        question = str(item.get('question', '')).strip()
        if not identifier or not question:
            raise GoldenQuestionError('第 ' + str(index + 1) + ' 题缺少 id 或 question')
        if identifier in seen_ids:
            raise GoldenQuestionError('重复的题目 id: ' + identifier)
        seen_ids.add(identifier)
        if category not in REQUIRED_CATEGORIES:
            raise GoldenQuestionError('未知类别: ' + category)
        raw_sources = item.get('expected_source_files', [])
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        if not isinstance(raw_sources, list):
            raise GoldenQuestionError('expected_source_files 必须是数组: ' + identifier)
        expected = [normalize_source(value) for value in raw_sources if str(value).strip()]
        expect_refusal = bool(item.get('expect_refusal', False))
        if category == 'out_of_domain' and not expect_refusal:
            raise GoldenQuestionError('域外题必须标记 expect_refusal=true: ' + identifier)
        if not expect_refusal and not expected:
            raise GoldenQuestionError('可回答题目必须提供期望来源: ' + identifier)
        questions.append(GoldenQuestion(
            id=identifier,
            category=category,
            question=question,
            expected_source_files=expected,
            expect_refusal=expect_refusal,
            notes=str(item.get('notes', '')),
        ))

    present = {item.category for item in questions}
    missing = sorted(REQUIRED_CATEGORIES - present)
    if missing:
        raise GoldenQuestionError('缺少必需类别: ' + ', '.join(missing))
    return questions
