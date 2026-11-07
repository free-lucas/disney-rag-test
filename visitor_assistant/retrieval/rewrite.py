from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


REWRITE_SYSTEM_PROMPT = (
    '你是迪士尼游客助手的查询理解模块。只做查询改写和意图拆分，不回答问题。'
    '不得引入历史中不存在的事实，不得把无关历史强行拼进独立问题。'
)
REWRITE_USER_PROMPT = '''请分析游客本轮问题，并只输出一个 JSON 对象：
{
  rewritten_query: 补全指代后的独立问题；不需要改写时保持原问题,
  sub_queries: [每个独立信息需求一个检索查询],
  needs_history: true 或 false
}
规则：
1. 如果本轮问题依赖上一轮（如“那它多少钱”“那酒店呢”），needs_history 为 true，并补全对象。
2. 如果本轮问题本身完整，needs_history 为 false，rewritten_query 使用原问题，不要把旧话题混入。
3. 复合问题按信息需求拆分；单一问题只返回一个 sub_query。
4. 保留门票名、项目名、酒店名、价格、地点等精确词。
5. 不要回答问题，不要输出 Markdown。'''


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    needs_history: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'original_query': self.original_query,
            'rewritten_query': self.rewritten_query,
            'sub_queries': list(self.sub_queries),
            'needs_history': self.needs_history,
            'error': self.error,
        }


def split_intents(query: str) -> list[str]:
    text = str(query).strip()
    if not text:
        return []
    pieces = re.split(r'[?？;；]+', text)
    candidates = [piece.strip(' ，,。.!！') for piece in pieces]
    queries: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries or [text]


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return '（无历史）'
    lines: list[str] = []
    for message in history[-6:]:
        role = '游客' if str(message.get('role', '')) == 'user' else '助手'
        content = str(message.get('content', '')).strip()
        if content:
            lines.append(role + '：' + content[:500])
    return '\n'.join(lines) or '（无历史）'


def parse_rewrite_response(content: str, original: str, *, has_history: bool) -> RewriteResult:
    text = str(content or '').strip()
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('查询改写响应不是 JSON 对象')
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError('查询改写响应必须是对象')

    raw_sub_queries = payload.get('sub_queries', [])
    if isinstance(raw_sub_queries, str):
        raw_sub_queries = [raw_sub_queries]
    if not isinstance(raw_sub_queries, list):
        raw_sub_queries = []
    sub_queries = _deduplicate([str(item) for item in raw_sub_queries])
    rewritten = str(payload.get('rewritten_query', '') or original).strip() or original
    needs_history = bool(payload.get('needs_history', False)) and has_history
    if not needs_history:
        rewritten = original
    if not sub_queries:
        sub_queries = split_intents(rewritten)
    return RewriteResult(
        original_query=original,
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        needs_history=needs_history,
    )


class QueryRewriter:
    def __init__(
        self,
        *,
        api_key: str = '',
        base_url: str = 'https://api.deepseek.com',
        model: str = 'deepseek-flash',
        timeout_seconds: int = 90,
        max_tokens: int = 500,
        max_retries: int = 2,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.client = client

    @classmethod
    def from_settings(cls, settings: object, client: object | None = None) -> 'QueryRewriter':
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_tokens=settings.rewrite_max_tokens,
            max_retries=settings.model_max_retries,
            client=client,
        )

    def _get_client(self) -> object:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError('缺少 DEEPSEEK_API_KEY')
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> RewriteResult:
        original = str(query).strip()
        if not original:
            return RewriteResult(original, original, [])
        has_history = bool(history)
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': REWRITE_SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': REWRITE_USER_PROMPT + '\n\n最近对话：\n' + _format_history(history) + '\n\n本轮问题：' + original,
                    },
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
            )
            content = str(response.choices[0].message.content or '')
            return parse_rewrite_response(content, original, has_history=has_history)
        except Exception as exc:
            fallback_subqueries = split_intents(original)
            return RewriteResult(
                original_query=original,
                rewritten_query=original,
                sub_queries=fallback_subqueries,
                needs_history=False,
                error=str(exc)[:500],
            )
