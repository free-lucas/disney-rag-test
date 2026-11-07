from __future__ import annotations

from typing import Callable, Sequence

from models import RetrievalResult


RerankCall = Callable[[str, Sequence[str]], list[float]]


class DashScopeReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = 'gte-rerank-v2',
        timeout_seconds: int = 60,
        call_fn: RerankCall | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.call_fn = call_fn

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if self.call_fn is not None:
            return [float(value) for value in self.call_fn(query, documents)]
        if not self.api_key:
            raise RuntimeError('缺少 DASHSCOPE_API_KEY')
        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError('未安装 dashscope 依赖') from exc
        response = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=list(documents),
            top_n=len(documents),
            return_documents=False,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )
        if int(getattr(response, 'status_code', 0) or 0) != 200:
            raise RuntimeError('rerank 请求失败: ' + str(getattr(response, 'message', response)))
        output = getattr(response, 'output', None) or {}
        items = output.get('results') if isinstance(output, dict) else None
        if not isinstance(items, list):
            raise RuntimeError('rerank 返回结构非法')
        scores = [0.0] * len(documents)
        for item in items:
            index = int(item.get('index', 0))
            if 0 <= index < len(scores):
                scores[index] = float(item.get('relevance_score', item.get('score', 0.0)))
        return scores


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    *,
    reranker: object | None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    if reranker is None or not results:
        return results[:top_k] if top_k is not None else list(results)
    try:
        scores = reranker.score(query, [result.chunk.content for result in results])
    except Exception:
        return results[:top_k] if top_k is not None else list(results)
    if len(scores) != len(results):
        return results[:top_k] if top_k is not None else list(results)
    for result, score in zip(results, scores):
        result.rerank_score = float(score)
    ranked = sorted(results, key=lambda item: (-float(item.rerank_score or 0.0), item.chunk.id))
    return ranked[:top_k] if top_k is not None else ranked
