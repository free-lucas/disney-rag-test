from __future__ import annotations

from dataclasses import dataclass, field

from models import Audience, ChunkType, RetrievalResult
from retrieval.bm25 import tokenize
from retrieval.embed import DashScopeEmbedder, EmbeddingCache
from retrieval.rerank import DashScopeReranker, rerank_results
from retrieval.rewrite import QueryRewriter, RewriteResult
from retrieval.store import IndexLoadError, IndexStore


class PublicFilterError(RuntimeError):
    pass


def assert_public(results: list[RetrievalResult]) -> list[RetrievalResult]:
    for result in results:
        if result.chunk.audience is not Audience.PUBLIC:
            raise PublicFilterError('游客检索结果混入 internal chunk: ' + result.chunk.id)
    return results


def _result_key(result: RetrievalResult) -> str:
    return result.chunk.content_hash or result.chunk.id


def rrf_fuse(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    *,
    rrf_k: int = 60,
    top_k: int = 6,
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for rank, result in enumerate(dense_results, start=1):
        key = _result_key(result)
        existing = merged.get(key)
        contribution = 1.0 / (rrf_k + rank)
        if existing is None:
            merged[key] = RetrievalResult(
                chunk=result.chunk,
                score=contribution,
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                matched_query=result.matched_query,
            )
        else:
            existing.score += contribution
            if result.dense_score is not None:
                existing.dense_score = max(existing.dense_score or float('-inf'), result.dense_score)
    for rank, result in enumerate(bm25_results, start=1):
        key = _result_key(result)
        existing = merged.get(key)
        contribution = 1.0 / (rrf_k + rank)
        if existing is None:
            merged[key] = RetrievalResult(
                chunk=result.chunk,
                score=contribution,
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                matched_query=result.matched_query,
            )
        else:
            existing.score += contribution
            if result.bm25_score is not None:
                existing.bm25_score = max(existing.bm25_score or float('-inf'), result.bm25_score)
    merged_results = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.id))
    return assert_public(merged_results[:top_k])


def fuse_result_groups(
    groups: list[list[RetrievalResult]],
    *,
    rrf_k: int = 60,
    top_k: int = 6,
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for group in groups:
        for rank, result in enumerate(group, start=1):
            key = _result_key(result)
            contribution = 1.0 / (rrf_k + rank)
            existing = merged.get(key)
            if existing is None:
                merged[key] = RetrievalResult(
                    chunk=result.chunk,
                    score=contribution,
                    dense_score=result.dense_score,
                    bm25_score=result.bm25_score,
                    matched_query=result.matched_query,
                )
            else:
                existing.score += contribution
                if result.dense_score is not None:
                    existing.dense_score = max(existing.dense_score or float('-inf'), result.dense_score)
                if result.bm25_score is not None:
                    existing.bm25_score = max(existing.bm25_score or float('-inf'), result.bm25_score)
    results = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.id))
    return assert_public(results[:top_k])


IMAGE_QUERY_TERMS = ('图片', '图', '价格', '票价', '地图', '时间表', '时刻', '房型', '邮轮', '酒店', '门票')


def associate_relevant_images(
    results: list[RetrievalResult],
    store: IndexStore,
    query: str,
    *,
    limit: int = 2,
) -> list[RetrievalResult]:
    if not results:
        return []
    existing_ids = {result.chunk.id for result in results}
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return results
    candidates: list[tuple[int, float, object]] = []
    for chunk in store.public_chunks:
        if chunk.chunk_type is not ChunkType.IMAGE_CAPTION or chunk.id in existing_ids:
            continue
        image_tokens = set(tokenize((chunk.caption or '') + ' ' + chunk.content + ' ' + ' '.join(chunk.section_path)))
        overlap = query_tokens & image_tokens
        explicit_visual_intent = any(term in query for term in IMAGE_QUERY_TERMS)
        if len(overlap) < 2 and not (overlap and explicit_visual_intent):
            continue
        candidates.append((len(overlap), float(len(overlap)) / max(1, len(query_tokens)), chunk))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2].id))
    combined = list(results)
    for _, score, chunk in candidates[:limit]:
        combined.append(RetrievalResult(chunk=chunk, score=score, matched_query='image-link'))
    return assert_public(combined)


@dataclass
class RetrievalBundle:
    results: list[RetrievalResult] = field(default_factory=list)
    rewrite: RewriteResult | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'results': [result.to_dict() for result in self.results],
            'rewrite': self.rewrite.to_dict() if self.rewrite else None,
        }


class HybridRetriever:
    def __init__(
        self,
        *,
        store: IndexStore,
        embedder: object,
        rewriter: object | None = None,
        reranker: object | None = None,
        top_k: int = 6,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        dense_min_score: float = 0.25,
        rerank_enabled: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rewriter = rewriter
        self.reranker = reranker
        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.dense_min_score = dense_min_score
        self.rerank_enabled = rerank_enabled
        self.last_rewrite: RewriteResult | None = None

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        store: IndexStore | None = None,
        embedder: object | None = None,
        rewriter: object | None = None,
        reranker: object | None = None,
    ) -> 'HybridRetriever':
        loaded_store = store or IndexStore.load(settings.index_dir)
        loaded_embedder = embedder or DashScopeEmbedder.from_settings(
            settings,
            cache=EmbeddingCache(settings.cache_dir / 'embeddings'),
        )
        loaded_rewriter = rewriter or QueryRewriter.from_settings(settings)
        loaded_reranker = reranker
        if settings.rerank_enabled and loaded_reranker is None:
            loaded_reranker = DashScopeReranker(
                api_key=settings.dashscope_api_key,
                timeout_seconds=settings.model_timeout_seconds,
            )
        return cls(
            store=loaded_store,
            embedder=loaded_embedder,
            rewriter=loaded_rewriter,
            reranker=loaded_reranker,
            top_k=settings.retrieval_top_k,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            rrf_k=settings.rrf_k,
            dense_min_score=settings.dense_min_score,
            rerank_enabled=settings.rerank_enabled,
        )

    def _dense(self, query: str, k: int) -> list[RetrievalResult]:
        vector = self.embedder.embed_query(query)
        results = []
        for index, score in self.store.dense_search(vector, k):
            if score < self.dense_min_score:
                continue
            results.append(RetrievalResult(
                chunk=self.store.chunks[index],
                score=score,
                dense_score=score,
                matched_query=query,
            ))
        return assert_public(results)

    def _bm25(self, query: str, k: int) -> list[RetrievalResult]:
        results = []
        for index, score in self.store.bm25_search(query, k):
            results.append(RetrievalResult(
                chunk=self.store.chunks[index],
                score=score,
                bm25_score=score,
                matched_query=query,
            ))
        return assert_public(results)

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        limit = top_k or self.top_k
        dense = self._dense(query, self.dense_top_k)
        bm25 = self._bm25(query, self.bm25_top_k)
        fused = rrf_fuse(dense, bm25, rrf_k=self.rrf_k, top_k=max(limit, self.top_k))
        if self.rerank_enabled and self.reranker is not None:
            fused = rerank_results(query, fused, reranker=self.reranker, top_k=limit)
        return assert_public(fused[:limit])

    def retrieve(self, question: str, history: list[dict[str, str]] | None = None) -> RetrievalBundle:
        rewrite = RewriteResult(str(question).strip(), str(question).strip(), [str(question).strip()])
        if self.rewriter is not None:
            rewrite = self.rewriter.rewrite(question, history)
        self.last_rewrite = rewrite
        queries = rewrite.sub_queries or [rewrite.rewritten_query or str(question)]
        groups = [self.search(query, top_k=self.top_k) for query in queries]
        if len(groups) == 1:
            results = groups[0]
        else:
            results = fuse_result_groups(groups, rrf_k=self.rrf_k, top_k=self.top_k)
        results = associate_relevant_images(
            results,
            self.store,
            rewrite.rewritten_query or str(question),
        )
        return RetrievalBundle(results=assert_public(results), rewrite=rewrite)
