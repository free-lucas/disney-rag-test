from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


class EmbeddingError(RuntimeError):
    pass


class EmbeddingCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(content_hash: str, model: str) -> str:
        raw = content_hash + '|' + model
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _path(self, content_hash: str, model: str) -> Path:
        return self.directory / (self.key(content_hash, model) + '.npy')

    def get(self, content_hash: str, model: str) -> list[float] | None:
        path = self._path(content_hash, model)
        if not path.is_file():
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0:
            return None
        return vector.astype(np.float32).tolist()

    def set(self, content_hash: str, model: str, vector: Sequence[float]) -> None:
        path = self._path(content_hash, model)
        array = np.asarray(vector, dtype=np.float32)
        temporary = path.with_suffix('.tmp.npy')
        np.save(temporary, array)
        temporary.replace(path)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def _dashscope_embedding(model: str, texts: list[str], api_key: str, timeout: int) -> list[list[float]]:
    try:
        import dashscope
    except ImportError as exc:
        raise EmbeddingError('未安装 dashscope 依赖') from exc

    response = dashscope.TextEmbedding.call(
        model=model,
        input=texts,
        api_key=api_key,
        timeout=timeout,
    )
    status_code = int(getattr(response, 'status_code', 0) or 0)
    if status_code != 200:
        message = getattr(response, 'message', '') or str(response)
        raise EmbeddingError('DashScope embedding 失败（' + str(status_code) + '）：' + str(message)[:300])
    output = getattr(response, 'output', None) or {}
    embeddings = output.get('embeddings') if isinstance(output, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingError('DashScope embedding 返回数量不匹配')
    ordered = sorted(embeddings, key=lambda item: int(item.get('text_index', item.get('index', 0))))
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get('embedding')
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError('DashScope embedding 返回空向量')
        vectors.append([float(value) for value in vector])
    return vectors


class DashScopeEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        primary_model: str = 'text-embedding-v4',
        fallback_model: str = 'text-embedding-v1',
        cache: EmbeddingCache | None = None,
        batch_size: int = 10,
        timeout_seconds: int = 90,
        max_retries: int = 3,
        call_fn: Callable[[str, list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.api_key = api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.cache = cache
        self.batch_size = max(1, batch_size)
        self.timeout_seconds = max(1, timeout_seconds)
        self.max_retries = max(0, max_retries)
        self.call_fn = call_fn
        self.active_model = primary_model
        self.last_fallback_reason: str | None = None
        self.api_calls = 0

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        cache: EmbeddingCache | None = None,
        call_fn: Callable[[str, list[str]], list[list[float]]] | None = None,
    ) -> 'DashScopeEmbedder':
        return cls(
            api_key=settings.dashscope_api_key,
            primary_model=settings.embedding_model,
            fallback_model=settings.embedding_fallback_model,
            cache=cache or EmbeddingCache(settings.cache_dir / 'embeddings'),
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
            call_fn=call_fn,
        )

    def _call(self, model: str, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingError('缺少 DASHSCOPE_API_KEY')
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.api_calls += 1
                if self.call_fn is not None:
                    return self.call_fn(model, texts)
                return _dashscope_embedding(model, texts, self.api_key, self.timeout_seconds)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(5.0, 0.4 * (2 ** attempt)))
        raise EmbeddingError(str(last_error or 'embedding 请求失败'))

    def _embed_with_model(self, texts: list[str], model: str) -> tuple[np.ndarray, int]:
        hashes = [_content_hash(text) for text in texts]
        vectors: list[list[float] | None] = [None] * len(texts)
        if self.cache is not None:
            for index, digest in enumerate(hashes):
                cached = self.cache.get(digest, model)
                if cached is not None:
                    vectors[index] = cached

        missing: dict[str, list[int]] = {}
        for index, digest in enumerate(hashes):
            if vectors[index] is None:
                missing.setdefault(digest, []).append(index)
        api_calls = 0
        unique_hashes = list(missing)
        for start in range(0, len(unique_hashes), self.batch_size):
            batch_hashes = unique_hashes[start:start + self.batch_size]
            batch_texts = [texts[missing[digest][0]] for digest in batch_hashes]
            batch_vectors = self._call(model, batch_texts)
            api_calls += 1
            if len(batch_vectors) != len(batch_texts):
                raise EmbeddingError('embedding 返回数量与请求数量不一致')
            for digest, vector in zip(batch_hashes, batch_vectors):
                if self.cache is not None:
                    self.cache.set(digest, model, vector)
                for index in missing[digest]:
                    vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise EmbeddingError('embedding 结果不完整')
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError('embedding 向量维度非法')
        dimensions = {int(row.shape[0]) for row in matrix}
        if len(dimensions) != 1:
            raise EmbeddingError('embedding 维度不一致: ' + str(sorted(dimensions)))
        return _normalize_rows(matrix), api_calls

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        items = [str(text).strip() for text in texts]
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        if any(not item for item in items):
            raise EmbeddingError('不能对空文本生成 embedding')
        try:
            matrix, _ = self._embed_with_model(items, self.primary_model)
            self.active_model = self.primary_model
            self.last_fallback_reason = None
            return matrix
        except Exception as exc:
            if not self.fallback_model or self.fallback_model == self.primary_model:
                raise EmbeddingError(str(exc)) from exc
            self.last_fallback_reason = str(exc)[:500]
        try:
            matrix, _ = self._embed_with_model(items, self.fallback_model)
        except Exception as exc:
            raise EmbeddingError('主模型和备用模型均失败: ' + str(exc)) from exc
        self.active_model = self.fallback_model
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


EmbeddingCall = Callable[[str, list[str]], list[list[float]]]
