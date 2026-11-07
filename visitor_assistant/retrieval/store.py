from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from models import Audience, Chunk
from retrieval.bm25 import BM25Index
from retrieval.faiss_io import read_index


class IndexLoadError(RuntimeError):
    pass


class IndexStore:
    def __init__(
        self,
        *,
        chunks: list[Chunk],
        faiss_index: object,
        bm25: BM25Index,
        manifest: dict[str, Any],
        directory: Path,
    ) -> None:
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.manifest = manifest
        self.directory = directory
        self.public_indices = [index for index, chunk in enumerate(chunks) if chunk.audience is Audience.PUBLIC]
        if not self.public_indices:
            raise IndexLoadError('索引中没有 public chunk')

        import faiss

        dimension = int(faiss_index.d)
        all_vectors = np.asarray(faiss_index.reconstruct_n(0, faiss_index.ntotal), dtype=np.float32)
        public_vectors = np.ascontiguousarray(all_vectors[self.public_indices], dtype=np.float32)
        faiss.normalize_L2(public_vectors)
        self.public_faiss = faiss.IndexFlatIP(dimension)
        self.public_faiss.add(public_vectors)
        self.public_bm25 = bm25.subset(self.public_indices)

    @classmethod
    def load(cls, directory: str | Path) -> 'IndexStore':
        import faiss

        root = Path(directory)
        manifest_path = root / 'manifest.json'
        chunks_path = root / 'chunks.jsonl'
        faiss_path = root / 'faiss.index'
        bm25_path = root / 'bm25.pkl'
        missing = [path for path in (manifest_path, chunks_path, faiss_path, bm25_path) if not path.is_file()]
        if missing:
            raise IndexLoadError('索引文件缺失: ' + ', '.join(path.name for path in missing))
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            chunks = [
                Chunk.from_json(line)
                for line in chunks_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            faiss_index = read_index(faiss_path)
            bm25 = BM25Index.load(bm25_path)
        except (OSError, ValueError, TypeError) as exc:
            raise IndexLoadError('索引加载失败: ' + str(exc)) from exc
        if not chunks:
            raise IndexLoadError('索引没有 chunk')
        if int(manifest.get('total_chunks', len(chunks))) != len(chunks):
            raise IndexLoadError('manifest 与 chunks.jsonl 数量不一致')
        if faiss_index.ntotal != len(chunks):
            raise IndexLoadError('FAISS 向量数量与 chunk 数量不一致')
        if len(bm25.tokens) != len(chunks):
            raise IndexLoadError('BM25 文档数量与 chunk 数量不一致')
        if int(manifest.get('embedding_dim', faiss_index.d)) != faiss_index.d:
            raise IndexLoadError('manifest 与 FAISS 维度不一致')
        return cls(chunks=chunks, faiss_index=faiss_index, bm25=bm25, manifest=manifest, directory=root)

    @property
    def public_chunks(self) -> list[Chunk]:
        return [self.chunks[index] for index in self.public_indices]

    @property
    def internal_chunks(self) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.audience is Audience.INTERNAL]

    def dense_search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.public_faiss.d:
            raise IndexLoadError('查询向量维度与索引不一致')
        scores, local_ids = self.public_faiss.search(vector, min(k, len(self.public_indices)))
        results: list[tuple[int, float]] = []
        for local_id, score in zip(local_ids[0], scores[0]):
            if int(local_id) < 0:
                continue
            original = self.public_indices[int(local_id)]
            results.append((original, float(score)))
        return results

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        local_results = self.public_bm25.search(query, k=min(k, len(self.public_indices)))
        return [(self.public_indices[local_index], score) for local_index, score in local_results]
