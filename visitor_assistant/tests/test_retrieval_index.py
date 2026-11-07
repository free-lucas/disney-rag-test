import hashlib
import json
from pathlib import Path

import numpy as np

from config import Settings
from ingest.build import build_index
from retrieval.bm25 import BM25Index, tokenize
from retrieval.embed import DashScopeEmbedder, EmbeddingCache
from retrieval.faiss_io import read_index
from retrieval.store import IndexStore


def _make_docx(path: Path, heading: str, body: str) -> None:
    from docx import Document

    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading(heading, level=1)
    document.add_paragraph(body)
    document.save(str(path))


class _NoCaptioner:
    model = 'deepseek-flash'
    prompt_version = 'v1'

    def caption_many(self, images: list[object]) -> object:
        from ingest.caption import CaptionBatchResult

        return CaptionBatchResult(results=[], captions={}, cached_count=0, api_calls=0)


def _settings(tmp_path: Path, kb: Path) -> Settings:
    return Settings(
        deepseek_api_key='test',
        dashscope_api_key='test',
        knowledge_base_root=kb,
        data_dir=tmp_path / '数据',
        index_dir=tmp_path / '数据' / '索引',
        cache_dir=tmp_path / '数据' / '缓存',
        reports_dir=tmp_path / '数据' / '报告',
        chunk_target_chars=120,
        chunk_min_chars=80,
        chunk_max_chars=160,
        chunk_overlap_chars=20,
        image_max_per_document=5,
    )


def _vector(text: str, dimensions: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    vector = [0.01] * dimensions
    for index in range(min(6, len(digest))):
        vector[index % dimensions] += float(digest[index] % 13 + 1)
    return vector


def _embedder(tmp_path: Path, calls: list[tuple[str, list[str]]] | None = None) -> DashScopeEmbedder:
    recorded = calls if calls is not None else []

    def call(model: str, texts: list[str]) -> list[list[float]]:
        recorded.append((model, list(texts)))
        return [_vector(text) for text in texts]

    return DashScopeEmbedder(
        api_key='test',
        cache=EmbeddingCache(tmp_path / 'embed-cache'),
        max_retries=0,
        call_fn=call,
    )


def test_embedding_normalizes_vectors_and_uses_cache(tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []
    embedder = _embedder(tmp_path, calls)

    first = embedder.embed_texts(['上海迪士尼门票', '迪士尼酒店'])
    second = embedder.embed_texts(['上海迪士尼门票', '迪士尼酒店'])

    assert first.shape == (2, 8)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.allclose(first, second)
    assert len(calls) == 1


def test_embedding_falls_back_and_reports_reason(tmp_path: Path) -> None:
    calls: list[str] = []

    def call(model: str, texts: list[str]) -> list[list[float]]:
        calls.append(model)
        if model == 'text-embedding-v4':
            raise RuntimeError('model unavailable')
        return [_vector(text) for text in texts]

    embedder = DashScopeEmbedder(
        api_key='test',
        primary_model='text-embedding-v4',
        fallback_model='text-embedding-v1',
        cache=EmbeddingCache(tmp_path / 'cache'),
        max_retries=0,
        call_fn=call,
    )
    matrix = embedder.embed_texts(['邮轮价格'])

    assert matrix.shape == (1, 8)
    assert embedder.active_model == 'text-embedding-v1'
    assert 'model unavailable' in (embedder.last_fallback_reason or '')
    assert calls == ['text-embedding-v4', 'text-embedding-v1']


def test_bm25_round_trip_and_exact_term_search(tmp_path: Path) -> None:
    index = BM25Index.from_texts([
        '迪士尼乐园一日票和两日票可以在官方渠道购买。',
        '酒店提供客房、泳池和健身房。',
        '邮轮价格表包含内舱和海景房。',
    ])
    path = index.save(tmp_path / 'bm25.pkl')
    loaded = BM25Index.load(path)

    results = loaded.search('一日票 价格', k=2)

    assert results
    assert results[0][0] == 0
    assert '门票' in tokenize('门票多少钱')


def test_build_creates_all_artifacts_and_reloads_same_top_result(tmp_path: Path) -> None:
    kb = tmp_path / 'kb'
    _make_docx(kb / '1-产品与服务详情' / '票务.docx', '门票价格', '上海迪士尼一日票和两日票请以官方渠道价格为准。')
    _make_docx(kb / '5-内部知识与工具' / '员工票.docx', '员工票', '员工票仅限内部员工使用，需出示工号。')
    settings = _settings(tmp_path, kb)
    embedder = _embedder(tmp_path)

    report = build_index(settings, embedder=embedder, captioner=_NoCaptioner(), allow_com=False)

    index_dir = settings.index_dir
    assert (index_dir / 'faiss.index').is_file()
    assert (index_dir / 'chunks.jsonl').is_file()
    assert (index_dir / 'bm25.pkl').is_file()
    assert (index_dir / 'manifest.json').is_file()
    assert Path(settings.reports_dir / 'build_report.json').is_file()
    assert report.public_chunks >= 1
    assert report.internal_chunks >= 1
    assert report.indexed_files == 2
    manifest = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['embedding_dim'] == 8
    assert manifest['embedding_model'] == 'text-embedding-v4'

    store = IndexStore.load(index_dir)
    faiss_index = store.faiss_index
    assert faiss_index.ntotal == len(store.chunks)
    query = embedder.embed_query('一日票价格').reshape(1, -1).astype('float32')
    scores, ids = faiss_index.search(query, 3)
    chunks = [json.loads(line) for line in (index_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    bm25 = BM25Index.load(index_dir / 'bm25.pkl')

    assert ids[0][0] >= 0
    bm25_results = bm25.search('门票价格', k=2)
    assert bm25_results
    assert '票务.docx' in chunks[bm25_results[0][0]]['source_file']
    assert np.array_equal(faiss_index.search(query, 3)[1], read_index(index_dir / 'faiss.index').search(query, 3)[1])


def test_incremental_build_reuses_unchanged_and_removes_deleted(tmp_path: Path) -> None:
    kb = tmp_path / 'kb'
    public_path = kb / '1-产品与服务详情' / '票务.docx'
    internal_path = kb / '5-内部知识与工具' / '员工票.docx'
    _make_docx(public_path, '门票', '一日票和两日票在官方渠道购买。')
    _make_docx(internal_path, '员工票', '员工票仅限内部员工使用。')
    settings = _settings(tmp_path, kb)
    calls: list[tuple[str, list[str]]] = []
    embedder = _embedder(tmp_path, calls)

    first = build_index(settings, embedder=embedder, captioner=_NoCaptioner(), allow_com=False)
    calls_after_first = len(calls)
    second = build_index(settings, embedder=embedder, captioner=_NoCaptioner(), allow_com=False)

    assert len(second.reused_files) == 2
    assert not second.rebuilt_files
    assert len(calls) == calls_after_first
    assert first.public_chunks == second.public_chunks

    _make_docx(public_path, '门票', '一日票、两日票和年票都在官方渠道购买，价格可能调整。')
    third = build_index(settings, embedder=embedder, captioner=_NoCaptioner(), allow_com=False)
    assert '1-产品与服务详情/票务.docx' in third.rebuilt_files
    assert '5-内部知识与工具/员工票.docx' in third.reused_files
    assert len(calls) > calls_after_first

    internal_path.unlink()
    fourth = build_index(settings, embedder=embedder, captioner=_NoCaptioner(), allow_com=False)
    assert '5-内部知识与工具/员工票.docx' in fourth.removed_files
