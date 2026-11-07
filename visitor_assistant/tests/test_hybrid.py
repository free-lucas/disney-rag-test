import json
from pathlib import Path
from types import SimpleNamespace

from config import Settings
from ingest.build import build_index
from models import Audience, Chunk, ChunkType, RetrievalResult
from retrieval.embed import DashScopeEmbedder, EmbeddingCache
from retrieval.hybrid import HybridRetriever, associate_relevant_images, rrf_fuse
from retrieval.rerank import DashScopeReranker, rerank_results
from retrieval.rewrite import QueryRewriter, RewriteResult, parse_rewrite_response, split_intents
from retrieval.store import IndexStore


def _make_docx(path: Path, heading: str, body: str) -> None:
    from docx import Document

    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading(heading, level=1)
    document.add_paragraph(body)
    document.save(str(path))


def _png(path: Path) -> Path:
    from PIL import Image

    Image.new('RGB', (300, 300), 'navy').save(path)
    return path


def _semantic_vector(text: str) -> list[float]:
    dimensions = {
        '门票': 0,
        '价格': 1,
        '酒店': 2,
        '邮轮': 3,
        '员工': 4,
        '停车': 5,
        '内舱': 6,
        '地图': 7,
    }
    vector = [0.05] * 8
    for token, index in dimensions.items():
        if token in text:
            vector[index] += 1.0
    return vector


class _FakeCaptioner:
    model = 'deepseek-flash'
    prompt_version = 'v1'

    def caption_many(self, images: list[object]) -> object:
        from ingest.caption import CaptionBatchResult, CaptionResult

        results = []
        captions = {}
        for image in images:
            caption = '迪士尼邮轮价格表显示内舱 735 美元起，海景房价格更高。'
            results.append(CaptionResult(image.path, caption=caption))
            captions[image.path] = caption
        return CaptionBatchResult(results=results, captions=captions, api_calls=len(images))


def _settings(tmp_path: Path, kb: Path) -> Settings:
    return Settings(
        deepseek_api_key='test',
        dashscope_api_key='test',
        knowledge_base_root=kb,
        data_dir=tmp_path / 'data',
        index_dir=tmp_path / 'data' / 'index',
        cache_dir=tmp_path / 'data' / 'cache',
        reports_dir=tmp_path / 'data' / 'reports',
        chunk_target_chars=120,
        chunk_min_chars=80,
        chunk_max_chars=160,
        chunk_overlap_chars=20,
        image_max_per_document=5,
    )


def _build_store(tmp_path: Path) -> tuple[IndexStore, Settings]:
    kb = tmp_path / 'kb'
    _make_docx(kb / '1-产品与服务详情' / '票务.docx', '门票价格', '一日票和两日票可在官方渠道购买，门票价格可能调整。')
    _make_docx(kb / '2-运营流程与标准作业程序' / '酒店信息.docx', '迪士尼酒店', '酒店提供客房、泳池和健身房。')
    _make_docx(kb / '5-内部知识与工具' / '员工票.docx', '员工票', '员工票仅限内部员工使用，入园需出示工号。')
    _png(kb / '1-产品与服务详情' / '迪士尼邮轮价格.jpeg')
    settings = _settings(tmp_path, kb)
    embedder = DashScopeEmbedder(
        api_key='test',
        cache=EmbeddingCache(tmp_path / 'embed-cache'),
        max_retries=0,
        call_fn=lambda model, texts: [_semantic_vector(text) for text in texts],
    )
    build_index(settings, embedder=embedder, captioner=_FakeCaptioner(), allow_com=False)
    return IndexStore.load(settings.index_dir), settings


class _NoRewrite:
    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> RewriteResult:
        return RewriteResult(query, query, [query])


def test_rewrite_resolves_pronoun_and_preserves_standalone_question() -> None:
    resolved = parse_rewrite_response(
        json.dumps({'rewritten_query': '上海迪士尼乐园酒店多少钱', 'sub_queries': ['上海迪士尼乐园酒店价格'], 'needs_history': True}, ensure_ascii=False),
        '那它多少钱',
        has_history=True,
    )
    standalone = parse_rewrite_response(
        json.dumps({'rewritten_query': '停车怎么收费', 'sub_queries': ['停车费用'], 'needs_history': True}, ensure_ascii=False),
        '停车怎么收费',
        has_history=False,
    )

    assert resolved.rewritten_query == '上海迪士尼乐园酒店多少钱'
    assert resolved.needs_history
    assert standalone.rewritten_query == '停车怎么收费'
    assert not standalone.needs_history


def test_rewrite_invalid_json_falls_back_and_split_intents() -> None:
    class _BrokenCompletions:
        def create(self, **kwargs: object) -> object:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='not json'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=_BrokenCompletions()))
    rewriter = QueryRewriter(api_key='test', client=client, max_retries=0)
    result = rewriter.rewrite('门票多少钱？停车怎么收费？')

    assert result.error
    assert result.rewritten_query == '门票多少钱？停车怎么收费？'
    assert result.sub_queries == ['门票多少钱', '停车怎么收费']
    assert split_intents('只有一个问题') == ['只有一个问题']


def test_retriever_filters_internal_before_both_channels(tmp_path: Path) -> None:
    store, settings = _build_store(tmp_path)
    retriever = HybridRetriever(
        store=store,
        embedder=DashScopeEmbedder(
            api_key='test',
            cache=EmbeddingCache(tmp_path / 'query-cache'),
            max_retries=0,
            call_fn=lambda model, texts: [_semantic_vector(text) for text in texts],
        ),
        rewriter=_NoRewrite(),
        top_k=5,
        dense_top_k=10,
        bm25_top_k=10,
    )

    bundle = retriever.retrieve('员工票使用规则')

    assert store.public_faiss.ntotal == len(store.public_chunks)
    assert all(result.chunk.audience is Audience.PUBLIC for result in bundle.results)
    assert bundle.results == []

    public_bundle = retriever.retrieve('门票价格')

    assert public_bundle.results
    assert all('员工票' not in result.chunk.content for result in public_bundle.results)
    assert all(result.chunk.audience is Audience.PUBLIC for result in public_bundle.results)


def _chunk(identifier: str, content: str, audience: Audience = Audience.PUBLIC) -> Chunk:
    return Chunk(identifier, content, 'doc.docx', '1-产品与服务详情', audience)


def test_rrf_merges_both_channels_and_deduplicates() -> None:
    first = _chunk('a', '门票价格说明')
    second = _chunk('b', '停车收费标准')
    third = _chunk('c', '酒店入住政策')
    dense = [
        RetrievalResult(first, score=0.9, dense_score=0.9),
        RetrievalResult(second, score=0.8, dense_score=0.8),
    ]
    bm25 = [
        RetrievalResult(second, score=12.0, bm25_score=12.0),
        RetrievalResult(third, score=8.0, bm25_score=8.0),
    ]

    fused = rrf_fuse(dense, bm25, rrf_k=60, top_k=3)

    assert [result.chunk.id for result in fused][0] == 'b'
    assert len({result.chunk.content_hash for result in fused}) == 3
    assert fused[0].dense_score is not None and fused[0].bm25_score is not None


def test_rerank_reorders_but_preserves_fused_score() -> None:
    first = RetrievalResult(_chunk('a', '一般相关'), score=0.7)
    second = RetrievalResult(_chunk('b', '高度相关'), score=0.2)
    reranker = DashScopeReranker(api_key='test', call_fn=lambda query, documents: [0.1, 0.9])

    ranked = rerank_results('酒店', [first, second], reranker=reranker, top_k=2)

    assert [result.chunk.id for result in ranked] == ['b', 'a']
    assert ranked[0].score == 0.2
    assert ranked[0].rerank_score == 0.9


def test_image_association_returns_caption_and_path(tmp_path: Path) -> None:
    store, _ = _build_store(tmp_path)
    text_result = next(RetrievalResult(chunk) for chunk in store.public_chunks if chunk.chunk_type is ChunkType.TEXT)

    results = associate_relevant_images([text_result], store, '迪士尼邮轮价格')

    images = [result.chunk for result in results if result.chunk.chunk_type is ChunkType.IMAGE_CAPTION]
    assert images
    assert images[0].image_path and Path(images[0].image_path).is_file()
    assert '735' in (images[0].caption or '')
