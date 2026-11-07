import pytest

from models import Answer, Audience, Chunk, ChunkType, Citation, ImageRef, RetrievalResult, SourceRef


def make_chunk() -> Chunk:
    return Chunk(
        id='doc-1-0001',
        content='上海迪士尼乐园门票信息。',
        source_file='1-产品与服务详情/票务信息.docx',
        category='1-产品与服务详情',
        audience='public',
        chunk_type='text',
        section_path=['票务信息'],
        page=2,
    )


def test_chunk_json_round_trip():
    chunk = make_chunk()
    restored = Chunk.from_json(chunk.to_json())
    assert restored == chunk
    assert restored.audience is Audience.PUBLIC
    assert restored.chunk_type is ChunkType.TEXT
    assert restored.content_hash == chunk.content_hash


def test_missing_optional_fields_and_source_ref():
    chunk = Chunk.from_dict({
        'id': 'x',
        'content': 'x',
        'source_file': 'a.docx',
        'category': 'c',
        'audience': 'internal',
    })
    assert chunk.page is None
    assert chunk.section_path == []
    assert chunk.source_ref().location == 'a.docx'


def test_invalid_audience_rejected():
    with pytest.raises(ValueError):
        Chunk(
            id='x',
            content='x',
            source_file='a.docx',
            category='c',
            audience='staff',
        )


def test_retrieval_and_answer_round_trip():
    result = RetrievalResult(chunk=make_chunk(), score=0.2, dense_score=0.8, bm25_score=3.0)
    restored = RetrievalResult.from_dict(result.to_dict())
    assert restored.chunk == result.chunk
    assert restored.score == 0.2

    answer = Answer(
        text='回答',
        citations=[Citation.from_source_ref(SourceRef('a.docx', section_path=['章节']), 'x')],
        images=[ImageRef(image_path='data/a.jpg', caption='价格表')],
        disclaimer=True,
    )
    assert Answer.from_dict(answer.to_dict()) == answer
