from types import SimpleNamespace

from generation.answer import AnswerGenerator, parse_generation_response, stream_answer
from generation.prompt import (
    ANSWER_SYSTEM_PROMPT,
    CONFIDENTIALITY_REFUSAL_MESSAGE,
    REFUSAL_MESSAGE,
    TIME_SENSITIVE_DISCLAIMER,
    build_messages,
    find_internal_leak,
    format_evidence,
    is_time_sensitive,
)
from models import Audience, Chunk, ChunkType, RetrievalResult


def _result(
    identifier: str = 'c1',
    content: str = '一日票价格为 475 元。',
    *,
    source_file: str = '1-产品与服务详情/票务.docx',
    section_path: list[str] | None = None,
    page: int | None = None,
    slide: int | None = None,
    chunk_type: ChunkType = ChunkType.TEXT,
    image_path: str | None = None,
    audience: Audience = Audience.PUBLIC,
) -> RetrievalResult:
    chunk = Chunk(
        identifier,
        content,
        source_file,
        '1-产品与服务详情',
        audience,
        chunk_type=chunk_type,
        section_path=section_path if section_path is not None else ['门票价格'],
        page=page,
        slide=slide,
        image_path=image_path,
        caption=content if chunk_type is ChunkType.IMAGE_CAPTION else None,
    )
    return RetrievalResult(chunk=chunk, score=0.9, dense_score=0.8, bm25_score=4.0)


class _Completions:
    def __init__(self, contents: list[str] | None = None, stream_events: list[str] | None = None) -> None:
        self.contents = list(contents or [])
        self.stream_events = list(stream_events or [])
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if kwargs.get('stream'):
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
                for piece in self.stream_events
            ])
        content = self.contents.pop(0) if self.contents else 'ANSWER\n默认回答 [1]。'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _client(contents: list[str] | None = None, stream_events: list[str] | None = None) -> tuple[object, _Completions]:
    completions = _Completions(contents, stream_events)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _generator(contents: list[str] | None = None, **kwargs: object) -> tuple[AnswerGenerator, _Completions]:
    client, completions = _client(contents)
    return AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None, **kwargs), completions


def test_system_prompt_contains_all_guardrails() -> None:
    required = [
        '只使用“公开证据”',
        '不编造',
        '证据编号',
        'REFUSE',
        '实时天气',
        '员工规则',
        '升级路径',
        '内部电话',
        '员工联系方式',
        '官方渠道当日信息为准',
        '不要因为证据不完整',
        '不要沿用无关的旧话题',
        'ANSWER 或 REFUSE',
    ]
    for phrase in required:
        assert phrase in ANSWER_SYSTEM_PROMPT

    messages = build_messages('门票多少钱', [_result()], [{'role': 'user', 'content': '之前问了酒店'}])

    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == ANSWER_SYSTEM_PROMPT
    assert '游客问题：门票多少钱' in messages[1]['content']
    assert '公开证据：' in messages[1]['content']
    assert '[1] 文件：1-产品与服务详情/票务.docx' in messages[1]['content']
    assert '位置：1-产品与服务详情/票务.docx（门票价格）' in messages[1]['content']
    assert '游客：之前问了酒店' in messages[1]['content']


def test_evidence_excludes_internal_chunks() -> None:
    public = _result('public', '公开门票信息')
    internal = _result('internal', '员工票内部规则', audience=Audience.INTERNAL)

    text = format_evidence([internal, public])

    assert '公开门票信息' in text
    assert '员工票内部规则' not in text
    assert text.count('文件：') == 1
    assert text.startswith('[1] 文件：')


def test_single_source_citation_contains_file_and_location() -> None:
    generator, _ = _generator(['ANSWER\n一日票价格为 475 元 [1]。'])
    result = _result(page=3)

    answer = generator.generate('门票多少钱', [result])

    assert not answer.refused
    assert len(answer.citations) == 1
    assert answer.citations[0].source_file.endswith('票务.docx')
    assert '门票价格' in answer.citations[0].label
    assert '第3页' in answer.citations[0].label


def test_multiple_sources_are_all_listed() -> None:
    generator, _ = _generator(['ANSWER\n门票见 [1]，酒店见 [2]。'])
    ticket = _result('ticket', '门票信息', source_file='票务.docx', page=2)
    hotel = _result('hotel', '酒店信息', source_file='酒店信息.docx', section_path=['酒店房型'], slide=7)

    answer = generator.generate('门票和酒店', [ticket, hotel])

    assert len(answer.citations) == 2
    labels = ' | '.join(citation.label for citation in answer.citations)
    assert '票务.docx' in labels and '第2页' in labels
    assert '酒店信息.docx' in labels and '第7张幻灯片' in labels


def test_answer_without_markers_still_cites_contributing_sources() -> None:
    generator, _ = _generator(['ANSWER\n一日票可以现场购买。'])
    results = [_result('a', '门票信息'), _result('b', '购票渠道', source_file='官网订票.docx')]

    answer = generator.generate('怎么买票', results)

    assert not answer.refused
    assert len(answer.citations) == 2


def test_no_evidence_refuses_without_calling_model() -> None:
    class _ExplodingCompletions:
        def create(self, **kwargs: object) -> object:
            raise AssertionError('没有证据时不应调用生成模型')

    client = SimpleNamespace(chat=SimpleNamespace(completions=_ExplodingCompletions()))
    generator = AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None)

    answer = generator.generate('今天火星天气怎么样', [])

    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert answer.citations == []
    assert answer.images == []


def test_model_refusal_is_respected() -> None:
    generator, _ = _generator(['REFUSE'])

    answer = generator.generate('域外问题', [_result()])

    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert answer.citations == []


def test_volatile_topics_get_disclaimer_but_stable_topics_do_not() -> None:
    volatile, _ = _generator(['ANSWER\n一日票价格为 475 元 [1]。'])
    volatile_answer = volatile.generate('门票多少钱', [_result()])

    assert volatile_answer.disclaimer
    assert TIME_SENSITIVE_DISCLAIMER in volatile_answer.text

    stable, _ = _generator(['ANSWER\n酒店提供泳池和健身房 [1]。'])
    stable_result = _result('hotel', '酒店提供泳池和健身房。', source_file='酒店信息.docx', section_path=['设施'])
    stable_answer = stable.generate('酒店有哪些设施', [stable_result])

    assert not stable_answer.disclaimer
    assert TIME_SENSITIVE_DISCLAIMER not in stable_answer.text


def test_internal_leak_is_blocked() -> None:
    leaked = '员工票仅限员工使用，入园需出示工号，有问题分机 8123。'
    assert find_internal_leak(leaked)

    generator, _ = _generator(['ANSWER\n' + leaked + ' [1]。'])
    answer = generator.generate('员工票规则', [_result(content=leaked)])

    assert answer.refused
    assert answer.text == CONFIDENTIALITY_REFUSAL_MESSAGE
    assert '员工票' not in answer.text
    assert '8123' not in answer.text
    assert answer.citations == []


def test_internal_results_are_never_sent_to_model() -> None:
    internal = _result('internal', '员工票内部规则', audience=Audience.INTERNAL)
    generator, completions = _generator()

    answer = generator.generate('员工票规则', [internal])

    assert answer.refused
    assert completions.calls == []


def test_image_evidence_is_returned_with_caption_and_path() -> None:
    image = _result(
        'image',
        '迪士尼邮轮价格表显示内舱 735 美元起。',
        source_file='迪士尼邮轮价格.jpeg',
        chunk_type=ChunkType.IMAGE_CAPTION,
        image_path='F:/kb/迪士尼邮轮价格.jpeg',
    )
    generator, _ = _generator(['ANSWER\n内舱 735 美元起 [1]。'])

    answer = generator.generate('邮轮价格', [image])

    assert not answer.refused
    assert len(answer.images) == 1
    assert answer.images[0].image_path.endswith('迪士尼邮轮价格.jpeg')
    assert '735' in answer.images[0].caption

    refused, _ = _generator(['REFUSE'])
    refused_answer = refused.generate('邮轮价格', [image])

    assert refused_answer.refused
    assert refused_answer.images == []


def test_images_follow_cited_evidence_only() -> None:
    cruise_image = _result(
        'cruise-image',
        '邮轮价格图片：内舱 735 美元起。',
        source_file='迪士尼邮轮价格7.jpeg',
        section_path=['邮轮价格'],
        chunk_type=ChunkType.IMAGE_CAPTION,
        image_path='F:/kb/迪士尼邮轮价格7.jpeg',
    )
    hotel_image = _result(
        'hotel-image',
        '酒店房型图片：标准房与套房。',
        source_file='酒店信息.docx',
        section_path=['酒店房型'],
        chunk_type=ChunkType.IMAGE_CAPTION,
        image_path='F:/kb/hotel_lobby.png',
    )
    hotel_text = _result(
        'hotel-text',
        '酒店入住时间为 15:00 后。',
        source_file='酒店信息.docx',
        section_path=['入住/退房政策'],
    )
    generator, _ = _generator(['ANSWER\n入住时间为 15:00 后 [3]，房型图片见 [2]。'])

    answer = generator.generate('酒店入住政策', [cruise_image, hotel_image, hotel_text])

    assert [image.image_path for image in answer.images] == ['F:/kb/hotel_lobby.png']
    cited_files = [citation.source_file for citation in answer.citations]
    assert '迪士尼邮轮价格7.jpeg' not in cited_files
    assert '酒店信息.docx' in cited_files


def test_retry_uses_exponential_backoff_and_recovers() -> None:
    class _FlakyCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError('rate limited')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ANSWER\n成功 [1]。'))])

    sleeps: list[float] = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=_FlakyCompletions()))
    generator = AnswerGenerator(
        api_key='test',
        client=client,
        max_retries=2,
        sleep_fn=sleeps.append,
        retry_base_seconds=0.5,
    )

    answer = generator.generate('门票', [_result()])

    assert not answer.refused
    assert client.chat.completions.calls == 3
    assert sleeps == [0.5, 1.0]


def test_generation_failure_returns_safe_refusal() -> None:
    class _FailingCompletions:
        def create(self, **kwargs: object) -> object:
            raise RuntimeError('service unavailable')

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FailingCompletions()))
    generator = AnswerGenerator(api_key='test', client=client, max_retries=1, sleep_fn=lambda seconds: None)

    answer = generator.generate('门票', [_result()])

    assert answer.refused
    assert answer.error and 'service unavailable' in answer.error


def test_streaming_answer_hides_decision_line_and_finalizes_citations() -> None:
    client, completions = _client(['ANSWER\n不可见'], ['ANSWER\n', '一日票', '价格 475 元 [1]。'])
    generator = AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None)

    stream = stream_answer(generator, '门票多少钱', [_result(page=4)])
    visible = ''.join(stream)
    answer = stream.finalize()

    assert 'ANSWER' not in visible
    assert '一日票价格 475 元' in visible
    assert not answer.refused
    assert answer.citations[0].page == 4
    assert completions.calls[0]['stream'] is True


def test_streaming_refusal_and_empty_evidence() -> None:
    client, _ = _client([], ['REFUSE', '\n'])
    generator = AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None)

    stream = stream_answer(generator, '域外问题', [_result()])
    visible = ''.join(stream)

    assert visible == REFUSAL_MESSAGE
    assert stream.finalize().refused

    empty = stream_answer(generator, '域外问题', [])
    assert ''.join(empty) == REFUSAL_MESSAGE
    assert empty.finalize().refused


def test_partial_answer_keeps_known_part_and_gap() -> None:
    generator, _ = _generator(['ANSWER\n一日票可以在官方渠道购买 [1]。知识库未覆盖儿童票的具体折扣。'])

    answer = generator.generate('儿童票怎么买', [_result()])

    assert not answer.refused
    assert '一日票可以在官方渠道购买' in answer.text
    assert '知识库未覆盖' in answer.text
    assert answer.citations


def test_history_is_bounded_and_topic_switch_rule_is_present() -> None:
    history = [
        {'role': 'user' if index % 2 == 0 else 'assistant', 'content': '第' + str(index) + '条历史'}
        for index in range(10)
    ]
    messages = build_messages('停车怎么收费', [_result()], history)
    content = messages[1]['content']

    assert '第9条历史' in content
    assert '第0条历史' not in content
    assert '如果本轮是新的独立主题，不要沿用无关的旧话题' in messages[0]['content']


def test_parser_accepts_answer_and_refuse_decisions() -> None:
    assert parse_generation_response('ANSWER\n正文 [1]').decision == 'ANSWER'
    assert parse_generation_response('REFUSE').decision == 'REFUSE'
    assert parse_generation_response('没有标签时按回答处理').body == '没有标签时按回答处理'


def test_empty_completion_is_retried_and_recovers() -> None:
    generator, completions = _generator(['', 'ANSWER\n一日票价格为 475 元 [1]。'])

    answer = generator.generate('门票多少钱', [_result()])

    assert not answer.refused
    assert len(completions.calls) == 2


def test_length_truncated_completion_is_retried() -> None:
    class _TruncatedCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason='length',
                    message=SimpleNamespace(content='ANSWER\n一日票价格'),
                )])
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='stop',
                message=SimpleNamespace(content='ANSWER\n一日票价格为 475 元 [1]。'),
            )])

    completions = _TruncatedCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None)

    answer = generator.generate('门票多少钱', [_result()])

    assert not answer.refused
    assert completions.calls == 2


def test_empty_stream_falls_back_to_non_streaming() -> None:
    image = _result(
        'image',
        '迪士尼邮轮价格表显示内舱 735 美元起。',
        source_file='迪士尼邮轮价格.jpeg',
        chunk_type=ChunkType.IMAGE_CAPTION,
        image_path='F:/kb/迪士尼邮轮价格.jpeg',
    )
    client, completions = _client(['ANSWER\n内舱 735 美元起 [1]。'], [])
    generator = AnswerGenerator(api_key='test', client=client, sleep_fn=lambda seconds: None)

    stream = stream_answer(generator, '邮轮价格', [image])
    visible = ''.join(stream)
    answer = stream.finalize()

    assert '735' in visible
    assert not answer.refused
    assert answer.images
    assert completions.calls[0]['stream'] is True
    assert completions.calls[-1].get('stream') is None
    assert is_time_sensitive('几点开园')
    assert not is_time_sensitive('酒店有哪些设施')
