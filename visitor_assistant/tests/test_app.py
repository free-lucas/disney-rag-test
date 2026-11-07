from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app_service
from app_service import RagService, ServiceError, build_service
from config import ConfigurationError, Settings
from models import Answer, Citation, ImageRef
from retrieval.store import IndexLoadError


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


class _FakeStream:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer

    def __iter__(self):
        for piece in [self.answer.text]:
            if piece:
                yield piece

    def finalize(self) -> Answer:
        return self.answer


class _FakeService:
    def __init__(self, answers: list[Answer]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def stream(self, question: str, history: list[dict[str, str]] | None = None) -> _FakeStream:
        self.calls.append((question, list(history or [])))
        if not self.answers:
            raise ServiceError('没有预置回答')
        return _FakeStream(self.answers.pop(0))


def _answer(
    text: str = '一日票价格为 475 元 [1]。',
    *,
    citations: list[Citation] | None = None,
    images: list[ImageRef] | None = None,
    refused: bool = False,
    error: str | None = None,
) -> Answer:
    return Answer(
        text=text,
        citations=list(citations or []),
        images=list(images or []),
        refused=refused,
        query='q',
        rewritten_query='q',
        error=error,
    )


def _run_app(monkeypatch: pytest.MonkeyPatch, service: object) -> AppTest:
    monkeypatch.setattr(app_service, 'get_service', lambda: service)
    app = AppTest.from_file(str(APP_PATH))
    app.run()
    return app


def test_chat_appends_messages_and_passes_history(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService([
        _answer('一日票价格为 475 元 [1]。'),
        _answer('迪士尼乐园酒店提供客房、泳池和健身房 [1]。'),
    ])
    app = _run_app(monkeypatch, service)

    app.chat_input[0].set_value('门票多少钱').run()

    assert not app.exception
    assert len(app.chat_message) == 2
    assert any('475' in item.value for item in app.markdown)
    assert app.session_state['history'][0] == {'role': 'user', 'content': '门票多少钱'}

    app.chat_input[0].set_value('那酒店呢').run()

    assert len(app.chat_message) == 4
    second_history = service.calls[1][1]
    assert any(item['role'] == 'user' and '门票多少钱' in item['content'] for item in second_history)
    assert any(item['role'] == 'assistant' and '475' in item['content'] for item in second_history)


def test_sources_are_collapsible_and_hidden_for_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    citations = [
        Citation(source_file='票务.docx', section_path=['门票价格'], page=2),
        Citation(source_file='酒店信息.docx', section_path=['酒店房型'], slide=4),
    ]
    service = _FakeService([
        _answer('门票和酒店信息 [1][2]。', citations=citations),
        _answer('抱歉，公开知识库中没有足够信息回答这个问题。', refused=True),
    ])
    app = _run_app(monkeypatch, service)

    app.chat_input[0].set_value('门票和酒店').run()

    labels = [expander.label for expander in app.expander]
    assert '来源（2）' in labels
    markdown = '\n'.join(item.value for item in app.markdown)
    assert '票务.docx' in markdown and '第2页' in markdown
    assert '酒店信息.docx' in markdown and '第4张幻灯片' in markdown

    app.chat_input[0].set_value('域外问题').run()

    assert '来源（0）' not in [expander.label for expander in app.expander]
    assert '知识库中没有足够信息' in '\n'.join(item.value for item in app.markdown)


def test_images_render_and_missing_files_show_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    image_path = tmp_path / '邮轮价格.png'
    Image.new('RGB', (80, 80), 'navy').save(image_path)
    service = _FakeService([
        _answer('内舱 735 美元起 [1]。', images=[ImageRef(str(image_path), '邮轮价格表')]),
        _answer('图片示例 [1]。', images=[ImageRef(str(tmp_path / 'missing.png'), '缺失图片')]),
    ])
    app = _run_app(monkeypatch, service)

    app.chat_input[0].set_value('邮轮价格').run()
    assert len(app.get('imgs')) == 1

    app.chat_input[0].set_value('再看一张图').run()
    captions = '\n'.join(item.value for item in app.caption)
    assert '图片暂不可用' in captions
    assert 'missing.png' in captions


def test_new_conversation_clears_messages_and_history(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService([_answer('一日票价格为 475 元 [1]。')])
    app = _run_app(monkeypatch, service)
    app.chat_input[0].set_value('门票多少钱').run()
    assert app.session_state['messages']

    app.sidebar.button[0].click().run()

    assert app.session_state['messages'] == []
    assert app.session_state['history'] == []
    assert len(app.chat_message) == 0


def test_service_error_shows_message_and_keeps_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken() -> object:
        raise ServiceError('索引不可用：索引文件缺失: manifest.json。请先运行 python build_index.py 构建索引。')

    monkeypatch.setattr(app_service, 'get_service', _broken)
    app = AppTest.from_file(str(APP_PATH))
    app.run()
    app.chat_input[0].set_value('门票多少钱').run()

    errors = '\n'.join(item.value for item in app.error)
    assert '索引不可用' in errors
    assert 'build_index.py' in errors
    assert app.session_state['messages'][0]['content'] == '门票多少钱'

    service = _FakeService([_answer('恢复后的回答 [1]。')])
    monkeypatch.setattr(app_service, 'get_service', lambda: service)
    app.chat_input[0].set_value('再试一次').run()

    assert app.session_state['messages'][0]['content'] == '门票多少钱'
    assert any('恢复后的回答' in item.value for item in app.markdown)


def test_generation_error_shows_warning_without_breaking_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService([
        _answer('抱歉，回答生成暂时不可用，请稍后重试。', refused=True, error='upstream_timeout'),
        _answer('第二次回答正常 [1]。'),
    ])
    app = _run_app(monkeypatch, service)

    app.chat_input[0].set_value('门票').run()

    warnings = '\n'.join(item.value for item in app.warning)
    assert '回答生成暂时不可用' in warnings

    app.chat_input[0].set_value('再问一次').run()

    assert any('第二次回答正常' in item.value for item in app.markdown)


def test_missing_api_keys_are_reported_without_exposing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)

    with pytest.raises(ServiceError) as captured:
        build_service()

    message = str(captured.value)
    assert '缺少 API 密钥' in message
    assert 'DEEPSEEK_API_KEY' in message
    assert 'sk-' not in message


def test_missing_index_is_reported_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kb = tmp_path / 'kb'
    kb.mkdir()
    policy = tmp_path / 'policy.yaml'
    policy.write_text('version: 1\n', encoding='utf-8')
    settings = Settings(
        deepseek_api_key='test',
        dashscope_api_key='test',
        knowledge_base_root=kb,
        content_policy_path=policy,
        data_dir=tmp_path / 'data',
        index_dir=tmp_path / 'data' / 'index',
        cache_dir=tmp_path / 'data' / 'cache',
        reports_dir=tmp_path / 'data' / 'reports',
    )

    def _missing(directory: object) -> object:
        raise IndexLoadError('索引文件缺失: manifest.json')

    monkeypatch.setattr(app_service.IndexStore, 'load', _missing)

    with pytest.raises(ServiceError) as captured:
        build_service(settings)

    assert '索引不可用' in str(captured.value)
    assert 'build_index.py' in str(captured.value)


def test_retrieval_failure_is_wrapped_as_friendly_service_error() -> None:
    class _BrokenRetriever:
        def retrieve(self, question: str, history: object = None) -> object:
            raise RuntimeError('embedding endpoint down')

    service = RagService(
        settings=Settings(deepseek_api_key='test', dashscope_api_key='test'),
        retriever=_BrokenRetriever(),  # type: ignore[arg-type]
        generator=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ServiceError) as captured:
        service.ask('门票')

    assert '检索服务暂时不可用' in str(captured.value)


def test_ui_reads_configuration_and_contains_no_hardcoded_secret() -> None:
    app_source = APP_PATH.read_text(encoding='utf-8')
    service_source = (APP_PATH.parent / 'app_service.py').read_text(encoding='utf-8')

    assert 'sk-' not in app_source
    assert 'sk-' not in service_source
    assert 'DEEPSEEK_API_KEY' not in app_source
    assert 'public_config' in app_source
    assert 'public_summary' in service_source
