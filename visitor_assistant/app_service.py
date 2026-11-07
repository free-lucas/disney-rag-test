from __future__ import annotations

from dataclasses import dataclass

from config import ConfigurationError, Settings, load_settings
from generation.answer import AnswerGenerator, StreamingAnswer, stream_answer
from models import Answer
from retrieval.embed import DashScopeEmbedder, EmbeddingCache
from retrieval.hybrid import HybridRetriever
from retrieval.rewrite import QueryRewriter
from retrieval.store import IndexLoadError, IndexStore


class ServiceError(RuntimeError):
    pass


def _friendly_configuration_error(exc: ConfigurationError) -> str:
    message = str(exc)
    if '缺少必需环境变量' in message:
        names = message.split(':', 1)[-1].strip()
        return (
            '缺少 API 密钥：' + names + '。'
            '请复制 .env.example 为 .env，填写密钥后重启应用；不要在代码中硬编码密钥。'
        )
    if '知识库目录不存在' in message:
        return message + '。请检查 DISNEY_KB_ROOT 配置。'
    if '内容策略文件不存在' in message:
        return message + '。请检查 CONTENT_POLICY_PATH 配置。'
    return '配置错误：' + message


@dataclass
class RagService:
    settings: Settings
    retriever: HybridRetriever
    generator: AnswerGenerator

    def retrieve(self, question: str, history: list[dict[str, str]] | None = None) -> object:
        try:
            return self.retriever.retrieve(question, history)
        except Exception as exc:
            raise ServiceError('检索服务暂时不可用，请稍后重试。') from exc

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> Answer:
        bundle = self.retrieve(question, history)
        rewrite = getattr(bundle, 'rewrite', None)
        rewritten = getattr(rewrite, 'rewritten_query', '') if rewrite is not None else ''
        return self.generator.generate(
            question,
            list(getattr(bundle, 'results', [])),
            history,
            rewritten_query=rewritten,
        )

    def stream(self, question: str, history: list[dict[str, str]] | None = None) -> StreamingAnswer:
        bundle = self.retrieve(question, history)
        rewrite = getattr(bundle, 'rewrite', None)
        rewritten = getattr(rewrite, 'rewritten_query', '') if rewrite is not None else ''
        return stream_answer(
            self.generator,
            question,
            list(getattr(bundle, 'results', [])),
            history,
            rewritten_query=rewritten,
        )


def build_service(settings: Settings | None = None) -> RagService:
    if settings is None:
        try:
            settings = load_settings(require_api_keys=True)
        except ConfigurationError as exc:
            raise ServiceError(_friendly_configuration_error(exc)) from exc
    try:
        settings.validate_knowledge_base()
    except ConfigurationError as exc:
        raise ServiceError(_friendly_configuration_error(exc)) from exc

    settings.ensure_directories()
    try:
        store = IndexStore.load(settings.index_dir)
    except IndexLoadError as exc:
        raise ServiceError(
            '索引不可用：' + str(exc) + '。'
            '请先在本目录运行 python build_index.py 构建索引。'
        ) from exc

    embedder = DashScopeEmbedder.from_settings(
        settings,
        cache=EmbeddingCache(settings.cache_dir / 'embeddings'),
    )
    rewriter = QueryRewriter.from_settings(settings)
    retriever = HybridRetriever.from_settings(
        settings,
        store=store,
        embedder=embedder,
        rewriter=rewriter,
    )
    generator = AnswerGenerator.from_settings(settings)
    return RagService(settings=settings, retriever=retriever, generator=generator)


_SERVICE: RagService | None = None


def get_service() -> RagService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = build_service()
    return _SERVICE


def reset_service_cache() -> None:
    global _SERVICE
    _SERVICE = None


def public_config() -> dict[str, object]:
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        raise ServiceError(_friendly_configuration_error(exc)) from exc
    summary = settings.public_summary()
    summary['index_exists'] = (settings.index_dir / 'manifest.json').is_file()
    return summary
