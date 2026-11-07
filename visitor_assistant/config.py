from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_KB_ROOT = (BASE_DIR.parent / '迪士尼RAG知识库（完整）').resolve()
DEFAULT_DATA_DIR = BASE_DIR / 'data'
DEFAULT_POLICY_PATH = BASE_DIR / 'config' / 'content_policy.yaml'


class ConfigurationError(RuntimeError):
    pass


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not str(value).strip() else str(value).strip()


def _integer(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ConfigurationError(name + ' 必须是整数') from exc
    if value < minimum:
        raise ConfigurationError(name + ' 必须大于等于 ' + str(minimum))
    return value


def _number(env: Mapping[str, str], name: str, default: float, minimum: float = 0.0) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ConfigurationError(name + ' 必须是数字') from exc
    if value < minimum:
        raise ConfigurationError(name + ' 必须大于等于 ' + str(minimum))
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    normalized = str(raw).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ConfigurationError(name + ' 必须是 true/false')


def _path(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default.resolve()
    candidate = Path(str(raw).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = field(default='', repr=False)
    dashscope_api_key: str = field(default='', repr=False)
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: str = 'deepseek-flash'
    embedding_model: str = 'text-embedding-v4'
    embedding_fallback_model: str = 'text-embedding-v1'
    knowledge_base_root: Path = DEFAULT_KB_ROOT
    data_dir: Path = DEFAULT_DATA_DIR
    index_dir: Path = DEFAULT_DATA_DIR / 'index'
    cache_dir: Path = DEFAULT_DATA_DIR / 'cache'
    reports_dir: Path = DEFAULT_DATA_DIR / 'reports'
    content_policy_path: Path = DEFAULT_POLICY_PATH
    retrieval_top_k: int = 6
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    dense_min_score: float = 0.25
    rerank_enabled: bool = False
    chunk_target_chars: int = 520
    chunk_min_chars: int = 400
    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 100
    image_min_width: int = 180
    image_min_height: int = 180
    image_min_area: int = 32400
    image_max_per_document: int = 80
    caption_prompt_version: str = 'v1'
    model_timeout_seconds: int = 90
    model_max_retries: int = 3
    answer_max_tokens: int = 4000
    rewrite_max_tokens: int = 500

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> 'Settings':
        if env is None:
            if load_env_file and load_dotenv is not None:
                load_dotenv(dotenv_path=BASE_DIR / '.env', override=False)
            values: Mapping[str, str] = os.environ
        else:
            values = env

        data_dir = _path(values, 'RAG_DATA_DIR', DEFAULT_DATA_DIR)
        index_dir = _path(values, 'RAG_INDEX_DIR', data_dir / 'index')
        cache_dir = _path(values, 'RAG_CACHE_DIR', data_dir / 'cache')
        reports_dir = _path(values, 'RAG_REPORTS_DIR', data_dir / 'reports')
        target = _integer(values, 'CHUNK_TARGET_CHARS', 520, 100)
        minimum = _integer(values, 'CHUNK_MIN_CHARS', 400, 50)
        maximum = _integer(values, 'CHUNK_MAX_CHARS', 600, 100)
        overlap = _integer(values, 'CHUNK_OVERLAP_CHARS', 100, 0)
        if not minimum <= target <= maximum:
            raise ConfigurationError('切片参数必须满足 CHUNK_MIN_CHARS <= CHUNK_TARGET_CHARS <= CHUNK_MAX_CHARS')
        if overlap >= target:
            raise ConfigurationError('CHUNK_OVERLAP_CHARS 必须小于 CHUNK_TARGET_CHARS')

        return cls(
            deepseek_api_key=_text(values, 'DEEPSEEK_API_KEY', ''),
            dashscope_api_key=_text(values, 'DASHSCOPE_API_KEY', ''),
            deepseek_base_url=_text(values, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            deepseek_model=_text(values, 'DEEPSEEK_MODEL', 'deepseek-flash'),
            embedding_model=_text(values, 'EMBEDDING_MODEL', 'text-embedding-v4'),
            embedding_fallback_model=_text(values, 'EMBEDDING_FALLBACK_MODEL', 'text-embedding-v1'),
            knowledge_base_root=_path(values, 'DISNEY_KB_ROOT', DEFAULT_KB_ROOT),
            data_dir=data_dir,
            index_dir=index_dir,
            cache_dir=cache_dir,
            reports_dir=reports_dir,
            content_policy_path=_path(values, 'CONTENT_POLICY_PATH', DEFAULT_POLICY_PATH),
            retrieval_top_k=_integer(values, 'RETRIEVAL_TOP_K', 6),
            dense_top_k=_integer(values, 'DENSE_TOP_K', 20),
            bm25_top_k=_integer(values, 'BM25_TOP_K', 20),
            rrf_k=_integer(values, 'RRF_K', 60),
            dense_min_score=_number(values, 'DENSE_MIN_SCORE', 0.25),
            rerank_enabled=_boolean(values, 'RERANK_ENABLED', False),
            chunk_target_chars=target,
            chunk_min_chars=minimum,
            chunk_max_chars=maximum,
            chunk_overlap_chars=overlap,
            image_min_width=_integer(values, 'IMAGE_MIN_WIDTH', 180, 1),
            image_min_height=_integer(values, 'IMAGE_MIN_HEIGHT', 180, 1),
            image_min_area=_integer(values, 'IMAGE_MIN_AREA', 32400, 1),
            image_max_per_document=_integer(values, 'IMAGE_MAX_PER_DOCUMENT', 80, 1),
            caption_prompt_version=_text(values, 'CAPTION_PROMPT_VERSION', 'v1'),
            model_timeout_seconds=_integer(values, 'MODEL_TIMEOUT_SECONDS', 90),
            model_max_retries=_integer(values, 'MODEL_MAX_RETRIES', 3, 0),
        answer_max_tokens=_integer(values, 'ANSWER_MAX_TOKENS', 4000),
            rewrite_max_tokens=_integer(values, 'REWRITE_MAX_TOKENS', 500),
        )

    def missing_api_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.deepseek_api_key:
            missing.append('DEEPSEEK_API_KEY')
        if not self.dashscope_api_key:
            missing.append('DASHSCOPE_API_KEY')
        return missing

    def require_api_keys(self, *, deepseek: bool = True, dashscope: bool = True) -> None:
        missing: list[str] = []
        if deepseek and not self.deepseek_api_key:
            missing.append('DEEPSEEK_API_KEY')
        if dashscope and not self.dashscope_api_key:
            missing.append('DASHSCOPE_API_KEY')
        if missing:
            raise ConfigurationError('缺少必需环境变量: ' + ', '.join(missing))

    def ensure_directories(self) -> None:
        for directory in (self.index_dir, self.cache_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_knowledge_base(self) -> None:
        if not self.knowledge_base_root.is_dir():
            raise ConfigurationError('知识库目录不存在: ' + str(self.knowledge_base_root))
        if not self.content_policy_path.is_file():
            raise ConfigurationError('内容策略文件不存在: ' + str(self.content_policy_path))

    def public_summary(self) -> dict[str, object]:
        return {
            'deepseek_model': self.deepseek_model,
            'deepseek_api_key_present': bool(self.deepseek_api_key),
            'dashscope_api_key_present': bool(self.dashscope_api_key),
            'embedding_model': self.embedding_model,
            'embedding_fallback_model': self.embedding_fallback_model,
            'knowledge_base_root': str(self.knowledge_base_root),
            'index_dir': str(self.index_dir),
            'retrieval_top_k': self.retrieval_top_k,
            'dense_top_k': self.dense_top_k,
            'bm25_top_k': self.bm25_top_k,
            'rrf_k': self.rrf_k,
            'dense_min_score': self.dense_min_score,
            'rerank_enabled': self.rerank_enabled,
            'chunk_target_chars': self.chunk_target_chars,
            'chunk_overlap_chars': self.chunk_overlap_chars,
        }

    def __repr__(self) -> str:
        return 'Settings(deepseek_model=' + repr(self.deepseek_model) + ', knowledge_base_root=' + repr(str(self.knowledge_base_root)) + ')'


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    load_env_file: bool = True,
    require_api_keys: bool = True,
) -> Settings:
    settings = Settings.from_env(env, load_env_file=load_env_file)
    if require_api_keys:
        settings.require_api_keys()
    return settings


def main() -> int:
    try:
        settings = load_settings(require_api_keys=True)
        settings.validate_knowledge_base()
    except ConfigurationError as exc:
        print('配置检查失败: ' + str(exc))
        return 2
    print('配置检查通过')
    for key, value in settings.public_summary().items():
        print('  ' + key + ': ' + str(value))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
