from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingest.extract import ExtractedDocument, ImageAsset


CAPTION_SYSTEM_PROMPT = '你是迪士尼游客知识库的图片描述助手。只描述图片中可见的信息，不猜测看不清的内容。'
CAPTION_USER_PROMPT = (
    '请用简体中文写 3-6 句图片描述，作为游客问答知识库的图片证据。'
    '优先提取标题、价格、日期、地点、时间、表格表头和地图标注等可见关键信息；'
    '如果是价格表或舱房表，请逐项写出可见的价格或区间；不要编造。'
)


@dataclass
class CaptionResult:
    image_path: str
    caption: str = ''
    cached: bool = False
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.caption) and self.error_type is None

    def to_dict(self) -> dict[str, object]:
        return {
            'image_path': self.image_path,
            'caption': self.caption,
            'cached': self.cached,
            'error_type': self.error_type,
            'error_message': self.error_message,
        }


@dataclass
class CaptionBatchResult:
    results: list[CaptionResult] = field(default_factory=list)
    captions: dict[str, str] = field(default_factory=dict)
    cached_count: int = 0
    api_calls: int = 0

    @property
    def errors(self) -> list[CaptionResult]:
        return [result for result in self.results if not result.ok]


class CaptionCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(image_hash: str, model: str, prompt_version: str) -> str:
        raw = image_hash + '|' + model + '|' + prompt_version
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _path(self, image_hash: str, model: str, prompt_version: str) -> Path:
        return self.directory / (self.key(image_hash, model, prompt_version) + '.json')

    def get(self, image_hash: str, model: str, prompt_version: str) -> str | None:
        path = self._path(image_hash, model, prompt_version)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        caption = str(payload.get('caption', '')).strip()
        return caption or None

    def set(self, image_hash: str, model: str, prompt_version: str, caption: str) -> None:
        path = self._path(image_hash, model, prompt_version)
        payload = {
            'image_hash': image_hash,
            'model': model,
            'prompt_version': prompt_version,
            'caption': caption,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _media_type(path: Path) -> str:
    extension = path.suffix.lower().lstrip('.')
    return {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp',
    }.get(extension, 'image/jpeg')


class ImageCaptioner:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        prompt_version: str,
        cache: CaptionCache,
        client: object | None = None,
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_version = prompt_version
        self.cache = cache
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)

    @classmethod
    def from_settings(cls, settings: object, cache: CaptionCache | None = None, client: object | None = None) -> 'ImageCaptioner':
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            prompt_version=settings.caption_prompt_version,
            cache=cache or CaptionCache(settings.cache_dir / 'captions'),
            client=client,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )

    def _get_client(self) -> object:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError('缺少 DEEPSEEK_API_KEY')
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def caption(self, image: ImageAsset) -> CaptionResult:
        path = Path(image.path)
        if not path.is_file():
            return CaptionResult(image.path, error_type='missing_image', error_message='图片文件不存在')
        cached = self.cache.get(image.image_hash, self.model, self.prompt_version)
        if cached:
            return CaptionResult(image.path, caption=cached, cached=True)
        try:
            client = self._get_client()
            data = base64.b64encode(path.read_bytes()).decode('ascii')
            url = 'data:' + _media_type(path) + ';base64,' + data
            content = [
                {'type': 'text', 'text': CAPTION_USER_PROMPT + ('\n邻近文本：' + image.context if image.context else '')},
                {'type': 'image_url', 'image_url': {'url': url}},
            ]
            response = None
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {'role': 'system', 'content': CAPTION_SYSTEM_PROMPT},
                            {'role': 'user', 'content': content},
                        ],
                        timeout=self.timeout_seconds,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(min(8.0, 0.5 * (2 ** attempt)))
            if response is None:
                raise last_error or RuntimeError('图片描述请求失败')
            caption = str(response.choices[0].message.content or '').strip()
            if not caption:
                return CaptionResult(image.path, error_type='empty_caption', error_message='视觉模型返回空描述')
            self.cache.set(image.image_hash, self.model, self.prompt_version, caption)
            return CaptionResult(image.path, caption=caption)
        except Exception as exc:
            return CaptionResult(image.path, error_type='api_error', error_message=str(exc)[:500])

    def caption_many(self, images: list[ImageAsset]) -> CaptionBatchResult:
        result = CaptionBatchResult()
        for image in images:
            caption_result = self.caption(image)
            result.results.append(caption_result)
            if caption_result.cached:
                result.cached_count += 1
            else:
                result.api_calls += 1
            if caption_result.ok:
                result.captions[image.path] = caption_result.caption
        return result


def apply_captions_to_document(document: ExtractedDocument, captions: dict[str, str]) -> None:
    for image in document.images:
        if image.path in captions:
            image.caption = captions[image.path]
    for block in document.blocks:
        if block.kind == 'image' and block.image_path in captions:
            block.caption = captions[block.image_path]
