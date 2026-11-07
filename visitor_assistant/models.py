from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Audience(str, Enum):
    PUBLIC = 'public'
    INTERNAL = 'internal'


class ChunkType(str, Enum):
    TEXT = 'text'
    TABLE = 'table'
    IMAGE_CAPTION = 'image_caption'


def _audience(value: Audience | str) -> Audience:
    try:
        return value if isinstance(value, Audience) else Audience(str(value))
    except ValueError as exc:
        raise ValueError('audience 必须为 public 或 internal') from exc


def _chunk_type(value: ChunkType | str) -> ChunkType:
    try:
        return value if isinstance(value, ChunkType) else ChunkType(str(value))
    except ValueError as exc:
        raise ValueError('chunk_type 必须为 text、table 或 image_caption') from exc


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _location(source_file: str, section_path: list[str], page: int | None, slide: int | None) -> str:
    parts: list[str] = []
    if section_path:
        parts.append(' > '.join(section_path))
    if page is not None:
        parts.append('第' + str(page) + '页')
    if slide is not None:
        parts.append('第' + str(slide) + '张幻灯片')
    return source_file + ('（' + '，'.join(parts) + '）' if parts else '')


@dataclass
class SourceRef:
    source_file: str
    source_type: str = ''
    category: str = ''
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    slide: int | None = None
    image_path: str | None = None

    @property
    def location(self) -> str:
        return _location(self.source_file, self.section_path, self.page, self.slide)

    def to_dict(self) -> dict[str, Any]:
        return {
            'source_file': self.source_file,
            'source_type': self.source_type,
            'category': self.category,
            'section_path': list(self.section_path),
            'page': self.page,
            'slide': self.slide,
            'image_path': self.image_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SourceRef':
        return cls(
            source_file=str(data['source_file']),
            source_type=str(data.get('source_type', '')),
            category=str(data.get('category', '')),
            section_path=[str(item) for item in data.get('section_path', [])],
            page=data.get('page'),
            slide=data.get('slide'),
            image_path=data.get('image_path'),
        )


@dataclass
class Citation:
    source_file: str
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    slide: int | None = None
    chunk_id: str | None = None

    @property
    def label(self) -> str:
        return _location(self.source_file, self.section_path, self.page, self.slide)

    @classmethod
    def from_source_ref(cls, ref: SourceRef, chunk_id: str | None = None) -> 'Citation':
        return cls(
            source_file=ref.source_file,
            section_path=list(ref.section_path),
            page=ref.page,
            slide=ref.slide,
            chunk_id=chunk_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'source_file': self.source_file,
            'section_path': list(self.section_path),
            'page': self.page,
            'slide': self.slide,
            'chunk_id': self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Citation':
        return cls(
            source_file=str(data['source_file']),
            section_path=[str(item) for item in data.get('section_path', [])],
            page=data.get('page'),
            slide=data.get('slide'),
            chunk_id=data.get('chunk_id'),
        )


@dataclass
class ImageRef:
    image_path: str
    caption: str = ''
    source_file: str = ''
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    slide: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'image_path': self.image_path,
            'caption': self.caption,
            'source_file': self.source_file,
            'section_path': list(self.section_path),
            'page': self.page,
            'slide': self.slide,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ImageRef':
        return cls(
            image_path=str(data['image_path']),
            caption=str(data.get('caption', '')),
            source_file=str(data.get('source_file', '')),
            section_path=[str(item) for item in data.get('section_path', [])],
            page=data.get('page'),
            slide=data.get('slide'),
        )


@dataclass
class Chunk:
    id: str
    content: str
    source_file: str
    category: str
    audience: Audience | str
    chunk_type: ChunkType | str = ChunkType.TEXT
    content_hash: str = ''
    source_type: str = ''
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    slide: int | None = None
    image_path: str | None = None
    caption: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    build_version: str = ''

    def __post_init__(self) -> None:
        self.audience = _audience(self.audience)
        self.chunk_type = _chunk_type(self.chunk_type)
        self.section_path = [str(item) for item in self.section_path]
        if not self.content_hash:
            self.content_hash = _text_hash(self.content)

    @property
    def location(self) -> str:
        return _location(self.source_file, self.section_path, self.page, self.slide)

    def source_ref(self) -> SourceRef:
        return SourceRef(
            source_file=self.source_file,
            source_type=self.source_type,
            category=self.category,
            section_path=list(self.section_path),
            page=self.page,
            slide=self.slide,
            image_path=self.image_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'content_hash': self.content_hash,
            'source_file': self.source_file,
            'source_type': self.source_type,
            'category': self.category,
            'audience': self.audience.value,
            'chunk_type': self.chunk_type.value,
            'section_path': list(self.section_path),
            'page': self.page,
            'slide': self.slide,
            'image_path': self.image_path,
            'caption': self.caption,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'build_version': self.build_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Chunk':
        return cls(
            id=str(data['id']),
            content=str(data['content']),
            source_file=str(data['source_file']),
            category=str(data.get('category', '')),
            audience=data.get('audience', Audience.PUBLIC),
            chunk_type=data.get('chunk_type', ChunkType.TEXT),
            content_hash=str(data.get('content_hash', '')),
            source_type=str(data.get('source_type', '')),
            section_path=[str(item) for item in data.get('section_path', [])],
            page=data.get('page'),
            slide=data.get('slide'),
            image_path=data.get('image_path'),
            caption=data.get('caption'),
            embedding_model=data.get('embedding_model'),
            embedding_dim=data.get('embedding_dim'),
            build_version=str(data.get('build_version', '')),
        )

    @classmethod
    def from_json(cls, text: str) -> 'Chunk':
        return cls.from_dict(json.loads(text))


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    matched_query: str = ''

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def source_ref(self) -> SourceRef:
        return self.chunk.source_ref()

    def to_dict(self) -> dict[str, Any]:
        return {
            'chunk': self.chunk.to_dict(),
            'score': self.score,
            'dense_score': self.dense_score,
            'bm25_score': self.bm25_score,
            'rerank_score': self.rerank_score,
            'matched_query': self.matched_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RetrievalResult':
        return cls(
            chunk=Chunk.from_dict(data['chunk']),
            score=float(data.get('score', 0.0)),
            dense_score=data.get('dense_score'),
            bm25_score=data.get('bm25_score'),
            rerank_score=data.get('rerank_score'),
            matched_query=str(data.get('matched_query', '')),
        )


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    refused: bool = False
    disclaimer: bool = False
    query: str = ''
    rewritten_query: str = ''
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'text': self.text,
            'citations': [item.to_dict() for item in self.citations],
            'images': [item.to_dict() for item in self.images],
            'refused': self.refused,
            'disclaimer': self.disclaimer,
            'query': self.query,
            'rewritten_query': self.rewritten_query,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Answer':
        return cls(
            text=str(data.get('text', '')),
            citations=[Citation.from_dict(item) for item in data.get('citations', [])],
            images=[ImageRef.from_dict(item) for item in data.get('images', [])],
            refused=bool(data.get('refused', False)),
            disclaimer=bool(data.get('disclaimer', False)),
            query=str(data.get('query', '')),
            rewritten_query=str(data.get('rewritten_query', '')),
            error=data.get('error'),
        )
