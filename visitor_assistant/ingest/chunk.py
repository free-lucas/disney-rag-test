from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ingest.extract import ExtractedBlock, ExtractedDocument, SENTENCE_END, clean_text
from ingest.policy import ContentPolicy
from models import Audience, Chunk, ChunkType


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = 520
    min_chars: int = 400
    max_chars: int = 600
    overlap_chars: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError('切片参数必须满足 0 < min <= target <= max')
        if not 0 <= self.overlap_chars < self.target_chars:
            raise ValueError('overlap_chars 必须小于 target_chars')


@dataclass
class ChunkingResult:
    chunks: list[Chunk] = field(default_factory=list)
    redactions: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def split_units(text: str, max_chars: int) -> list[str]:
    units: list[str] = []
    for paragraph in clean_text(text).split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = re.split(r'(?<=[' + SENTENCE_END + r'])', paragraph)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_chars:
                units.append(part)
                continue
            start = 0
            while start < len(part):
                units.append(part[start:start + max_chars].strip())
                start += max_chars
    return [unit for unit in units if unit]


def overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ''
    tail = text[-overlap_chars:]
    for index, character in enumerate(tail):
        if character in SENTENCE_END and index < len(tail) - 10:
            return tail[index + 1:].strip()
    return tail.strip()


class _ChunkBuilder:
    def __init__(self, document: ExtractedDocument, category: str, audience: Audience, config: ChunkConfig, build_version: str) -> None:
        self.document = document
        self.category = category
        self.audience = audience
        self.config = config
        self.build_version = build_version
        self.result = ChunkingResult()
        self.heading_path: list[str] = [document.title.strip()] if document.title.strip() else []
        self.section_path: list[str] = list(self.heading_path)
        self.current = ''
        self.page: int | None = None
        self.slide: int | None = None
        self.prefix = hashlib.sha1(document.source_file.encode('utf-8')).hexdigest()[:10]

    def _emit(self, content: str, chunk_type: ChunkType, *, page: int | None = None, slide: int | None = None, image_path: str | None = None, caption: str | None = None) -> None:
        normalized = clean_text(content)
        if not normalized:
            return
        chunk = Chunk(
            id=self.prefix + '-' + str(len(self.result.chunks)).zfill(5),
            content=normalized,
            source_file=self.document.source_file,
            source_type=self.document.source_type,
            category=self.category,
            audience=self.audience,
            chunk_type=chunk_type,
            section_path=list(self.section_path),
            page=page,
            slide=slide,
            image_path=image_path,
            caption=caption,
            build_version=self.build_version,
        )
        self.result.chunks.append(chunk)

    def flush(self, *, overlap: bool = True) -> None:
        content = self.current.strip()
        if content:
            self._emit(content, ChunkType.TEXT, page=self.page, slide=self.slide)
            self.current = overlap_tail(content, self.config.overlap_chars) if overlap else ''
        else:
            self.current = ''

    def _start_with_overlap(self, unit: str) -> None:
        tail = overlap_tail(self.current, self.config.overlap_chars) if self.current else ''
        if tail and len(tail) + 1 + len(unit) <= self.config.max_chars:
            self.current = tail + '\n' + unit
        else:
            self.current = unit

    def _add_text(self, text: str, page: int | None, slide: int | None) -> None:
        if page is not None and self.page is not None and page != self.page and self.current.strip():
            self.flush(overlap=False)
        if slide is not None and self.slide is not None and slide != self.slide and self.current.strip():
            self.flush(overlap=False)
        if page is not None:
            self.page = page
        if slide is not None:
            self.slide = slide
        for unit in split_units(text, self.config.max_chars):
            if not self.current:
                self.current = unit
                continue
            candidate_length = len(self.current) + 1 + len(unit)
            if candidate_length > self.config.max_chars or (len(self.current) >= self.config.target_chars and candidate_length > self.config.target_chars):
                overflow = self.current
                self.flush(overlap=False)
                tail = overlap_tail(overflow, self.config.overlap_chars)
                if tail and len(tail) + 1 + len(unit) <= self.config.max_chars:
                    self.current = tail + '\n' + unit
                else:
                    self.current = unit
            else:
                self.current += '\n' + unit

    def add_heading(self, block: ExtractedBlock) -> None:
        page_changed = block.page is not None and self.page is not None and block.page != self.page
        slide_changed = block.slide is not None and self.slide is not None and block.slide != self.slide
        if page_changed or slide_changed:
            if self.current.strip():
                self.flush(overlap=False)
        elif self.current.strip() and len(self.current.strip()) >= self.config.min_chars:
            self.flush(overlap=False)

        level = max(1, block.level)
        self.heading_path = self.heading_path[:level - 1] + [block.text.strip()]
        if block.page is not None:
            self.page = block.page
        if block.slide is not None:
            self.slide = block.slide
        if not self.current.strip():
            self.section_path = list(self.heading_path)
            if block.include_in_content and block.text.strip():
                self.current = block.text.strip()
        elif block.include_in_content and block.text.strip():
            self.current += '\n' + block.text.strip()

    def add_table(self, block: ExtractedBlock) -> None:
        prefix = ''
        if self.current.strip():
            if len(self.current.strip()) < self.config.min_chars:
                prefix = self.current.strip()
            else:
                self.flush(overlap=False)
        content = (prefix + '\n' + block.text) if prefix else block.text
        self._emit(content, ChunkType.TABLE, page=block.page if block.page is not None else self.page, slide=block.slide if block.slide is not None else self.slide)
        self.current = ''

    def add_image(self, block: ExtractedBlock) -> None:
        caption = clean_text(block.caption or '')
        if not caption:
            self.result.warnings.append(self.document.source_file + ': 图片缺少描述，未生成图片 chunk')
            return
        if self.current.strip():
            self.flush(overlap=False)
        self._emit(
            caption,
            ChunkType.IMAGE_CAPTION,
            page=block.page if block.page is not None else self.page,
            slide=block.slide if block.slide is not None else self.slide,
            image_path=block.image_path,
            caption=caption,
        )
        self.current = ''

    def finish(self) -> ChunkingResult:
        if self.current.strip():
            self.flush(overlap=False)
        return self.result


def _redact_blocks(blocks: list[ExtractedBlock], policy: ContentPolicy, source_file: str) -> tuple[list[ExtractedBlock], list[dict[str, object]]]:
    redacted_blocks: list[ExtractedBlock] = []
    records: list[dict[str, object]] = []
    for block in blocks:
        text, redactions = policy.redact(block.text)
        for record in redactions:
            records.append({
                'source_file': source_file,
                'block_type': block.kind,
                'type': record.get('type', 'unknown'),
                'count': record.get('count', 0),
            })
        redacted_blocks.append(ExtractedBlock(
            kind=block.kind,
            text=text,
            level=block.level,
            page=block.page,
            slide=block.slide,
            image_path=block.image_path,
            caption=block.caption,
            include_in_content=block.include_in_content,
        ))
    return redacted_blocks, records


def chunk_document(
    document: ExtractedDocument,
    *,
    audience: Audience | str,
    config: ChunkConfig | None = None,
    policy: ContentPolicy | None = None,
    category: str | None = None,
    build_version: str = '',
) -> ChunkingResult:
    audience_value = audience if isinstance(audience, Audience) else Audience(str(audience))
    configuration = config or ChunkConfig()
    document_category = category
    if document_category is None:
        document_category = document.source_file.split('/', 1)[0] if '/' in document.source_file else ''
    blocks = document.blocks
    result_prefix = ChunkingResult(warnings=list(document.warnings))
    if audience_value is Audience.PUBLIC and policy is not None:
        blocks, redactions = _redact_blocks(blocks, policy, document.source_file)
        result_prefix.redactions.extend(redactions)

    builder = _ChunkBuilder(document, document_category, audience_value, configuration, build_version)
    for block in blocks:
        if block.kind == 'heading':
            builder.add_heading(block)
        elif block.kind == 'table':
            builder.add_table(block)
        elif block.kind == 'image':
            builder.add_image(block)
        else:
            builder._add_text(block.text, block.page, block.slide)
    result = builder.finish()
    result.redactions = result_prefix.redactions + result.redactions
    result.warnings = result_prefix.warnings + result.warnings
    return result
