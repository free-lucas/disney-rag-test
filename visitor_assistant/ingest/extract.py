from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable

from ingest.discover import sha256_file
from ingest.normalize import NormalizationError, normalized_source


HEADING_STYLE = re.compile(r'^(?:Heading|标题)\s*(\d+)$', re.IGNORECASE)
HEADING_CHAPTER = re.compile(r'^第[一二三四五六七八九十百零〇0-9]+[章节篇部分]')
HEADING_CHINESE = re.compile(r'^[一二三四五六七八九十百]+、')
HEADING_PAREN = re.compile(r'^[（(][一二三四五六七八九十0-9]+[）)]')
HEADING_NUMBER = re.compile(r'^(\d+(?:\.\d+){0,3})(?:[、．]|\s+)')
HEADING_DOTTED = re.compile(r'^(\d+)\.\s+')
SENTENCE_END = '。！？；!?;'


class ExtractionError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ImageAsset:
    path: str
    image_hash: str
    source_file: str
    width: int | None = None
    height: int | None = None
    page: int | None = None
    slide: int | None = None
    context: str = ''
    caption: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'image_hash': self.image_hash,
            'source_file': self.source_file,
            'width': self.width,
            'height': self.height,
            'page': self.page,
            'slide': self.slide,
            'context': self.context,
            'caption': self.caption,
        }


@dataclass
class ExtractedBlock:
    kind: str
    text: str = ''
    level: int = 0
    page: int | None = None
    slide: int | None = None
    image_path: str | None = None
    caption: str | None = None
    include_in_content: bool = True


@dataclass
class ExtractedDocument:
    source_file: str
    source_type: str
    title: str
    blocks: list[ExtractedBlock] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text_length(self) -> int:
        return sum(len(block.text) for block in self.blocks if block.kind in {'text', 'table', 'heading'})

    @property
    def has_text(self) -> bool:
        return any(block.kind in {'text', 'table'} and block.text.strip() for block in self.blocks)

    def to_dict(self) -> dict[str, object]:
        return {
            'source_file': self.source_file,
            'source_type': self.source_type,
            'title': self.title,
            'text_length': self.text_length,
            'has_text': self.has_text,
            'blocks': [
                {
                    'kind': block.kind,
                    'text': block.text,
                    'level': block.level,
                    'page': block.page,
                    'slide': block.slide,
                    'image_path': block.image_path,
                    'caption': block.caption,
                }
                for block in self.blocks
            ],
            'images': [image.to_dict() for image in self.images],
            'warnings': list(self.warnings),
        }


def _asset_directory(assets_dir: Path | None, prefix: str, source_file: str) -> Path | None:
    if assets_dir is None:
        return None
    directory = Path(assets_dir) / (prefix + '_' + hashlib.sha1(source_file.encode('utf-8')).hexdigest()[:10])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _image_dimensions(blob: bytes) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as image:
            return image.size
    except Exception:
        return None, None


def _image_extension(content_type: str, part_name: str = '') -> str:
    if part_name and '.' in part_name:
        return part_name.rsplit('.', 1)[-1].lower()
    mapping = {
        'image/jpeg': 'jpg',
        'image/png': 'png',
        'image/gif': 'gif',
        'image/webp': 'webp',
        'image/bmp': 'bmp',
        'image/tiff': 'tiff',
    }
    return mapping.get(content_type.lower(), 'png')


def _save_image_blob(blob: bytes, directory: Path | None, content_type: str, part_name: str = '') -> str | None:
    if directory is None:
        return None
    digest = hashlib.sha256(blob).hexdigest()
    extension = _image_extension(content_type, part_name)
    destination = directory / ('image_' + digest[:12] + '.' + extension)
    if not destination.is_file():
        destination.write_bytes(blob)
    return str(destination.resolve())


def clean_text(text: str) -> str:
    normalized = str(text).replace('\r\n', '\n').replace('\r', '\n').replace('\u3000', ' ')
    lines: list[str] = []
    for raw_line in normalized.split('\n'):
        line = re.sub(r'[ \t]+', ' ', raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != '':
            lines.append('')
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines).strip()


def heading_level_from_text(text: str) -> int:
    candidate = text.strip()
    if not candidate or len(candidate) > 50:
        return 0
    if candidate[0] in '•·-*':
        return 0
    if candidate.endswith(('。', '！', '；', '，', ',', ';')):
        return 0
    if sum(candidate.count(mark) for mark in '。！？；') > 2:
        return 0
    if HEADING_CHAPTER.match(candidate):
        return 1
    if HEADING_CHINESE.match(candidate):
        return 1
    if HEADING_PAREN.match(candidate):
        return 2
    match = HEADING_NUMBER.match(candidate)
    if match:
        return min(match.group(1).count('.') + 1, 4)
    if HEADING_DOTTED.match(candidate):
        return 1
    return 0


def _docx_heading_level(paragraph: object, text: str) -> int:
    style = getattr(paragraph, 'style', None)
    style_name = str(getattr(style, 'name', '') or '')
    match = HEADING_STYLE.match(style_name.strip())
    if match:
        return max(1, int(match.group(1)))
    element = getattr(paragraph, '_p', None)
    if element is not None:
        try:
            from docx.oxml.ns import qn

            properties = element.pPr
            if properties is not None:
                outline = properties.find(qn('w:outlineLvl'))
                if outline is not None:
                    value = outline.get(qn('w:val'))
                    if value is not None:
                        return max(1, int(value) + 1)
        except Exception:
            pass
    return heading_level_from_text(text)


def rows_to_markdown(rows: Iterable[Iterable[str]]) -> str:
    cleaned_rows: list[list[str]] = []
    for row in rows:
        values = [clean_text(str(cell)).replace('\n', ' / ').replace('|', '\\|') for cell in row]
        if any(values):
            cleaned_rows.append(values)
    if not cleaned_rows:
        return ''
    width = max(len(row) for row in cleaned_rows)
    for row in cleaned_rows:
        row.extend([''] * (width - len(row)))
    header = cleaned_rows[0]
    if not any(header):
        header = ['列' + str(index + 1) for index in range(width)]
        cleaned_rows[0] = header
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * width]
    for row in cleaned_rows[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _append_paragraph_blocks(
    blocks: list[ExtractedBlock],
    text: str,
    *,
    page: int | None = None,
    slide: int | None = None,
    level: int | None = None,
) -> None:
    lines = [line.strip() for line in text.replace('\r', '\n').split('\n')]
    pending_text: list[str] = []
    table_lines: list[str] = []

    def flush_text() -> None:
        nonlocal pending_text
        if pending_text:
            combined = clean_text('\n'.join(pending_text))
            if combined:
                blocks.append(ExtractedBlock('text', combined, page=page, slide=slide))
            pending_text = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            combined = clean_text('\n'.join(table_lines))
            if combined:
                blocks.append(ExtractedBlock('table', combined, page=page, slide=slide))
            table_lines = []

    for line in lines:
        if not line:
            flush_table()
            flush_text()
            continue
        if line.startswith('|') and line.count('|') >= 2:
            flush_text()
            table_lines.append(line)
            continue
        flush_table()
        detected_level = level if level is not None else heading_level_from_text(line)
        if detected_level:
            flush_text()
            blocks.append(ExtractedBlock('heading', line, level=detected_level, page=page, slide=slide))
        else:
            pending_text.append(line)
    flush_table()
    flush_text()


def _docx_paragraph_images(element: object, document: object, directory: Path | None, source_file: str, context: str) -> list[ImageAsset]:
    from docx.oxml.ns import qn

    images: list[ImageAsset] = []
    for drawing in element.iter(qn('w:drawing')):
        for blip in drawing.iter(qn('a:blip')):
            relation_id = blip.get(qn('r:embed'))
            if not relation_id:
                continue
            part = document.part.related_parts.get(relation_id)
            blob = getattr(part, 'blob', None) if part is not None else None
            if not blob:
                continue
            path = _save_image_blob(
                blob,
                directory,
                str(getattr(part, 'content_type', '') or ''),
                str(getattr(part, 'partname', '') or ''),
            )
            if not path:
                continue
            width, height = _image_dimensions(blob)
            images.append(ImageAsset(
                path=path,
                image_hash=hashlib.sha256(blob).hexdigest(),
                source_file=source_file,
                width=width,
                height=height,
                context=context,
            ))
    return images


def _extract_docx(path: Path, source_file: str, source_type: str, assets_dir: Path | None) -> ExtractedDocument:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[ExtractedBlock] = []
    images: list[ImageAsset] = []
    image_dir = _asset_directory(assets_dir, 'docx', source_file)
    last_context = ''
    for child in document.element.body.iterchildren():
        if child.tag == qn('w:p'):
            paragraph = Paragraph(child, document)
            text = clean_text(paragraph.text)
            if text:
                last_context = text
                level = _docx_heading_level(paragraph, text)
                if level:
                    blocks.append(ExtractedBlock('heading', text, level=level))
                else:
                    _append_paragraph_blocks(blocks, text)
            for image in _docx_paragraph_images(child, document, image_dir, source_file, text or last_context):
                images.append(image)
                blocks.append(ExtractedBlock('image', image_path=image.path))
        elif child.tag == qn('w:tbl'):
            table = Table(child, document)
            markdown = rows_to_markdown([[cell.text for cell in row.cells] for row in table.rows])
            if markdown:
                blocks.append(ExtractedBlock('table', markdown))

    title = ''
    for block in blocks:
        if block.kind == 'heading':
            title = block.text
            break
    if not title:
        title = path.stem
    return ExtractedDocument(source_file, source_type, title, blocks, images)


def _iter_pptx_shapes(shapes: object) -> Iterable[object]:
    for shape in shapes:
        yield shape
        nested = getattr(shape, 'shapes', None)
        if nested is not None:
            yield from _iter_pptx_shapes(nested)


def _extract_pptx(path: Path, source_file: str, source_type: str, assets_dir: Path | None) -> ExtractedDocument:
    from pptx import Presentation

    presentation = Presentation(str(path))
    blocks: list[ExtractedBlock] = []
    images: list[ImageAsset] = []
    image_dir = _asset_directory(assets_dir, 'pptx', source_file)
    title = path.stem
    for slide_number, slide in enumerate(presentation.slides, start=1):
        slide_title = ''
        try:
            if slide.shapes.title is not None:
                slide_title = clean_text(slide.shapes.title.text)
        except Exception:
            slide_title = ''
        if slide_number == 1 and slide_title:
            title = slide_title
        if slide_title:
            blocks.append(ExtractedBlock('heading', slide_title, level=1, slide=slide_number))
        else:
            blocks.append(ExtractedBlock('heading', '第' + str(slide_number) + '张幻灯片', level=1, slide=slide_number, include_in_content=False))
        for shape in _iter_pptx_shapes(slide.shapes):
            try:
                is_title = slide.shapes.title is not None and shape.shape_id == slide.shapes.title.shape_id
            except Exception:
                is_title = False
            if is_title:
                continue
            if getattr(shape, 'has_table', False):
                markdown = rows_to_markdown([[cell.text for cell in row.cells] for row in shape.table.rows])
                if markdown:
                    blocks.append(ExtractedBlock('table', markdown, slide=slide_number))
                continue
            if getattr(shape, 'has_text_frame', False):
                for paragraph in shape.text_frame.paragraphs:
                    text = clean_text(paragraph.text)
                    if text:
                        _append_paragraph_blocks(blocks, text, slide=slide_number)
            try:
                picture = shape.image
            except Exception:
                picture = None
            if picture is not None:
                blob = picture.blob
                extension = str(getattr(picture, 'ext', 'png') or 'png')
                context = ('幻灯片标题：' + slide_title + '；' if slide_title else '') + '图片名称：' + str(getattr(shape, 'name', ''))
                saved = _save_image_blob(blob, image_dir, 'image/' + extension, 'image.' + extension)
                if saved:
                    width, height = _image_dimensions(blob)
                    image = ImageAsset(
                        path=saved,
                        image_hash=hashlib.sha256(blob).hexdigest(),
                        source_file=source_file,
                        width=width,
                        height=height,
                        slide=slide_number,
                        context=context,
                    )
                    images.append(image)
                    blocks.append(ExtractedBlock('image', slide=slide_number, image_path=saved))
    return ExtractedDocument(source_file, source_type, title, blocks, images)


def _extract_pdf(path: Path, source_file: str, source_type: str, assets_dir: Path | None) -> ExtractedDocument:
    import fitz

    document = fitz.open(str(path))
    blocks: list[ExtractedBlock] = []
    images: list[ImageAsset] = []
    warnings: list[str] = []
    title = clean_text(str(document.metadata.get('title') or '')) or path.stem
    page_dir = None
    if assets_dir is not None:
        page_dir = Path(assets_dir) / ('pdf_' + hashlib.sha1(source_file.encode('utf-8')).hexdigest()[:10])
        page_dir.mkdir(parents=True, exist_ok=True)

    for page_index, page in enumerate(document, start=1):
        page_text = clean_text(page.get_text('text'))
        if len(page_text) < 20:
            if page_dir is None:
                warnings.append('第' + str(page_index) + '页没有文本层，但未配置图片输出目录')
            else:
                image_path = page_dir / ('page_' + str(page_index).zfill(3) + '.png')
                pixmap = page.get_pixmap(dpi=150)
                pixmap.save(str(image_path))
                images.append(ImageAsset(
                    path=str(image_path.resolve()),
                    image_hash=sha256_file(image_path),
                    source_file=source_file,
                    width=pixmap.width,
                    height=pixmap.height,
                    page=page_index,
                    context=title,
                ))
                blocks.append(ExtractedBlock('image', page=page_index, image_path=str(image_path.resolve())))
            continue
        for line in page_text.split('\n'):
            text = clean_text(line)
            if not text:
                continue
            level = heading_level_from_text(text)
            if level:
                blocks.append(ExtractedBlock('heading', text, level=level, page=page_index))
            else:
                blocks.append(ExtractedBlock('text', text, page=page_index))
    document.close()
    return ExtractedDocument(source_file, source_type, title, blocks, images, warnings)


def _extract_image(path: Path, source_file: str, source_type: str) -> ExtractedDocument:
    width = None
    height = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        pass
    image = ImageAsset(
        path=str(path.resolve()),
        image_hash=sha256_file(path),
        source_file=source_file,
        width=width,
        height=height,
    )
    return ExtractedDocument(
        source_file,
        source_type,
        path.stem,
        [ExtractedBlock('image', image_path=str(path.resolve()))],
        [image],
    )


def extract_document(
    path: str | Path,
    source_file: str,
    *,
    assets_dir: str | Path | None = None,
    allow_com: bool = True,
) -> ExtractedDocument:
    try:
        with normalized_source(path, allow_com=allow_com) as source:
            assets = Path(assets_dir) if assets_dir is not None else None
            if source.source_format == 'docx':
                return _extract_docx(source.path, source_file, source.source_type, assets)
            if source.source_format == 'pptx':
                return _extract_pptx(source.path, source_file, source.source_type, assets)
            if source.source_format == 'pdf':
                return _extract_pdf(source.path, source_file, source.source_type, assets)
            if source.source_format in {'jpeg', 'image'}:
                return _extract_image(source.path, source_file, source.source_type)
            raise ExtractionError('unsupported_format', '不支持的文件格式: ' + source.source_format)
    except NormalizationError as exc:
        raise ExtractionError(exc.error_type, str(exc)) from exc
