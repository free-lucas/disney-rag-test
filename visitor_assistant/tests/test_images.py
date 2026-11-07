from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest.caption import CaptionCache, ImageCaptioner, apply_captions_to_document
from ingest.chunk import chunk_document
from ingest.extract import ExtractedBlock, ExtractedDocument, ImageAsset, extract_document
from ingest.images import ImageFilterConfig, apply_captions, filter_images


KB_ROOT = Path(__file__).resolve().parents[2] / '迪士尼RAG知识库（完整）'


def _png(path: Path, size: tuple[int, int] = (300, 300), color: str = 'red') -> Path:
    from PIL import Image

    Image.new('RGB', size, color).save(path)
    return path


def test_filters_small_duplicate_and_over_limit_images() -> None:
    images = [
        ImageAsset('a.png', 'hash-a', 'doc.docx', width=300, height=300),
        ImageAsset('b.png', 'hash-a', 'doc.docx', width=300, height=300),
        ImageAsset('c.png', 'hash-c', 'doc.docx', width=100, height=300),
        ImageAsset('d.png', 'hash-d', 'doc.docx', width=300, height=100),
        ImageAsset('e.png', 'hash-e', 'doc.docx', width=150, height=150),
    ]

    result = filter_images(images, config=ImageFilterConfig(max_per_document=2))

    assert [image.image_hash for image in result.kept] == ['hash-a']
    reasons = {(item.image.image_hash, item.reason) for item in result.filtered}
    assert ('hash-a', 'duplicate') in reasons
    assert ('hash-c', 'too_narrow') in reasons
    assert ('hash-d', 'too_short') in reasons
    assert ('hash-e', 'too_narrow') in reasons


def test_limit_per_document() -> None:
    images = [ImageAsset(str(index) + '.png', 'hash-' + str(index), 'doc.docx', width=300, height=300) for index in range(4)]

    result = filter_images(images, config=ImageFilterConfig(max_per_document=2))

    assert len(result.kept) == 2
    assert {item.reason for item in result.filtered} == {'over_document_limit'}


class _FakeCompletions:
    def __init__(self, content: str = '图片中包含邮轮价格表，成人价 735 美元。', error: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.content = content
        self.error = error

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _captioner(tmp_path: Path, completions: _FakeCompletions, *, max_retries: int = 0) -> ImageCaptioner:
    return ImageCaptioner(
        api_key='test-key',
        base_url='https://example.invalid',
        model='deepseek-flash',
        prompt_version='v1',
        cache=CaptionCache(tmp_path / 'cache'),
        client=_FakeClient(completions),
        timeout_seconds=5,
        max_retries=max_retries,
    )


def test_caption_uses_data_url_and_cache(tmp_path: Path) -> None:
    image_path = _png(tmp_path / 'price.png')
    image = ImageAsset(str(image_path), 'image-hash', 'cruise.jpeg', width=300, height=300, context='邮轮价格')
    completions = _FakeCompletions()
    captioner = _captioner(tmp_path, completions)

    first = captioner.caption(image)
    second = captioner.caption(image)

    assert first.ok and not first.cached
    assert second.ok and second.cached
    assert len(completions.calls) == 1
    user_content = completions.calls[0]['messages'][1]['content']
    assert user_content[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert '邮轮价格' in user_content[0]['text']


def test_caption_empty_and_api_error_are_recorded(tmp_path: Path) -> None:
    image_path = _png(tmp_path / 'price.png')
    image = ImageAsset(str(image_path), 'image-hash', 'cruise.jpeg', width=300, height=300)

    empty = _captioner(tmp_path, _FakeCompletions(content='')).caption(image)
    failed = _captioner(tmp_path, _FakeCompletions(error=RuntimeError('rate limited'))).caption(image)

    assert empty.error_type == 'empty_caption'
    assert failed.error_type == 'api_error'
    assert 'rate limited' in failed.error_message


def test_cache_key_includes_model_and_prompt_version(tmp_path: Path) -> None:
    first = CaptionCache.key('hash', 'deepseek-flash', 'v1')
    second = CaptionCache.key('hash', 'deepseek-flash', 'v2')
    third = CaptionCache.key('hash', 'other-model', 'v1')

    assert len({first, second, third}) == 3


def test_docx_embedded_image_is_extracted_with_context(tmp_path: Path) -> None:
    from docx import Document

    picture = _png(tmp_path / 'picture.png')
    path = tmp_path / 'document.docx'
    document = Document()
    document.add_paragraph('邮轮价格参考图')
    document.add_picture(str(picture))
    document.save(str(path))

    extracted = extract_document(path, '1-产品与服务详情/document.docx', assets_dir=tmp_path / 'assets')

    assert len(extracted.images) == 1
    assert extracted.images[0].width == 300
    assert '邮轮价格参考图' in extracted.images[0].context
    assert any(block.kind == 'image' and block.image_path == extracted.images[0].path for block in extracted.blocks)


def test_pptx_embedded_image_is_traceable_to_slide(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    picture = _png(tmp_path / 'picture.png')
    path = tmp_path / 'slides.pptx'
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = '邮轮价格'
    slide.shapes.add_picture(str(picture), Inches(1), Inches(1), Inches(2), Inches(2))
    presentation.save(str(path))

    extracted = extract_document(path, '1-产品与服务详情/slides.pptx', assets_dir=tmp_path / 'assets')

    assert len(extracted.images) == 1
    assert extracted.images[0].slide == 1
    assert '邮轮价格' in extracted.images[0].context


def test_captioned_image_becomes_chunk_and_empty_caption_is_skipped(tmp_path: Path) -> None:
    image_path = _png(tmp_path / 'price.png')
    image = ImageAsset(str(image_path), 'hash-price', 'cruise.jpeg', width=300, height=300)
    document = ExtractedDocument(
        '1-产品与服务详情/cruise.jpeg',
        'jpeg',
        '邮轮价格',
        [ExtractedBlock('heading', '邮轮价格', level=1), ExtractedBlock('image', image_path=str(image_path))],
        [image],
    )

    skipped = chunk_document(document, audience='public')
    assert not [chunk for chunk in skipped.chunks if chunk.chunk_type.value == 'image_caption']
    assert skipped.warnings

    apply_captions_to_document(document, {str(image_path): '邮轮价格表显示内舱 735 美元起。'})
    result = chunk_document(document, audience='public')
    image_chunks = [chunk for chunk in result.chunks if chunk.chunk_type.value == 'image_caption']

    assert len(image_chunks) == 1
    assert image_chunks[0].image_path == str(image_path)
    assert image_chunks[0].caption == '邮轮价格表显示内舱 735 美元起。'
    assert image_chunks[0].source_file == '1-产品与服务详情/cruise.jpeg'
    assert image_chunks[0].section_path == ['邮轮价格']


@pytest.mark.skipif(not KB_ROOT.is_dir(), reason='完整知识库不存在')
def test_all_nine_standalone_cruise_images_are_kept(tmp_path: Path) -> None:
    paths = sorted((KB_ROOT / '1-产品与服务详情').glob('迪士尼邮轮价格*.jpeg'))
    assert len(paths) == 9

    kept = 0
    filtered = 0
    seen: set[str] = set()
    for path in paths:
        document = extract_document(path, path.relative_to(KB_ROOT).as_posix(), assets_dir=tmp_path)
        result = filter_images(document.images, seen_hashes=seen)
        kept += len(result.kept)
        filtered += len(result.filtered)

    assert kept == 9
    assert filtered == 0
