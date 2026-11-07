import shutil
from pathlib import Path

import pytest

from ingest.chunk import ChunkConfig, chunk_document, overlap_tail
from ingest.extract import ExtractionError, ExtractedBlock, ExtractedDocument, extract_document
from ingest.policy import ContentPolicy


KB_ROOT = Path(__file__).resolve().parents[2] / '迪士尼RAG知识库（完整）'


def _make_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_heading('第一章 门票', level=1)
    document.add_paragraph('游客可在官方渠道购买一日票和两日票。')
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = '票种'
    table.cell(0, 1).text = '价格'
    table.cell(1, 0).text = '一日票'
    table.cell(1, 1).text = '请以官方渠道为准'
    document.add_heading('第二章 酒店', level=1)
    document.add_paragraph('度假区酒店提供客房和餐饮服务。')
    document.save(str(path))


def _make_pptx(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = '邮轮价格'
    textbox = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(1))
    textbox.text_frame.text = '迪士尼邮轮提供多种舱房选择。'
    table_shape = slide.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(5), Inches(1))
    table = table_shape.table
    table.cell(0, 0).text = '舱房'
    table.cell(0, 1).text = '价格'
    table.cell(1, 0).text = '内舱'
    table.cell(1, 1).text = '以官方报价为准'
    second = presentation.slides.add_slide(presentation.slide_layouts[5])
    second.shapes.title.text = '餐饮'
    second.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(1)).text_frame.text = '园区内有多种餐厅。'
    presentation.save(str(path))


def test_docx_extracts_heading_text_and_table(tmp_path: Path) -> None:
    path = tmp_path / 'sample.docx'
    _make_docx(path)

    document = extract_document(path, '1-产品与服务详情/sample.docx')
    result = chunk_document(document, audience='public')

    assert document.has_text
    assert any(block.kind == 'heading' and block.level == 1 for block in document.blocks)
    assert any(block.kind == 'table' and '| 票种 | 价格 |' in block.text for block in document.blocks)
    assert any('第一章 门票' in chunk.section_path for chunk in result.chunks)
    assert any(chunk.chunk_type.value == 'table' and '| 票种 | 价格 |' in chunk.content for chunk in result.chunks)


def test_ooxml_with_doc_extension_uses_same_parser(tmp_path: Path) -> None:
    source = tmp_path / 'sample.docx'
    disguised = tmp_path / 'sample.doc'
    _make_docx(source)
    shutil.copyfile(source, disguised)

    document = extract_document(disguised, '1-产品与服务详情/sample.doc')

    assert document.source_type == 'docx'
    assert document.text_length > 0
    assert any(block.kind == 'table' for block in document.blocks)


def test_pptx_extracts_slides_and_tables(tmp_path: Path) -> None:
    path = tmp_path / 'sample.pptx'
    _make_pptx(path)

    document = extract_document(path, '1-产品与服务详情/sample.pptx')

    assert {block.slide for block in document.blocks} == {1, 2}
    assert any(block.kind == 'heading' and block.text == '邮轮价格' for block in document.blocks)
    assert any(block.kind == 'table' and '舱房' in block.text for block in document.blocks)


def test_pptx_with_ppt_extension_uses_same_parser(tmp_path: Path) -> None:
    source = tmp_path / 'sample.pptx'
    disguised = tmp_path / 'sample.ppt'
    _make_pptx(source)
    shutil.copyfile(source, disguised)

    document = extract_document(disguised, '1-产品与服务详情/sample.ppt')

    assert document.source_type == 'pptx'
    assert any(block.slide == 1 for block in document.blocks)


def test_pdf_keeps_page_numbers_and_renders_scanned_pages(tmp_path: Path) -> None:
    import fitz

    path = tmp_path / 'sample.pdf'
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), 'Disneyland ticket information for visitors.')
    pdf.new_page()
    pdf.save(str(path))
    pdf.close()

    document = extract_document(path, '1-产品与服务详情/sample.pdf', assets_dir=tmp_path / 'assets')

    assert any(block.page == 1 and block.kind == 'text' for block in document.blocks)
    assert any(block.page == 2 and block.kind == 'image' for block in document.blocks)
    assert len(document.images) == 1
    assert Path(document.images[0].path).is_file()


def test_ole_records_unsupported_when_com_disabled(tmp_path: Path) -> None:
    path = tmp_path / 'legacy.doc'
    path.write_bytes(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 64)

    with pytest.raises(ExtractionError) as error:
        extract_document(path, 'legacy.doc', allow_com=False)

    assert error.value.error_type == 'com_disabled'


@pytest.mark.skipif(not KB_ROOT.is_dir(), reason='完整知识库不存在')
def test_real_ole_word_conversion(tmp_path: Path) -> None:
    path = KB_ROOT / '1-产品与服务详情/迪士尼动画大全.doc'
    try:
        document = extract_document(path, '1-产品与服务详情/迪士尼动画大全.doc', assets_dir=tmp_path)
    except ExtractionError as error:
        if error.error_type in {'com_unavailable', 'com_disabled', 'com_failed'}:
            pytest.skip('当前环境无法使用 Office COM')
        raise

    assert document.source_type == 'ole_word'
    assert document.has_text


@pytest.mark.skipif(not KB_ROOT.is_dir(), reason='完整知识库不存在')
def test_real_scanned_pdf_is_marked_without_text(tmp_path: Path) -> None:
    path = KB_ROOT / '1-产品与服务详情/迪士尼公司发展分析.pdf'
    document = extract_document(path, path.name, assets_dir=tmp_path)

    assert not document.has_text
    assert len(document.images) == 33
    assert {image.page for image in document.images} == set(range(1, 34))


def _long_document() -> ExtractedDocument:
    sentence = '游客可以在官方渠道查询当天的开放时间和演出安排，并以现场公告为准。'
    blocks = [ExtractedBlock('heading', '一、测试章节', level=1)]
    blocks.extend(ExtractedBlock('text', sentence) for _ in range(60))
    blocks.append(ExtractedBlock('heading', '二、表格章节', level=1))
    table = '| 票种 | 价格 |\n|---|---|\n' + '\n'.join('| 票种' + str(index) + ' | 价格' + str(index) + ' |' for index in range(40))
    blocks.append(ExtractedBlock('table', table))
    return ExtractedDocument('1-产品与服务详情/test.docx', 'docx', '测试文档', blocks, [])


def test_chunking_respects_target_overlap_and_heading_boundaries() -> None:
    result = chunk_document(_long_document(), audience='public', config=ChunkConfig())
    text_chunks = [chunk for chunk in result.chunks if chunk.chunk_type.value == 'text']

    assert len(text_chunks) >= 3
    assert all(len(chunk.content) <= ChunkConfig().max_chars for chunk in text_chunks)
    assert all(chunk.section_path[:1] == ['一、测试章节'] for chunk in text_chunks)
    tail = overlap_tail(text_chunks[0].content, ChunkConfig().overlap_chars)
    assert tail
    assert tail in text_chunks[1].content
    assert any('二、表格章节' in chunk.content for chunk in result.chunks)


def test_long_table_is_kept_whole() -> None:
    result = chunk_document(_long_document(), audience='public', config=ChunkConfig())
    tables = [chunk for chunk in result.chunks if chunk.chunk_type.value == 'table']

    assert len(tables) == 1
    assert len(tables[0].content) > ChunkConfig().max_chars
    assert tables[0].content.count('| 票种') == 41


def test_public_chunks_are_redacted_and_internal_chunks_are_not() -> None:
    policy = ContentPolicy({
        'redaction_rules': [{
            'type': 'internal_phone',
            'pattern': '内部电话：[0-9-]+',
            'replacement': '内部联系方式已隐藏',
        }],
    })
    document = ExtractedDocument(
        '3-特殊情况与应急预案/应急.docx',
        'docx',
        '应急',
        [ExtractedBlock('heading', '一、求助', level=1), ExtractedBlock('text', '内部电话：021-12345678')],
        [],
    )

    public_result = chunk_document(document, audience='public', policy=policy)
    internal_result = chunk_document(document, audience='internal', policy=policy)

    assert all('021-12345678' not in chunk.content for chunk in public_result.chunks)
    assert any('021-12345678' in chunk.content for chunk in internal_result.chunks)
    assert public_result.redactions[0]['type'] == 'internal_phone'
    assert public_result.redactions[0]['source_file'] == '3-特殊情况与应急预案/应急.docx'


def test_policy_classifies_mixed_categories() -> None:
    policy = ContentPolicy.from_file(Path(__file__).resolve().parents[1] / 'config' / 'content_policy.yaml')

    assert policy.classify('1-产品与服务详情/票务.docx').audience.value == 'public'
    assert policy.classify('2-运营流程与标准作业程序/迪士尼乐园酒店信息 包括房型.docx').audience.value == 'public'
    assert policy.classify('2-运营流程与标准作业程序/迪士尼乐园度假区客诉处理流程.docx').audience.value == 'internal'
    assert policy.classify('5-内部知识与工具/员工手册.docx').audience.value == 'internal'
    emergency = policy.classify('3-特殊情况与应急预案/迪士尼乐园「紧急情况处理」一站式速查表.docx')
    assert emergency.audience.value == 'public'


@pytest.mark.skipif(not KB_ROOT.is_dir(), reason='完整知识库不存在')
def test_real_emergency_document_redacts_internal_extension() -> None:
    policy = ContentPolicy.from_file(Path(__file__).resolve().parents[1] / 'config' / 'content_policy.yaml')
    path = KB_ROOT / '3-特殊情况与应急预案/迪士尼乐园「紧急情况处理」一站式速查表.docx'
    document = extract_document(path, path.name)

    result = chunk_document(document, audience='public', policy=policy)

    content = '\n'.join(chunk.content for chunk in result.chunks)
    assert '转 3' not in content
    assert any(record['type'] == 'internal_extension' for record in result.redactions)
