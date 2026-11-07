import json
import zipfile
from pathlib import Path

import pytest

from ingest.discover import detect_format, discover_files, sha256_file


def _write_zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, 'w') as archive:
        for name in names:
            archive.writestr(name, '<xml/>')


def test_detects_ooxml_from_signature_not_extension(tmp_path: Path) -> None:
    docx = tmp_path / 'legacy.doc'
    pptx = tmp_path / 'legacy.ppt'
    _write_zip(docx, ['word/document.xml'])
    _write_zip(pptx, ['ppt/presentation.xml'])

    assert detect_format(docx).detected_format == 'docx'
    assert detect_format(pptx).detected_format == 'pptx'


def test_detects_ole_by_signature(tmp_path: Path) -> None:
    path = tmp_path / 'old.doc'
    path.write_bytes(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 32)

    info = detect_format(path)

    assert info.detected_format == 'ole'
    assert info.ole_kind == 'word'


def test_detects_pdf_jpeg_and_empty_zip(tmp_path: Path) -> None:
    pdf = tmp_path / 'scan.bin'
    jpeg = tmp_path / 'photo.dat'
    empty_zip = tmp_path / 'broken.ppt'
    pdf.write_bytes(b'%PDF-1.7\n')
    jpeg.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 16)
    with zipfile.ZipFile(empty_zip, 'w'):
        pass

    assert detect_format(pdf).detected_format == 'pdf'
    assert detect_format(jpeg).detected_format == 'jpeg'
    broken = detect_format(empty_zip)
    assert broken.detected_format == 'corrupted_archive'
    assert broken.error_type == 'empty_zip'


def test_discovery_records_duplicates_and_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / 'a' / 'one.pdf'
    second = tmp_path / 'b' / 'two.pdf'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b'%PDF-1.4\nsame')
    second.write_bytes(b'%PDF-1.4\nsame')
    unique = tmp_path / 'unique.jpg'
    unique.write_bytes(b'\xff\xd8\xff\xe0unique')

    result = discover_files(tmp_path)

    assert result.summary()['total_files'] == 3
    assert result.summary()['duplicate_files'] == 1
    assert len(result.primary_records) == 2
    assert all(not record.relative_path.startswith(str(tmp_path)) for record in result.records)
    assert result.duplicates[0].duplicate_of in {'a/one.pdf', 'b/two.pdf'}
    assert sha256_file(first) == sha256_file(second)


def test_corrupted_file_does_not_stop_discovery(tmp_path: Path) -> None:
    broken = tmp_path / 'broken.ppt'
    broken.write_bytes(b'PK\x03\x04not-a-zip')
    valid = tmp_path / 'valid.pdf'
    valid.write_bytes(b'%PDF-1.5\ncontent')

    result = discover_files(tmp_path)

    assert result.summary()['total_files'] == 2
    assert result.summary()['failed_files'] == 1
    assert result.errors[0].relative_path == 'broken.ppt'
    assert result.errors[0].error_type == 'bad_zip'
    assert any(record.relative_path == 'valid.pdf' and record.error_type is None for record in result.records)


def test_writes_json_report(tmp_path: Path) -> None:
    source = tmp_path / 'kb'
    source.mkdir()
    (source / 'doc.pdf').write_bytes(b'%PDF-1.4\ndata')
    result = discover_files(source)
    destination = result.write_report(tmp_path / 'reports' / 'discover.json')

    payload = json.loads(destination.read_text(encoding='utf-8'))

    assert payload['summary']['total_files'] == 1
    assert payload['records'][0]['relative_path'] == 'doc.pdf'
    assert payload['records'][0]['detected_format'] == 'pdf'


@pytest.mark.parametrize('relative_path, expected', [
    ('2-运营流程与标准作业程序/迪士尼乐园酒店信息 包括各家迪士尼酒店的房型、定价、设施（泳池、健身房）、入住_退房政策和酒店宾客专属福利.docx', 'docx'),
    ('3-特殊情况与应急预案/迪士尼乐园「紧急情况处理」一站式速查表.docx', 'docx'),
])
def test_real_knowledge_base_has_expected_formats(relative_path: str, expected: str) -> None:
    root = Path(__file__).resolve().parents[2] / '迪士尼RAG知识库（完整）'
    if not root.is_dir():
        pytest.skip('完整知识库不存在')

    info = detect_format(root / relative_path)

    assert info.detected_format == expected


def test_complete_knowledge_base_file_count() -> None:
    root = Path(__file__).resolve().parents[2] / '迪士尼RAG知识库（完整）'
    if not root.is_dir():
        pytest.skip('完整知识库不存在')

    result = discover_files(root)

    assert result.summary()['total_files'] == 65
    assert result.summary()['failed_files'] == 0
