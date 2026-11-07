from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JPEG_MAGIC = b'\xff\xd8\xff'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
GIF_MAGIC = b'GIF8'
PDF_MAGIC = b'%PDF'
ZIP_MAGIC = b'PK\x03\x04'
OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'


@dataclass
class FileRecord:
    relative_path: str
    extension: str
    size: int
    sha256: str
    detected_format: str
    ole_kind: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'relative_path': self.relative_path,
            'extension': self.extension,
            'size': self.size,
            'sha256': self.sha256,
            'detected_format': self.detected_format,
            'ole_kind': self.ole_kind,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'duplicate_of': self.duplicate_of,
        }


@dataclass
class DiscoveryResult:
    records: list[FileRecord] = field(default_factory=list)
    duplicates: list[FileRecord] = field(default_factory=list)
    errors: list[FileRecord] = field(default_factory=list)

    @property
    def primary_records(self) -> list[FileRecord]:
        return [record for record in self.records if record.duplicate_of is None]

    def summary(self) -> dict[str, Any]:
        formats: dict[str, int] = {}
        for record in self.records:
            formats[record.detected_format] = formats.get(record.detected_format, 0) + 1
        return {
            'total_files': len(self.records),
            'primary_files': len(self.primary_records),
            'duplicate_files': len(self.duplicates),
            'failed_files': len(self.errors),
            'formats': formats,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            'summary': self.summary(),
            'records': [record.to_dict() for record in self.records],
            'duplicates': [record.to_dict() for record in self.duplicates],
            'errors': [record.to_dict() for record in self.errors],
        }

    def write_report(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return destination


@dataclass(frozen=True)
class FormatInfo:
    detected_format: str
    ole_kind: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ole_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == '.doc':
        return 'word'
    if suffix == '.ppt':
        return 'powerpoint'
    if suffix == '.xls':
        return 'excel'
    return None


def _detect_zip(path: Path) -> FormatInfo:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError) as exc:
        return FormatInfo('corrupted_archive', error_type='bad_zip', error_message=str(exc))
    if not names:
        return FormatInfo('corrupted_archive', error_type='empty_zip', error_message='压缩包没有条目')
    if 'word/document.xml' in names:
        return FormatInfo('docx')
    if 'ppt/presentation.xml' in names:
        return FormatInfo('pptx')
    if 'xl/workbook.xml' in names:
        return FormatInfo('xlsx')
    return FormatInfo('unsupported_zip', error_type='unsupported_zip', error_message='未识别的 OOXML 容器')


def detect_format(path: str | Path) -> FormatInfo:
    candidate = Path(path)
    try:
        with candidate.open('rb') as handle:
            header = handle.read(16)
    except OSError as exc:
        return FormatInfo('unknown', error_type='read_error', error_message=str(exc))
    if header.startswith(JPEG_MAGIC):
        return FormatInfo('jpeg')
    if header.startswith(PNG_MAGIC):
        return FormatInfo('image')
    if header.startswith(GIF_MAGIC):
        return FormatInfo('image')
    if header.startswith(PDF_MAGIC):
        return FormatInfo('pdf')
    if header.startswith(ZIP_MAGIC) or header.startswith(b'PK\x05\x06'):
        return _detect_zip(candidate)
    if header.startswith(OLE_MAGIC):
        kind = _ole_kind(candidate)
        return FormatInfo('ole', ole_kind=kind)
    if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return FormatInfo('image')
    return FormatInfo('unknown', error_type='unknown_format', error_message='无法识别的文件签名')


SUPPORTED_FORMATS = {'docx', 'pptx', 'pdf', 'jpeg', 'image', 'ole'}


def discover_files(knowledge_base_root: str | Path) -> DiscoveryResult:
    root = Path(knowledge_base_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('知识库目录不存在: ' + str(root))
    records: list[FileRecord] = []
    for path in sorted((item for item in root.rglob('*') if item.is_file()), key=lambda item: str(item).casefold()):
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError as exc:
            size = -1
            records.append(FileRecord(relative, path.suffix.lower(), size, '', 'unknown', error_type='stat_error', error_message=str(exc)))
            continue
        try:
            digest = sha256_file(path)
        except OSError as exc:
            records.append(FileRecord(relative, path.suffix.lower(), size, '', 'unknown', error_type='hash_error', error_message=str(exc)))
            continue
        info = detect_format(path)
        records.append(FileRecord(
            relative_path=relative,
            extension=path.suffix.lower(),
            size=size,
            sha256=digest,
            detected_format=info.detected_format,
            ole_kind=info.ole_kind,
            error_type=info.error_type,
            error_message=info.error_message,
        ))

    duplicates: list[FileRecord] = []
    groups: dict[str, list[FileRecord]] = {}
    for record in records:
        if record.sha256:
            groups.setdefault(record.sha256, []).append(record)
    for same_hash in groups.values():
        if len(same_hash) < 2:
            continue
        primary = sorted(same_hash, key=lambda item: (item.error_type is not None, item.detected_format not in SUPPORTED_FORMATS, item.relative_path.casefold()))[0]
        for record in same_hash:
            if record is primary:
                continue
            record.duplicate_of = primary.relative_path
            duplicates.append(record)

    errors = [record for record in records if record.error_type is not None]
    return DiscoveryResult(records=records, duplicates=duplicates, errors=errors)


def format_summary(result: DiscoveryResult) -> str:
    summary = result.summary()
    formats = ', '.join(
        key + '=' + str(value) for key, value in sorted(summary['formats'].items())
    )
    lines = [
        '发现文件: ' + str(summary['total_files']),
        '可索引主来源: ' + str(summary['primary_files']),
        '重复文件: ' + str(summary['duplicate_files']),
        '失败文件: ' + str(summary['failed_files']),
        '格式分布: ' + (formats or '无'),
    ]
    for record in result.errors:
        lines.append('失败: ' + record.relative_path + ' [' + str(record.error_type) + '] ' + str(record.error_message or ''))
    for record in result.duplicates:
        lines.append('重复: ' + record.relative_path + ' -> ' + str(record.duplicate_of))
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='盘点迪士尼知识库文件并输出格式识别报告')
    parser.add_argument('--root', default=None, help='知识库根目录，默认读取配置中的 DISNEY_KB_ROOT')
    parser.add_argument('--output', default=None, help='报告输出路径，默认写入 data/reports/discover_report.json')
    args = parser.parse_args(argv)

    if args.root:
        root = Path(args.root)
    else:
        from config import load_settings

        root = load_settings(require_api_keys=False).knowledge_base_root

    try:
        result = discover_files(root)
    except FileNotFoundError as exc:
        print('盘点失败: ' + str(exc))
        return 2

    output = Path(args.output) if args.output else Path('data/reports/discover_report.json')
    destination = result.write_report(output)
    print(format_summary(result))
    print('报告已写入: ' + str(destination))
    return 1 if result.errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
