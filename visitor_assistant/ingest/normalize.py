from __future__ import annotations

import tempfile
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ingest.discover import FormatInfo, detect_format


class NormalizationError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class NormalizedSource:
    path: Path
    source_format: str
    source_type: str
    ole_kind: str | None = None
    converted: bool = False


def convert_ole(source: str | Path, ole_kind: str, output_dir: str | Path, *, enabled: bool = True) -> Path:
    if not enabled:
        raise NormalizationError('com_disabled', '已禁用 Office COM 转换')
    if ole_kind == 'excel':
        raise NormalizationError('unsupported_ole_kind', '暂不支持旧版 Excel 文档')
    if ole_kind not in {'word', 'powerpoint'}:
        raise NormalizationError('unsupported_ole_kind', '无法判断旧版 Office 文档类型')

    source_path = Path(source).resolve()
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = '.docx' if ole_kind == 'word' else '.pptx'
    destination = destination_dir / (source_path.stem + suffix)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                '-m',
                'ingest.com_convert',
                '--source',
                str(source_path),
                '--kind',
                ole_kind,
                '--destination',
                str(destination),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise NormalizationError('com_timeout', 'Office COM 转换超时') from exc
    except OSError as exc:
        raise NormalizationError('com_unavailable', '无法启动 Office COM 转换进程: ' + str(exc)) from exc

    if destination.is_file() and destination.stat().st_size > 0:
        info = detect_format(destination)
        if info.detected_format in {'docx', 'pptx'}:
            return destination
    message = (completed.stderr or completed.stdout or '').strip()
    if not message:
        message = '转换进程退出码 ' + str(completed.returncode)
    raise NormalizationError('com_failed', 'Office COM 转换失败: ' + message[-500:])


def _source_type(info: FormatInfo) -> str:
    if info.detected_format == 'ole':
        return 'ole_' + str(info.ole_kind or 'unknown')
    return info.detected_format


@contextmanager
def normalized_source(path: str | Path, *, allow_com: bool = True) -> Iterator[NormalizedSource]:
    source = Path(path)
    info = detect_format(source)
    if info.detected_format in {'docx', 'pptx', 'pdf', 'jpeg', 'image'}:
        yield NormalizedSource(source, info.detected_format, _source_type(info))
        return
    if info.detected_format == 'ole':
        if info.ole_kind not in {'word', 'powerpoint', 'excel'}:
            raise NormalizationError('unsupported_ole_kind', '无法判断旧版 Office 文档类型')
        if info.ole_kind == 'excel':
            raise NormalizationError('unsupported_ole_kind', '暂不支持旧版 Excel 文档')
        with tempfile.TemporaryDirectory(prefix='disney_rag_', ignore_cleanup_errors=True) as temp_dir:
            converted = convert_ole(source, info.ole_kind, temp_dir, enabled=allow_com)
            converted_info = detect_format(converted)
            yield NormalizedSource(
                converted,
                converted_info.detected_format,
                _source_type(info),
                ole_kind=info.ole_kind,
                converted=True,
            )
        return
    error_type = info.error_type or 'unsupported_format'
    message = info.error_message or '不支持的文件格式'
    raise NormalizationError(error_type, message)
