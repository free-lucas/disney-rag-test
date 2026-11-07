from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest.caption import CaptionResult, ImageCaptioner, apply_captions_to_document
from ingest.chunk import ChunkConfig, chunk_document
from ingest.discover import DiscoveryResult, FileRecord, discover_files
from ingest.extract import ExtractionError, ExtractedDocument, extract_document
from ingest.images import FilteredImage, ImageFilterConfig, filter_images
from ingest.policy import ContentPolicy
from models import Audience, Chunk, ChunkType
from retrieval.bm25 import BM25Index
from retrieval.embed import DashScopeEmbedder, EmbeddingCache
from retrieval.faiss_io import write_index


PIPELINE_VERSION = 'ingest-v1'


class BuildError(RuntimeError):
    pass


@dataclass
class BuildFailure:
    source_file: str
    error_type: str
    error_message: str = ''
    chunk_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            'source_file': self.source_file,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'chunk_count': self.chunk_count,
        }


@dataclass
class DocumentCacheEntry:
    file_sha256: str
    signature: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    filtered_images: list[dict[str, Any]] = field(default_factory=list)
    caption_errors: list[dict[str, Any]] = field(default_factory=list)
    redactions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kept_image_hashes: list[str] = field(default_factory=list)
    caption_api_calls: int = 0
    caption_cached_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'file_sha256': self.file_sha256,
            'signature': self.signature,
            'chunks': self.chunks,
            'filtered_images': self.filtered_images,
            'caption_errors': self.caption_errors,
            'redactions': self.redactions,
            'warnings': self.warnings,
            'kept_image_hashes': self.kept_image_hashes,
            'caption_api_calls': self.caption_api_calls,
            'caption_cached_count': self.caption_cached_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DocumentCacheEntry':
        return cls(
            file_sha256=str(data.get('file_sha256', '')),
            signature=str(data.get('signature', '')),
            chunks=list(data.get('chunks', [])),
            filtered_images=list(data.get('filtered_images', [])),
            caption_errors=list(data.get('caption_errors', [])),
            redactions=list(data.get('redactions', [])),
            warnings=[str(item) for item in data.get('warnings', [])],
            kept_image_hashes=[str(item) for item in data.get('kept_image_hashes', [])],
            caption_api_calls=int(data.get('caption_api_calls', 0)),
            caption_cached_count=int(data.get('caption_cached_count', 0)),
        )


@dataclass
class BuildReport:
    build_version: str
    embedding_model: str
    embedding_dim: int
    total_files: int = 0
    duplicate_files: int = 0
    indexable_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    public_chunks: int = 0
    internal_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    filtered_images: int = 0
    caption_errors: int = 0
    redactions: int = 0
    embedding_api_calls: int = 0
    caption_api_calls: int = 0
    reused_files: list[str] = field(default_factory=list)
    rebuilt_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failures: list[BuildFailure] = field(default_factory=list)
    duplicate_records: list[dict[str, str]] = field(default_factory=list)
    filtered_image_records: list[dict[str, Any]] = field(default_factory=list)
    caption_error_records: list[dict[str, Any]] = field(default_factory=list)
    redaction_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    fallback_reason: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            'total_files': self.total_files,
            'duplicate_files': self.duplicate_files,
            'indexable_files': self.indexable_files,
            'indexed_files': self.indexed_files,
            'skipped_files': self.skipped_files,
            'failed_files': self.failed_files,
            'public_chunks': self.public_chunks,
            'internal_chunks': self.internal_chunks,
            'text_chunks': self.text_chunks,
            'table_chunks': self.table_chunks,
            'image_chunks': self.image_chunks,
            'filtered_images': self.filtered_images,
            'caption_errors': self.caption_errors,
            'redactions': self.redactions,
            'embedding_api_calls': self.embedding_api_calls,
            'caption_api_calls': self.caption_api_calls,
            'reused_file_count': len(self.reused_files),
            'rebuilt_file_count': len(self.rebuilt_files),
            'removed_file_count': len(self.removed_files),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            'build_version': self.build_version,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'summary': self.summary(),
            'reused_files': self.reused_files,
            'rebuilt_files': self.rebuilt_files,
            'removed_files': self.removed_files,
            'skipped': self.skipped,
            'failures': [item.to_dict() for item in self.failures],
            'duplicates': self.duplicate_records,
            'filtered_images': self.filtered_image_records,
            'caption_errors': self.caption_error_records,
            'redactions': self.redaction_records,
            'warnings': self.warnings,
            'artifacts': self.artifacts,
            'fallback_reason': self.fallback_reason,
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        return destination


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _signature(settings: object, policy: ContentPolicy) -> str:
    policy_bytes = Path(settings.content_policy_path).read_bytes()
    payload = '|'.join([
        PIPELINE_VERSION,
        hashlib.sha256(policy_bytes).hexdigest(),
        str(settings.deepseek_model),
        str(settings.caption_prompt_version),
        str(settings.chunk_target_chars),
        str(settings.chunk_min_chars),
        str(settings.chunk_max_chars),
        str(settings.chunk_overlap_chars),
        str(settings.image_min_width),
        str(settings.image_min_height),
        str(settings.image_min_area),
        str(settings.image_max_per_document),
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class _DocumentCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, file_sha256: str, signature: str) -> Path:
        return self.directory / (file_sha256 + '-' + signature[:12] + '.json')

    def load(self, file_sha256: str, signature: str) -> DocumentCacheEntry | None:
        path = self._path(file_sha256, signature)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            entry = DocumentCacheEntry.from_dict(payload)
        except (OSError, ValueError, TypeError):
            return None
        if entry.file_sha256 != file_sha256 or entry.signature != signature:
            return None
        return entry

    def save(self, entry: DocumentCacheEntry) -> Path:
        destination = self._path(entry.file_sha256, entry.signature)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(destination)
        return destination


def _build_document(
    *,
    root: Path,
    record: FileRecord,
    settings: object,
    policy: ContentPolicy,
    decision_audience: Audience,
    signature: str,
    build_version: str,
    captioner: object,
    seen_image_hashes: set[str],
    allow_com: bool,
) -> DocumentCacheEntry:
    source_path = root / record.relative_path
    assets_dir = Path(settings.data_dir) / 'assets' / record.sha256[:12]
    document = extract_document(
        source_path,
        record.relative_path,
        assets_dir=assets_dir,
        allow_com=allow_com,
    )
    filter_result = filter_images(
        document.images,
        config=ImageFilterConfig(
            min_width=settings.image_min_width,
            min_height=settings.image_min_height,
            min_area=settings.image_min_area,
            max_per_document=settings.image_max_per_document,
        ),
        seen_hashes=seen_image_hashes,
    )
    caption_result = captioner.caption_many(filter_result.kept)
    apply_captions_to_document(document, caption_result.captions)
    chunk_result = chunk_document(
        document,
        audience=decision_audience,
        config=ChunkConfig(
            target_chars=settings.chunk_target_chars,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        policy=policy,
        build_version=build_version,
    )
    warnings = list(document.warnings) + list(chunk_result.warnings)
    return DocumentCacheEntry(
        file_sha256=record.sha256,
        signature=signature,
        chunks=[chunk.to_dict() for chunk in chunk_result.chunks],
        filtered_images=[item.to_dict() for item in filter_result.filtered],
        caption_errors=[item.to_dict() for item in caption_result.errors],
        redactions=list(chunk_result.redactions),
        warnings=warnings,
        kept_image_hashes=[image.image_hash for image in filter_result.kept],
        caption_api_calls=int(getattr(caption_result, 'api_calls', 0)),
        caption_cached_count=int(getattr(caption_result, 'cached_count', 0)),
    )


def _previous_manifest(index_dir: Path) -> dict[str, Any]:
    path = index_dir / 'manifest.json'
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: Path, chunks: list[Chunk]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(chunk.to_json() + '\n')
    temporary.replace(path)
    return path


def build_index(
    settings: object,
    *,
    root: str | Path | None = None,
    index_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    embedder: object | None = None,
    captioner: object | None = None,
    allow_com: bool = True,
    force_rebuild: bool = False,
) -> BuildReport:
    knowledge_root = Path(root or settings.knowledge_base_root).resolve()
    destination_index = Path(index_dir or settings.index_dir)
    destination_index.mkdir(parents=True, exist_ok=True)
    report_destination = Path(report_path) if report_path else Path(settings.reports_dir) / 'build_report.json'
    if not knowledge_root.is_dir():
        raise BuildError('知识库目录不存在: ' + str(knowledge_root))
    if not Path(settings.content_policy_path).is_file():
        raise BuildError('内容策略文件不存在: ' + str(settings.content_policy_path))

    if embedder is None:
        settings.require_api_keys(deepseek=False, dashscope=True)
        embedder = DashScopeEmbedder.from_settings(
            settings,
            cache=EmbeddingCache(Path(settings.cache_dir) / 'embeddings'),
        )
    if captioner is None:
        settings.require_api_keys(deepseek=True, dashscope=False)
        captioner = ImageCaptioner.from_settings(settings)

    policy = ContentPolicy.from_file(settings.content_policy_path)
    signature = _signature(settings, policy)
    build_version = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    discovery = discover_files(knowledge_root)
    previous = _previous_manifest(destination_index)
    previous_hashes = previous.get('file_hashes', {}) if isinstance(previous.get('file_hashes', {}), dict) else {}
    current_hashes = {record.relative_path: record.sha256 for record in discovery.records}
    removed_files = sorted(set(previous_hashes) - set(current_hashes))

    document_cache = _DocumentCache(Path(settings.cache_dir) / 'documents')
    report = BuildReport(
        build_version=build_version,
        embedding_model=str(getattr(embedder, 'active_model', settings.embedding_model)),
        embedding_dim=0,
        total_files=len(discovery.records),
        duplicate_files=len(discovery.duplicates),
        duplicate_records=[
            {'relative_path': item.relative_path, 'duplicate_of': str(item.duplicate_of)}
            for item in discovery.duplicates
        ],
        removed_files=removed_files,
    )

    all_chunks: list[Chunk] = []
    seen_image_hashes: set[str] = set()
    for record in discovery.primary_records:
        if record.error_type:
            report.failures.append(BuildFailure(
                record.relative_path,
                record.error_type,
                record.error_message or '',
                0,
            ))
            continue
        decision = policy.classify(record.relative_path)
        if not decision.index:
            report.skipped.append({
                'source_file': record.relative_path,
                'rule_id': decision.rule_id,
                'reason': decision.reason or '内容策略不索引',
            })
            continue
        report.indexable_files += 1
        entry = None if force_rebuild else document_cache.load(record.sha256, signature)
        if entry is None:
            try:
                entry = _build_document(
                    root=knowledge_root,
                    record=record,
                    settings=settings,
                    policy=policy,
                    decision_audience=decision.audience,
                    signature=signature,
                    build_version=build_version,
                    captioner=captioner,
                    seen_image_hashes=seen_image_hashes,
                    allow_com=allow_com,
                )
                document_cache.save(entry)
                report.rebuilt_files.append(record.relative_path)
            except (ExtractionError, OSError, ValueError, RuntimeError) as exc:
                error_type = getattr(exc, 'error_type', exc.__class__.__name__)
                report.failures.append(BuildFailure(
                    record.relative_path,
                    str(error_type),
                    str(exc)[:1000],
                    0,
                ))
                continue
        else:
            report.reused_files.append(record.relative_path)
            seen_image_hashes.update(entry.kept_image_hashes)

        report.indexed_files += 1
        report.filtered_images += len(entry.filtered_images)
        report.filtered_image_records.extend(entry.filtered_images)
        report.caption_errors += len(entry.caption_errors)
        report.caption_error_records.extend(entry.caption_errors)
        report.caption_api_calls += entry.caption_api_calls
        report.redactions += len(entry.redactions)
        report.redaction_records.extend(entry.redactions)
        report.warnings.extend(entry.warnings)
        for payload in entry.chunks:
            chunk = Chunk.from_dict(payload)
            chunk.build_version = build_version
            all_chunks.append(chunk)

    report.failed_files = len(report.failures)
    report.skipped_files = len(report.skipped)
    if not all_chunks:
        raise BuildError('没有可建立索引的 chunk')

    texts = [chunk.content for chunk in all_chunks]
    matrix = embedder.embed_texts(texts)
    if matrix.ndim != 2 or matrix.shape[0] != len(all_chunks):
        raise BuildError('embedding 矩阵与 chunk 数量不一致')
    report.embedding_dim = int(matrix.shape[1])
    report.embedding_model = str(getattr(embedder, 'active_model', settings.embedding_model))
    report.embedding_api_calls = int(getattr(embedder, 'api_calls', 0))
    report.fallback_reason = getattr(embedder, 'last_fallback_reason', None)
    for chunk in all_chunks:
        chunk.embedding_model = report.embedding_model
        chunk.embedding_dim = report.embedding_dim

    import faiss

    vectors = matrix.astype('float32')
    faiss.normalize_L2(vectors)
    faiss_index = faiss.IndexFlatIP(report.embedding_dim)
    faiss_index.add(vectors)
    faiss_path = destination_index / 'faiss.index'
    write_index(faiss_index, faiss_path)

    chunks_path = _write_jsonl(destination_index / 'chunks.jsonl', all_chunks)
    bm25_path = BM25Index.from_texts(texts).save(destination_index / 'bm25.pkl')

    report.public_chunks = sum(1 for chunk in all_chunks if chunk.audience is Audience.PUBLIC)
    report.internal_chunks = sum(1 for chunk in all_chunks if chunk.audience is Audience.INTERNAL)
    report.text_chunks = sum(1 for chunk in all_chunks if chunk.chunk_type is ChunkType.TEXT)
    report.table_chunks = sum(1 for chunk in all_chunks if chunk.chunk_type is ChunkType.TABLE)
    report.image_chunks = sum(1 for chunk in all_chunks if chunk.chunk_type is ChunkType.IMAGE_CAPTION)
    report.artifacts = {
        'faiss_index': str(faiss_path),
        'chunks': str(chunks_path),
        'bm25': str(bm25_path),
        'manifest': str(destination_index / 'manifest.json'),
        'build_report': str(report_destination),
    }

    manifest = {
        'schema_version': 1,
        'pipeline_version': PIPELINE_VERSION,
        'build_version': build_version,
        'embedding_model': report.embedding_model,
        'embedding_dim': report.embedding_dim,
        'total_chunks': len(all_chunks),
        'public_chunks': report.public_chunks,
        'internal_chunks': report.internal_chunks,
        'file_hashes': current_hashes,
        'pipeline_signature': signature,
        'artifacts': report.artifacts,
    }
    manifest_path = destination_index / 'manifest.json'
    temporary_manifest = manifest_path.with_suffix('.json.tmp')
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary_manifest.replace(manifest_path)
    report.write(report_destination)
    return report
