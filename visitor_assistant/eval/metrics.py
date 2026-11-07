from __future__ import annotations

from eval.golden import normalize_source


def source_matches(actual: str, expected: str) -> bool:
    actual_normalized = normalize_source(actual)
    expected_normalized = normalize_source(expected)
    if actual_normalized == expected_normalized:
        return True
    if expected_normalized in actual_normalized:
        return True
    return (
        actual_normalized.endswith('/' + expected_normalized)
        or expected_normalized.endswith('/' + actual_normalized)
    )


def first_relevant_rank(results: list[object], expected_files: list[str], k: int) -> int | None:
    for rank, result in enumerate(results[:k], start=1):
        source_file = str(result.chunk.source_file)
        if any(source_matches(source_file, expected) for expected in expected_files):
            return rank
    return None


def score_retrieval(
    results: list[object],
    expected_files: list[str],
    *,
    k: int = 6,
) -> dict[str, object]:
    top_k = []
    for result in results[:k]:
        top_k.append({
            'source_file': str(result.chunk.source_file),
            'chunk_id': str(result.chunk.id),
            'score': round(float(result.score), 6),
        })
    if not expected_files:
        return {
            'hit': None,
            'rank': None,
            'mrr': None,
            'expected': [],
            'top_k': top_k,
            'skipped': 'expect_refusal',
        }
    rank = first_relevant_rank(results, expected_files, k)
    return {
        'hit': rank is not None,
        'rank': rank,
        'mrr': (1.0 / rank) if rank is not None else 0.0,
        'expected': list(expected_files),
        'top_k': top_k,
        'skipped': None,
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_records(records: list[dict[str, object]], *, with_generation: bool) -> dict[str, object]:
    retrieval_records = [
        record for record in records
        if isinstance(record.get('retrieval'), dict)
        and not record['retrieval'].get('error')
        and record['retrieval'].get('hit') is not None
    ]
    hit_values = [1.0 if record['retrieval'].get('hit') else 0.0 for record in retrieval_records]
    mrr_values = [float(record['retrieval'].get('mrr') or 0.0) for record in retrieval_records]

    checks = [
        record['checks'] for record in records
        if isinstance(record.get('checks'), dict)
    ]
    answer_records = [
        record for record in records
        if not record.get('expect_refusal') and isinstance(record.get('checks'), dict)
    ]
    citation_values = [
        1.0 if record['checks'].get('citation_present') else 0.0
        for record in answer_records
    ] if with_generation else []
    refusal_values = [
        1.0 if check.get('refusal_correct') else 0.0 for check in checks
    ] if with_generation else []
    leaks = sum(1 for check in checks if check.get('internal_leak')) if with_generation else 0
    passed = sum(1 for check in checks if check.get('passed')) if with_generation else None

    errors = sum(1 for record in records if record.get('error'))
    return {
        'total': len(records),
        'retrieval_evaluated': len(retrieval_records),
        'hit_at_k': _mean(hit_values),
        'mrr': _mean(mrr_values),
        'answer_evaluated': len(answer_records) if with_generation else 0,
        'citation_rate': _mean(citation_values),
        'refusal_accuracy': _mean(refusal_values),
        'internal_leaks': leaks,
        'passed': passed,
        'failed': (len(records) - int(passed or 0)) if with_generation else None,
        'errors': errors,
    }


def aggregate_by_category(records: list[dict[str, object]], *, with_generation: bool) -> dict[str, object]:
    categories: dict[str, list[dict[str, object]]] = {}
    for record in records:
        categories.setdefault(str(record.get('category', 'unknown')), []).append(record)
    return {
        category: aggregate_records(items, with_generation=with_generation)
        for category, items in sorted(categories.items())
    }


def collect_misses(records: list[dict[str, object]]) -> list[dict[str, object]]:
    misses: list[dict[str, object]] = []
    for record in records:
        retrieval = record.get('retrieval')
        if not isinstance(retrieval, dict) or retrieval.get('hit') is None:
            continue
        if retrieval.get('hit'):
            continue
        misses.append({
            'id': record.get('id'),
            'category': record.get('category'),
            'question': record.get('question'),
            'expected': retrieval.get('expected', []),
            'top_k': retrieval.get('top_k', []),
            'error': retrieval.get('error'),
        })
    return misses
