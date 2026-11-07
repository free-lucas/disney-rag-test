from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.evaluate import (
    compare_reports,
    render_comparison,
    run_evaluation,
    write_reports,
)
from eval.golden import (
    REQUIRED_CATEGORIES,
    GoldenQuestion,
    GoldenQuestionError,
    load_golden_questions,
)
from eval.metrics import aggregate_records, collect_misses, score_retrieval, source_matches
from eval.quality import detect_internal_leak, evaluate_answer
from models import Answer, Audience, Chunk, Citation, RetrievalResult


GOLDEN_PATH = Path(__file__).resolve().parents[1] / 'eval' / 'golden_questions.json'


def _result(
    source_file: str = '1-产品与服务详情/票务.docx',
    *,
    identifier: str = 'c1',
    score: float = 0.9,
) -> RetrievalResult:
    chunk = Chunk(
        identifier,
        '一日票价格为 475 元。',
        source_file,
        '1-产品与服务详情',
        Audience.PUBLIC,
    )
    return RetrievalResult(chunk=chunk, score=score)


def _answer(
    text: str = '一日票价格为 475 元 [1]。',
    *,
    citations: list[Citation] | None = None,
    refused: bool = False,
) -> Answer:
    return Answer(
        text=text,
        citations=list(citations or []),
        refused=refused,
        query='q',
        rewritten_query='q',
    )


def _citation(source_file: str = '1-产品与服务详情/票务.docx') -> Citation:
    return Citation(source_file=source_file, section_path=['门票价格'])


def test_golden_file_loads_all_required_categories() -> None:
    questions = load_golden_questions(GOLDEN_PATH)

    assert len(questions) == 13
    assert {item.category for item in questions} == REQUIRED_CATEGORIES
    assert len({item.id for item in questions}) == 13
    for item in questions:
        if item.category == 'out_of_domain':
            assert item.expect_refusal is True
            assert item.expected_source_files == []
        else:
            assert item.expect_refusal is False
            assert item.expected_source_files


def test_golden_loader_rejects_invalid_payloads(tmp_path: Path) -> None:
    base = json.loads(GOLDEN_PATH.read_text(encoding='utf-8'))

    duplicate = copy.deepcopy(base)
    duplicate['questions'].append(copy.deepcopy(duplicate['questions'][0]))

    unknown = copy.deepcopy(base)
    unknown['questions'][0]['category'] = 'weather'

    out_of_domain = copy.deepcopy(base)
    out_item = next(item for item in out_of_domain['questions'] if item['category'] == 'out_of_domain')
    out_item['expect_refusal'] = False

    unsourced = copy.deepcopy(base)
    answer_item = next(item for item in unsourced['questions'] if item['category'] == 'ticket')
    answer_item['expected_source_files'] = []

    missing_category = copy.deepcopy(base)
    missing_category['questions'] = [
        item for item in missing_category['questions'] if item['category'] != 'cruise'
    ]

    cases = {
        'duplicate': (duplicate, '重复'),
        'unknown': (unknown, '未知类别'),
        'out_of_domain': (out_of_domain, '域外题'),
        'unsourced': (unsourced, '期望来源'),
        'missing_category': (missing_category, '缺少必需类别'),
    }
    for name, (payload, message) in cases.items():
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        with pytest.raises(GoldenQuestionError, match=message):
            load_golden_questions(path)

    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{not valid json', encoding='utf-8')
    with pytest.raises(GoldenQuestionError, match='无法读取'):
        load_golden_questions(malformed)


def test_source_matching_normalizes_windows_paths() -> None:
    assert source_matches('a\\b\\ticket.docx', 'b/ticket.docx')
    assert source_matches('./b/ticket.docx', 'b/ticket.docx')
    assert not source_matches('a/other.docx', 'b/ticket.docx')


def test_hit_at_k_and_mrr_are_hand_calculated() -> None:
    results = [
        _result('a.docx', identifier='a'),
        _result('ticket.docx', identifier='b'),
        _result('c.docx', identifier='c'),
    ]

    scored = score_retrieval(results, ['folder/ticket.docx'], k=3)

    assert scored['hit'] is True
    assert scored['rank'] == 2
    assert scored['mrr'] == 0.5
    assert scored['skipped'] is None
    assert [item['chunk_id'] for item in scored['top_k']] == ['a', 'b', 'c']

    cutoff = score_retrieval(results, ['folder/ticket.docx'], k=1)

    assert cutoff['hit'] is False
    assert cutoff['rank'] is None
    assert cutoff['mrr'] == 0.0

    skipped = score_retrieval(results, [], k=3)

    assert skipped['hit'] is None
    assert skipped['mrr'] is None
    assert skipped['skipped'] == 'expect_refusal'


def test_retrieval_metrics_are_hand_calculated() -> None:
    records = [
        {'retrieval': {'hit': True, 'mrr': 1.0}, 'checks': None},
        {'retrieval': {'hit': True, 'mrr': 0.5}, 'checks': None},
        {'retrieval': {'hit': False, 'mrr': 0.0}, 'checks': None},
    ]

    summary = aggregate_records(records, with_generation=False)

    assert summary['total'] == 3
    assert summary['retrieval_evaluated'] == 3
    assert summary['hit_at_k'] == 0.6667
    assert summary['mrr'] == 0.5
    assert summary['citation_rate'] is None
    assert summary['refusal_accuracy'] is None
    assert summary['internal_leaks'] == 0
    assert summary['passed'] is None


def test_generation_metrics_are_hand_calculated() -> None:
    passed = {
        'citation_present': True,
        'refusal_correct': True,
        'internal_leak': False,
        'passed': True,
    }
    refused = {
        'citation_present': False,
        'refusal_correct': True,
        'internal_leak': False,
        'passed': True,
    }
    leaked = {
        'citation_present': True,
        'refusal_correct': True,
        'internal_leak': True,
        'passed': False,
    }
    records = [
        {'expect_refusal': False, 'retrieval': {'hit': True, 'mrr': 1.0}, 'checks': passed},
        {'expect_refusal': True, 'retrieval': {'hit': None, 'mrr': None}, 'checks': refused},
        {'expect_refusal': False, 'retrieval': {'hit': True, 'mrr': 1.0}, 'checks': leaked},
    ]

    summary = aggregate_records(records, with_generation=True)

    assert summary['answer_evaluated'] == 2
    assert summary['citation_rate'] == 1.0
    assert summary['refusal_accuracy'] == 1.0
    assert summary['internal_leaks'] == 1
    assert summary['passed'] == 2
    assert summary['failed'] == 1


def test_collect_misses_includes_expected_and_actual_sources() -> None:
    records = [{
        'id': 'ticket-001',
        'category': 'ticket',
        'question': '门票多少钱',
        'expect_refusal': False,
        'retrieval': {
            'hit': False,
            'mrr': 0.0,
            'expected': ['1-产品与服务详情/票务.docx'],
            'top_k': [{'source_file': 'other.docx', 'chunk_id': 'x', 'score': 0.4}],
            'error': None,
        },
        'checks': None,
        'answer': None,
        'error': None,
    }]

    misses = collect_misses(records)

    assert len(misses) == 1
    assert misses[0]['id'] == 'ticket-001'
    assert misses[0]['expected'] == ['1-产品与服务详情/票务.docx']
    assert misses[0]['top_k'][0]['source_file'] == 'other.docx'


def test_evaluate_answer_requires_citations_for_answers() -> None:
    checks = evaluate_answer(_answer(citations=[]), expect_refusal=False)

    assert checks['citation_present'] is False
    assert checks['refusal_correct'] is True
    assert checks['passed'] is False


def test_evaluate_answer_marks_wrong_refusal_decision() -> None:
    checks = evaluate_answer(_answer(refused=False), expect_refusal=True)

    assert checks['refusal_correct'] is False
    assert checks['passed'] is False


def test_evaluate_answer_flags_rule_based_internal_leak() -> None:
    answer = _answer('请联系内部电话 021-12345678 [1]。', citations=[_citation()])

    checks = evaluate_answer(answer, expect_refusal=False)

    assert checks['internal_leak'] is True
    assert checks['leak_detail'] == '内部电话'
    assert checks['passed'] is False


def test_evaluate_answer_flags_reused_internal_text() -> None:
    internal = '游客服务团队会在十分钟内完成现场安全确认。'
    answer = _answer('请注意。' + internal, citations=[_citation()])

    assert detect_internal_leak(answer.text, [internal]) == internal.rstrip('。')
    checks = evaluate_answer(answer, expect_refusal=False, internal_texts=[internal])

    assert checks['internal_leak'] is True
    assert checks['leak_detail'] == internal.rstrip('。')
    assert checks['passed'] is False


def test_write_reports_creates_json_and_markdown(tmp_path: Path) -> None:
    summary = {
        'total': 1,
        'retrieval_evaluated': 1,
        'hit_at_k': 0.5,
        'mrr': 0.5,
        'answer_evaluated': 1,
        'citation_rate': 1.0,
        'refusal_accuracy': 1.0,
        'internal_leaks': 0,
        'passed': 1,
        'failed': 0,
        'errors': 0,
    }
    report = {
        'version': 1,
        'generated_at': '2026-11-07T12:00:00+00:00',
        'model': 'deepseek-flash',
        'top_k': 6,
        'with_generation': True,
        'summary': summary,
        'by_category': {'ticket': summary},
        'questions': [],
        'misses': [{
            'id': 'ticket-001',
            'category': 'ticket',
            'question': '门票多少钱',
            'expected': ['票务.docx'],
            'top_k': [{'source_file': 'other.docx', 'score': 0.4}],
        }],
        'errors': [{'id': 'q1', 'stage': 'retrieval', 'error': 'boom'}],
    }

    json_path, markdown_path = write_reports(report, tmp_path)

    assert json_path.name == 'eval_report_2026-11-07T12-00-00_00-00.json'
    assert markdown_path.name == 'eval_report_2026-11-07T12-00-00_00-00.md'
    assert json.loads(json_path.read_text(encoding='utf-8')) == report
    markdown = markdown_path.read_text(encoding='utf-8')
    assert markdown.startswith('# 迪士尼游客 RAG 评测报告')
    assert '| hit@k | 0.5000 |' in markdown
    assert 'ticket-001' in markdown
    assert '未命中题目' in markdown
    assert '失败记录' in markdown
    assert 'deepseek-flash' in markdown


class _ListRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = list(results)

    def retrieve(self, question: str, history: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace(
            results=list(self.results),
            rewrite=SimpleNamespace(rewritten_query=question),
        )


class _FlakyRetriever:
    def retrieve(self, question: str, history: list[dict[str, str]]) -> SimpleNamespace:
        if '失败' in question:
            raise RuntimeError('retrieval boom')
        return SimpleNamespace(
            results=[_result()],
            rewrite=SimpleNamespace(rewritten_query=question),
        )


class _FlakyGenerator:
    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        history: list[dict[str, str]],
        *,
        rewritten_query: str = '',
    ) -> Answer:
        if '失败' in question:
            raise RuntimeError('generation boom')
        return _answer(citations=[_citation()])


def test_run_evaluation_records_hits_and_misses() -> None:
    question = GoldenQuestion(
        'ticket-001',
        'ticket',
        '门票多少钱',
        ['1-产品与服务详情/票务.docx'],
    )

    hit_report = run_evaluation(
        [question],
        _ListRetriever([_result()]),
        None,
        top_k=2,
        with_generation=False,
        generated_at='fixed',
        model='deepseek-flash',
    )

    assert hit_report['top_k'] == 2
    assert hit_report['summary']['hit_at_k'] == 1.0
    assert hit_report['summary']['mrr'] == 1.0
    assert hit_report['questions'][0]['retrieval']['rank'] == 1
    assert hit_report['questions'][0]['retrieval']['top_k'][0]['chunk_id'] == 'c1'
    assert hit_report['misses'] == []

    miss_report = run_evaluation(
        [question],
        _ListRetriever([_result('other.docx')]),
        None,
        top_k=2,
        with_generation=False,
        generated_at='fixed',
        model='deepseek-flash',
    )

    assert miss_report['summary']['hit_at_k'] == 0.0
    assert len(miss_report['misses']) == 1
    assert miss_report['misses'][0]['id'] == 'ticket-001'


def test_retrieval_failure_is_isolated_per_question() -> None:
    questions = [
        GoldenQuestion('q1', 'ticket', '失败问题', ['票务.docx']),
        GoldenQuestion('q2', 'ticket', '正常问题', ['票务.docx']),
    ]

    report = run_evaluation(
        questions,
        _FlakyRetriever(),
        None,
        with_generation=False,
        generated_at='fixed',
    )

    first, second = report['questions']
    assert first['error'] == 'RuntimeError: retrieval boom'
    assert first['checks'] is None
    assert second['retrieval']['hit'] is True
    assert report['summary']['errors'] == 1
    assert report['errors'][0]['stage'] == 'retrieval'


def test_generation_failure_is_isolated_per_question() -> None:
    questions = [
        GoldenQuestion('q1', 'ticket', '失败问题', ['票务.docx']),
        GoldenQuestion('q2', 'ticket', '正常问题', ['票务.docx']),
    ]

    report = run_evaluation(
        questions,
        _ListRetriever([_result()]),
        _FlakyGenerator(),
        with_generation=True,
        generated_at='fixed',
    )

    first, second = report['questions']
    assert first['error'] == 'RuntimeError: generation boom'
    assert first['checks']['passed'] is False
    assert second['checks']['passed'] is True
    assert report['summary']['passed'] == 1
    assert report['summary']['failed'] == 1
    assert report['errors'][0]['stage'] == 'generation'


def test_compare_reports_computes_overall_and_category_deltas() -> None:
    old_summary = {
        'hit_at_k': 0.5,
        'mrr': 0.4,
        'citation_rate': 0.8,
        'refusal_accuracy': 1.0,
        'internal_leaks': 1,
    }
    new_summary = {
        'hit_at_k': 0.75,
        'mrr': 0.5,
        'citation_rate': 0.9,
        'refusal_accuracy': 1.0,
        'internal_leaks': 0,
    }
    old = {
        'generated_at': 'old',
        'summary': old_summary,
        'by_category': {'ticket': old_summary, 'hotel': old_summary},
    }
    new = {
        'generated_at': 'new',
        'summary': new_summary,
        'by_category': {'ticket': new_summary, 'hotel': old_summary, 'cruise': new_summary},
    }

    comparison = compare_reports(old, new)

    assert comparison['overall']['hit_at_k']['delta'] == 0.25
    assert comparison['overall']['mrr']['delta'] == 0.1
    assert comparison['overall']['citation_rate']['delta'] == 0.1
    assert comparison['overall']['refusal_accuracy']['delta'] == 0.0
    assert comparison['overall']['internal_leaks']['delta'] == -1.0
    assert comparison['by_category']['hotel']['mrr']['delta'] == 0.0
    assert comparison['by_category']['cruise']['hit_at_k']['old'] is None
    assert comparison['by_category']['cruise']['hit_at_k']['delta'] is None

    rendered = render_comparison(comparison)
    assert rendered.startswith('# 评测报告版本对比')
    assert '| hit_at_k | 0.5000 | 0.7500 | 0.2500 |' in rendered
