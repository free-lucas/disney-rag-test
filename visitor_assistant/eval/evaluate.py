from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.golden import GoldenQuestion, load_golden_questions
from eval.metrics import aggregate_by_category, aggregate_records, collect_misses, score_retrieval
from eval.quality import evaluate_answer
from models import Answer


def _answer_record(answer: Answer) -> dict[str, object]:
    return {
        'text': answer.text,
        'refused': answer.refused,
        'disclaimer': answer.disclaimer,
        'citation_count': len(answer.citations),
        'citations': [citation.label for citation in answer.citations],
        'image_count': len(answer.images),
        'error': answer.error,
    }


def run_evaluation(
    questions: list[GoldenQuestion],
    retriever: object,
    generator: object | None,
    *,
    top_k: int = 6,
    with_generation: bool = True,
    model: str = '',
    internal_texts: list[str] | None = None,
    generated_at: str | None = None,
    progress: object | None = None,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for index, question in enumerate(questions, start=1):
        record: dict[str, object] = {
            'id': question.id,
            'category': question.category,
            'question': question.question,
            'expect_refusal': question.expect_refusal,
            'retrieval': None,
            'answer': None,
            'checks': None,
            'error': None,
        }
        try:
            bundle = retriever.retrieve(question.question, [])
            results = list(getattr(bundle, 'results', []))
            rewrite = getattr(bundle, 'rewrite', None)
            rewritten_query = getattr(rewrite, 'rewritten_query', '') if rewrite is not None else ''
        except Exception as exc:
            message = type(exc).__name__ + ': ' + str(exc)
            record['retrieval'] = {'error': message}
            record['error'] = message
            records.append(record)
            errors.append({'id': question.id, 'stage': 'retrieval', 'error': message})
            continue

        record['retrieval'] = score_retrieval(
            results,
            question.expected_source_files,
            k=top_k,
        )

        if with_generation and generator is not None:
            try:
                answer = generator.generate(
                    question.question,
                    results,
                    [],
                    rewritten_query=rewritten_query,
                )
                record['answer'] = _answer_record(answer)
                record['checks'] = evaluate_answer(
                    answer,
                    expect_refusal=question.expect_refusal,
                    internal_texts=internal_texts,
                )
                if answer.error:
                    errors.append({'id': question.id, 'stage': 'generation', 'error': answer.error})
            except Exception as exc:
                message = type(exc).__name__ + ': ' + str(exc)
                record['answer'] = {'error': message}
                record['checks'] = {
                    'citation_present': False,
                    'refusal_correct': False,
                    'internal_leak': False,
                    'leak_detail': None,
                    'passed': False,
                }
                record['error'] = message
                errors.append({'id': question.id, 'stage': 'generation', 'error': message})

        records.append(record)
        if callable(progress):
            progress(index, len(questions), record)

    summary = aggregate_records(records, with_generation=with_generation)
    summary['k'] = top_k
    return {
        'version': 1,
        'generated_at': generated_at or datetime.now(timezone.utc).isoformat(),
        'model': model,
        'top_k': top_k,
        'with_generation': with_generation,
        'summary': summary,
        'by_category': aggregate_by_category(records, with_generation=with_generation),
        'questions': records,
        'misses': collect_misses(records),
        'errors': errors,
    }


def _format_metric(value: object) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def render_markdown(report: dict[str, object]) -> str:
    summary = report.get('summary', {})
    lines = [
        '# 迪士尼游客 RAG 评测报告',
        '',
        '- 生成时间：' + str(report.get('generated_at', '')),
        '- 模型：' + str(report.get('model', '')),
        '- TopK：' + str(report.get('top_k', '')),
        '- 是否运行生成：' + ('是' if report.get('with_generation') else '否'),
        '',
        '## 总览',
        '',
        '| 指标 | 值 |',
        '|---|---:|',
        '| 题目数 | ' + _format_metric(summary.get('total')) + ' |',
        '| hit@k | ' + _format_metric(summary.get('hit_at_k')) + ' |',
        '| MRR | ' + _format_metric(summary.get('mrr')) + ' |',
        '| 引用率 | ' + _format_metric(summary.get('citation_rate')) + ' |',
        '| 拒答正确率 | ' + _format_metric(summary.get('refusal_accuracy')) + ' |',
        '| 内部泄露数 | ' + _format_metric(summary.get('internal_leaks')) + ' |',
        '| 通过题数 | ' + _format_metric(summary.get('passed')) + ' |',
        '| 错误数 | ' + _format_metric(summary.get('errors')) + ' |',
        '',
        '## 分类指标',
        '',
        '| 类别 | 题数 | hit@k | MRR | 引用率 | 拒答正确率 | 泄露 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for category, metrics in report.get('by_category', {}).items():
        lines.append(
            '| ' + category + ' | ' + _format_metric(metrics.get('total'))
            + ' | ' + _format_metric(metrics.get('hit_at_k'))
            + ' | ' + _format_metric(metrics.get('mrr'))
            + ' | ' + _format_metric(metrics.get('citation_rate'))
            + ' | ' + _format_metric(metrics.get('refusal_accuracy'))
            + ' | ' + _format_metric(metrics.get('internal_leaks')) + ' |'
        )

    misses = report.get('misses', [])
    lines.extend(['', '## 未命中题目', ''])
    if not misses:
        lines.append('无。')
    else:
        for miss in misses:
            lines.append('### ' + str(miss.get('id')) + ' ' + str(miss.get('question')))
            lines.append('')
            lines.append('- 期望来源：' + '；'.join(miss.get('expected', [])))
            actual = '；'.join(
                str(item.get('source_file')) + ' (' + str(item.get('score')) + ')'
                for item in miss.get('top_k', [])
            )
            lines.append('- 实际 TopK：' + (actual or '无'))
            lines.append('')

    errors = report.get('errors', [])
    lines.extend(['', '## 失败记录', ''])
    if not errors:
        lines.append('无。')
    else:
        for error in errors:
            lines.append('- ' + str(error.get('id')) + ' [' + str(error.get('stage')) + ']：' + str(error.get('error')))
    lines.append('')
    return '\n'.join(lines)


def write_reports(report: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get('generated_at', 'report')).replace(':', '-').replace('+', '_')
    json_path = destination / ('eval_report_' + stamp + '.json')
    markdown_path = destination / ('eval_report_' + stamp + '.md')
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    markdown_path.write_text(render_markdown(report), encoding='utf-8')
    return json_path, markdown_path


COMPARE_METRICS = ('hit_at_k', 'mrr', 'citation_rate', 'refusal_accuracy', 'internal_leaks')


def _delta(old_value: object, new_value: object) -> float | None:
    if old_value is None or new_value is None:
        return None
    return round(float(new_value) - float(old_value), 4)


def compare_reports(old: dict[str, object], new: dict[str, object]) -> dict[str, object]:
    old_summary = old.get('summary', {})
    new_summary = new.get('summary', {})
    overall = {}
    for metric in COMPARE_METRICS:
        old_value = old_summary.get(metric)
        new_value = new_summary.get(metric)
        overall[metric] = {
            'old': old_value,
            'new': new_value,
            'delta': _delta(old_value, new_value),
        }

    old_categories = old.get('by_category', {})
    new_categories = new.get('by_category', {})
    categories = sorted(set(old_categories) | set(new_categories))
    by_category = {}
    for category in categories:
        old_metrics = old_categories.get(category, {})
        new_metrics = new_categories.get(category, {})
        by_category[category] = {
            metric: {
                'old': old_metrics.get(metric),
                'new': new_metrics.get(metric),
                'delta': _delta(old_metrics.get(metric), new_metrics.get(metric)),
            }
            for metric in COMPARE_METRICS
        }
    return {
        'old_generated_at': old.get('generated_at'),
        'new_generated_at': new.get('generated_at'),
        'overall': overall,
        'by_category': by_category,
    }


def render_comparison(comparison: dict[str, object]) -> str:
    lines = [
        '# 评测报告版本对比',
        '',
        '- 旧报告：' + str(comparison.get('old_generated_at', '')),
        '- 新报告：' + str(comparison.get('new_generated_at', '')),
        '',
        '| 指标 | 旧值 | 新值 | 差值 |',
        '|---|---:|---:|---:|',
    ]
    for metric, values in comparison.get('overall', {}).items():
        lines.append(
            '| ' + metric + ' | ' + _format_metric(values.get('old'))
            + ' | ' + _format_metric(values.get('new'))
            + ' | ' + _format_metric(values.get('delta')) + ' |'
        )
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='迪士尼游客 RAG 质量评测')
    parser.add_argument('--golden', default='eval/golden_questions.json', help='金标准问题 JSON')
    parser.add_argument('--output-dir', default='data/reports', help='报告输出目录')
    parser.add_argument('--top-k', type=int, default=6, help='检索评估 TopK')
    parser.add_argument('--limit', type=int, default=0, help='只运行前 N 题，0 表示全部')
    parser.add_argument('--no-generation', action='store_true', help='只评测检索，不调用生成模型')
    parser.add_argument('--compare', nargs=2, metavar=('OLD', 'NEW'), help='对比两份 JSON 报告')

    args = parser.parse_args(argv)

    if args.compare:
        old = json.loads(Path(args.compare[0]).read_text(encoding='utf-8'))
        new = json.loads(Path(args.compare[1]).read_text(encoding='utf-8'))
        comparison = compare_reports(old, new)
        print(render_comparison(comparison))
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'eval_comparison.md').write_text(render_comparison(comparison), encoding='utf-8')
        return 0

    try:
        questions = load_golden_questions(args.golden)
    except Exception as exc:
        print('金标准加载失败: ' + str(exc))
        return 2
    if args.limit > 0:
        questions = questions[:args.limit]

    try:
        from app_service import build_service

        service = build_service()
    except Exception as exc:
        print('服务初始化失败: ' + str(exc))
        return 2

    store = service.retriever.store
    internal_texts = [chunk.content for chunk in store.internal_chunks]
    report = run_evaluation(
        questions,
        service.retriever,
        None if args.no_generation else service.generator,
        top_k=args.top_k,
        with_generation=not args.no_generation,
        model=service.settings.deepseek_model,
        internal_texts=internal_texts,
        progress=lambda index, total, record: print(
            '[' + str(index) + '/' + str(total) + '] ' + str(record.get('id'))
            + (' ERROR' if record.get('error') else '')
        ),
    )
    json_path, markdown_path = write_reports(report, args.output_dir)
    summary = report['summary']
    print('评测完成')
    print('  hit@k: ' + _format_metric(summary.get('hit_at_k')))
    print('  MRR: ' + _format_metric(summary.get('mrr')))
    print('  引用率: ' + _format_metric(summary.get('citation_rate')))
    print('  拒答正确率: ' + _format_metric(summary.get('refusal_accuracy')))
    print('  内部泄露: ' + _format_metric(summary.get('internal_leaks')))
    print('  JSON: ' + str(json_path))
    print('  Markdown: ' + str(markdown_path))
    return 0
