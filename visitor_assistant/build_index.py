from __future__ import annotations

import argparse
import sys

from config import ConfigurationError, load_settings
from ingest.build import BuildError, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='构建迪士尼游客 RAG 索引')
    parser.add_argument('--root', default=None, help='知识库根目录')
    parser.add_argument('--index-dir', default=None, help='索引输出目录')
    parser.add_argument('--report', default=None, help='建库报告输出路径')
    parser.add_argument('--no-com', action='store_true', help='禁用 Windows Office COM 转换')
    parser.add_argument('--force-rebuild', action='store_true', help='忽略文档级缓存并重新抽取全部文件')
    args = parser.parse_args(argv)

    try:
        settings = load_settings(require_api_keys=False)
        report = build_index(
            settings,
            root=args.root,
            index_dir=args.index_dir,
            report_path=args.report,
            allow_com=not args.no_com,
            force_rebuild=args.force_rebuild,
        )
    except (ConfigurationError, BuildError) as exc:
        print('建库失败: ' + str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print('建库失败: ' + exc.__class__.__name__ + ': ' + str(exc), file=sys.stderr)
        return 1

    summary = report.summary()
    print('建库完成: ' + report.build_version)
    print('  embedding: ' + report.embedding_model + ' / ' + str(report.embedding_dim) + ' 维')
    for key, value in summary.items():
        print('  ' + key + ': ' + str(value))
    if report.failures:
        print('失败文件:')
        for failure in report.failures:
            print('  - ' + failure.source_file + ' [' + failure.error_type + '] ' + failure.error_message)
    print('报告: ' + report.artifacts.get('build_report', ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
