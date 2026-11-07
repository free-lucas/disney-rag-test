from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_in_process(source: Path, ole_kind: str, destination: Path) -> None:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    application = None
    document = None
    try:
        if ole_kind == 'word':
            application = win32com.client.DispatchEx('Word.Application')
            application.Visible = False
            application.DisplayAlerts = 0
            document = application.Documents.Open(str(source), False, True, False)
            try:
                document.SaveAs(str(destination), FileFormat=16)
            except Exception:
                document.SaveAs2(str(destination), FileFormat=16)
        elif ole_kind == 'powerpoint':
            application = win32com.client.DispatchEx('PowerPoint.Application')
            document = application.Presentations.Open(str(source), True, False, False)
            document.SaveAs(str(destination), 24)
        else:
            raise ValueError('不支持的旧版 Office 类型: ' + ole_kind)
    finally:
        if document is not None:
            try:
                document.Close(False)
            except Exception:
                pass
        if application is not None:
            try:
                application.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--kind', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    try:
        convert_in_process(Path(args.source), args.kind, Path(args.destination))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error_type': 'com_failed', 'message': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({'ok': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
