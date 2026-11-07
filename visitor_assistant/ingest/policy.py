from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models import Audience


@dataclass(frozen=True)
class PolicyDecision:
    audience: Audience
    index: bool
    rule_id: str
    reason: str = ''


class ContentPolicy:
    def __init__(self, data: dict[str, Any]) -> None:
        self.version = int(data.get('version', 1))
        self.default_audience = Audience(str(data.get('default_audience', 'internal')))
        self.default_index = bool(data.get('default_index', True))
        self.rules = list(data.get('rules', []))
        self.redaction_rules = list(data.get('redaction_rules', []))
        for rule in self.redaction_rules:
            re.compile(str(rule['pattern']))

    @classmethod
    def from_file(cls, path: str | Path) -> 'ContentPolicy':
        with Path(path).open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError('内容策略必须是 YAML 对象')
        return cls(data)

    def classify(self, relative_path: str | Path) -> PolicyDecision:
        normalized = str(relative_path).replace('\\', '/').lstrip('./')
        for rule in self.rules:
            if self._matches(normalized, rule):
                return PolicyDecision(
                    audience=Audience(str(rule.get('audience', self.default_audience.value))),
                    index=bool(rule.get('index', True)),
                    rule_id=str(rule.get('id', '')),
                    reason=str(rule.get('reason', '')),
                )
        return PolicyDecision(self.default_audience, self.default_index, 'default')

    @staticmethod
    def _matches(path: str, rule: dict[str, Any]) -> bool:
        needle = str(rule.get('match', '')).replace('\\', '/')
        match_type = str(rule.get('match_type', 'contains'))
        if match_type == 'path':
            return path.casefold() == needle.casefold()
        if match_type == 'prefix':
            return path.casefold().startswith(needle.casefold())
        if match_type == 'suffix':
            return path.casefold().endswith(needle.casefold())
        if match_type == 'contains':
            return needle.casefold() in path.casefold()
        raise ValueError('未知 match_type: ' + match_type)

    def redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        redacted = text
        records: list[dict[str, Any]] = []
        for rule in self.redaction_rules:
            pattern = str(rule['pattern'])
            replacement = str(rule.get('replacement', ''))
            redacted, count = re.subn(pattern, replacement, redacted)
            if count:
                records.append({
                    'type': str(rule.get('type', 'unknown')),
                    'count': count,
                })
        return redacted, records
