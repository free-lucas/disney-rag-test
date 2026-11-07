from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


TOKENIZER_VERSION = 'jieba-v1'

STOPWORDS = {
    '的', '了', '和', '与', '及', '或', '是', '在', '有', '我', '你', '他', '她', '它',
    '我们', '你们', '他们', '这', '那', '这个', '那个', '请问', '一下', '怎么', '如何',
    '什么', '哪些', '可以', '能否', '需要', '关于', '以及', '还有', '吗', '呢', '吧',
    '啊', '呀', '请', '想', '要', '去', '到', '在', '里', '中', '为', '对', '从',
}


def tokenize(text: str) -> list[str]:
    import jieba

    jieba.setLogLevel(logging.ERROR)
    normalized = str(text).lower()
    tokens: list[str] = []
    for raw in jieba.lcut(normalized):
        token = raw.strip()
        if not token or token in STOPWORDS:
            continue
        if re.fullmatch(r'[\W_]+', token):
            continue
        tokens.append(token)
    return tokens


def tokenize_many(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]


def _token_matches(query_token: str, document_tokens: Sequence[str]) -> bool:
    for document_token in document_tokens:
        if not document_token:
            continue
        if query_token in document_token or document_token in query_token:
            return True
    return False


class BM25Index:
    def __init__(self, tokens: Sequence[Sequence[str]]) -> None:
        from rank_bm25 import BM25Plus

        self.tokens = [list(item) for item in tokens]
        self._bm25 = BM25Plus(self.tokens) if self.tokens else None

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> 'BM25Index':
        return cls(tokenize_many(texts))

    def subset(self, indices: Sequence[int]) -> 'BM25Index':
        return BM25Index([self.tokens[index] for index in indices])

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if self._bm25 is None or k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        significant = [token for token in query_tokens if len(token) >= 2] or query_tokens
        results: list[tuple[int, float]] = []
        for index, score in ranked:
            if float(score) <= 0:
                continue
            if not any(_token_matches(token, self.tokens[index]) for token in significant):
                continue
            results.append((index, float(score)))
            if len(results) >= k:
                break
        return results

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': 1,
            'tokenizer': TOKENIZER_VERSION,
            'tokens': self.tokens,
        }
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(payload, handle)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> 'BM25Index':
        with Path(path).open('rb') as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or 'tokens' not in payload:
            raise ValueError('BM25 索引格式非法')
        return cls(payload['tokens'])


@dataclass
class BM25SearchResult:
    index: int
    score: float = field(default=0.0)
