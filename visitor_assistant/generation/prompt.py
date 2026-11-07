from __future__ import annotations

import re

from models import Audience, ChunkType, RetrievalResult


DECISION_ANSWER = 'ANSWER'
DECISION_REFUSE = 'REFUSE'

REFUSAL_MESSAGE = (
    '抱歉，公开知识库中没有足够信息回答这个问题。'
    '建议你查看上海迪士尼度假区官方网站或官方 App 的最新公告，也可以联系官方客服获取帮助。'
)

CONFIDENTIALITY_REFUSAL_MESSAGE = (
    '这个问题涉及内部信息，无法向游客提供。'
    '如需帮助，请通过上海迪士尼度假区官方网站或官方 App 联系官方客服。'
)

PARTIAL_EVIDENCE_NOTE = (
    '知识库暂未覆盖上述未提及的部分，请以上海迪士尼度假区官方网站或官方 App 的信息为准。'
)

TIME_SENSITIVE_DISCLAIMER = (
    '票价、开闭园时间、演出时刻和相关政策可能调整，'
    '请以上海迪士尼度假区官方网站或官方 App 的当日信息为准。'
)

ANSWER_SYSTEM_PROMPT = '''你是上海迪士尼度假区的游客服务助手，只面向游客回答。

必须遵守以下规则：
1. 只使用“公开证据”中的信息回答，不使用未提供的常识补全事实，不编造价格、时间、地点、政策或联系方式。
2. 每个事实性陈述都要标注证据编号，例如 [1]、[2]，编号必须与公开证据一致。
3. 与上海迪士尼度假区无关的域外问题（例如实时天气、新闻、股价、通用知识），即使证据中出现个别相关词，也必须第一行输出 REFUSE，不要猜测，也不要改用内部资料。
4. 除第 3 条外，只有当公开证据与问题完全无关、无法提供任何可确认信息时，才输出 REFUSE。
5. 只要证据中有与问题直接相关的内容，就必须回答能够确认的部分，再明确说明“知识库未覆盖……”并引导游客查看官方网站或官方 App；不要因为证据不完整、缺少细节或无法覆盖所有方面而整体拒答。
6. 不得输出员工规则、员工票、工号、员工价、内部流程、升级路径、内部电话、分机、责任人或员工联系方式。
7. 涉及票价、开闭园时间、演出时刻、预约、退改等易变信息时，提醒游客以官方渠道当日信息为准。
8. 结合最近对话理解追问；如果本轮是新的独立主题，不要沿用无关的旧话题。
9. 使用简体中文，语气友好、简洁，先给结论再给必要说明。
10. 输出格式：第一行只写 ANSWER 或 REFUSE；从第二行开始写面向游客的正文。REFUSE 时不要写正文。'''


def format_history(history: list[dict[str, str]] | None, *, limit: int = 6) -> str:
    if not history:
        return '（无历史）'
    lines: list[str] = []
    for message in history[-limit:]:
        role = '游客' if str(message.get('role', '')) == 'user' else '助手'
        content = str(message.get('content', '')).strip()
        if content:
            lines.append(role + '：' + content[:600])
    return '\n'.join(lines) or '（无历史）'


def evidence_text(result: RetrievalResult) -> str:
    chunk = result.chunk
    if chunk.chunk_type is ChunkType.IMAGE_CAPTION:
        body = chunk.caption or chunk.content
        kind = '图片描述'
    elif chunk.chunk_type is ChunkType.TABLE:
        body = chunk.content
        kind = '表格'
    else:
        body = chunk.content
        kind = '正文'
    return body.strip() + '\n（类型：' + kind + '）'


def format_evidence(results: list[RetrievalResult]) -> str:
    lines: list[str] = []
    number = 0
    for result in results:
        if result.chunk.audience is not Audience.PUBLIC:
            continue
        number += 1
        lines.append(
            '[' + str(number) + '] 文件：' + result.chunk.source_file
            + ' | 位置：' + result.chunk.location
        )
        lines.append(evidence_text(result))
        lines.append('')
    return '\n'.join(lines).strip() or '（无公开证据）'


def public_evidence(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return [result for result in results if result.chunk.audience is Audience.PUBLIC]


def build_messages(
    question: str,
    results: list[RetrievalResult],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    user_prompt = (
        '游客问题：' + str(question).strip() + '\n\n'
        + '最近对话：\n' + format_history(history) + '\n\n'
        + '公开证据：\n' + format_evidence(results) + '\n\n'
        + '请严格按系统要求的格式回答。'
    )
    return [
        {'role': 'system', 'content': ANSWER_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


TIME_SENSITIVE_TERMS = (
    '票价', '门票', '价格', '多少钱', '费用', '收费',
    '开园', '闭园', '开放时间', '营业时间', '几点',
    '演出', '巡游', '烟花', '时刻表', '节目表',
    '预约', '退改', '改签', '退款', '限流', '入园规定', '政策',
)


def is_time_sensitive(text: str) -> bool:
    normalized = str(text or '')
    return any(term in normalized for term in TIME_SENSITIVE_TERMS)


INTERNAL_LEAK_TERMS = (
    '员工票', '员工价', '工号', '内部员工', '仅限员工', '员工专用',
    '内部流程', '内部规定', '升级路径', '上报路径', '逐级上报',
    '内部电话', '内线', '分机', '员工热线', '内部联系人',
    '责任人', '值班经理', '员工联系方式', '内部通讯录', '企业微信', '钉钉群',
)

INTERNAL_LEAK_PATTERNS = (
    re.compile(r'(?:转|分机)\s*\d{1,4}'),
    re.compile(r'(?:内部|员工)(?:电话|热线)\s*[：: ]*\d[\d\- ]{5,}'),
)


def find_internal_leak(text: str) -> str | None:
    normalized = str(text or '')
    for term in INTERNAL_LEAK_TERMS:
        if term in normalized:
            return term
    for pattern in INTERNAL_LEAK_PATTERNS:
        if pattern.search(normalized):
            return pattern.pattern
    return None


ANSWER_MARKER_PATTERN = re.compile(r'\[(\d{1,2})\]')


def cited_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in ANSWER_MARKER_PATTERN.findall(str(text or '')):
        value = int(match)
        if value not in numbers:
            numbers.append(value)
    return numbers
