from __future__ import annotations

from pathlib import Path

import streamlit as st

import app_service
from models import Answer


st.set_page_config(page_title='迪士尼游客 RAG 助手', page_icon='🏰', layout='centered')


def _reset_conversation() -> None:
    st.session_state['messages'] = []
    st.session_state['history'] = []


def _ensure_state() -> None:
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('history', [])


def _render_sources(answer: Answer) -> None:
    if answer.refused or not answer.citations:
        return
    with st.expander('来源（' + str(len(answer.citations)) + '）', expanded=False):
        for citation in answer.citations:
            st.markdown('- ' + citation.label)


def _render_images(answer: Answer) -> None:
    if answer.refused:
        return
    for image in answer.images:
        path = Path(image.image_path)
        if path.is_file():
            st.image(str(path), caption=image.caption or image.image_path)
        else:
            st.caption('图片暂不可用：' + image.image_path)


def _render_extras(answer: Answer) -> None:
    _render_sources(answer)
    _render_images(answer)


def _render_answer_text(answer: Answer) -> None:
    st.markdown(answer.text)
    if answer.error:
        st.warning('回答生成暂时不可用，已返回安全提示。请稍后重试。')
    _render_extras(answer)


def _append_message(role: str, content: str, answer: Answer | None = None) -> None:
    message: dict[str, object] = {'role': role, 'content': content}
    if answer is not None:
        message['answer'] = answer.to_dict()
    st.session_state['messages'].append(message)


def _handle_question(question: str) -> None:
    _append_message('user', question)
    history = list(st.session_state['history'])[-6:]
    st.session_state['history'].append({'role': 'user', 'content': question})
    st.session_state['history'] = st.session_state['history'][-12:]

    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        try:
            service = app_service.get_service()
        except app_service.ServiceError as exc:
            message = str(exc)
            st.error(message)
            _append_message('assistant', message)
            return

        try:
            stream = service.stream(question, history)
        except app_service.ServiceError as exc:
            message = str(exc)
            st.error(message)
            _append_message('assistant', message)
            return

        placeholder = st.empty()
        buffer = ''
        answer: Answer
        try:
            for piece in stream:
                buffer += piece
                placeholder.markdown(buffer)
            answer = stream.finalize()
        except Exception:
            answer = Answer(
                text='抱歉，回答生成暂时不可用，请稍后重试。',
                refused=True,
                query=question,
                rewritten_query=question,
                error='stream_failed',
            )

        if answer.text.strip() != buffer.strip():
            placeholder.markdown(answer.text)
        if answer.error:
            st.warning('回答生成暂时不可用，已返回安全提示。请稍后重试。')
        _render_extras(answer)
        _append_message('assistant', answer.text, answer=answer)
        st.session_state['history'].append({'role': 'assistant', 'content': answer.text})
        st.session_state['history'] = st.session_state['history'][-12:]


_ensure_state()

with st.sidebar:
    st.header('对话')
    if st.button('新对话', use_container_width=True):
        _reset_conversation()
    st.divider()
    st.caption('回答只使用知识库公开内容，并标注来源。')
    try:
        configuration = app_service.public_config()
        st.caption('生成模型：' + str(configuration.get('deepseek_model', '')))
        st.caption('索引目录：' + str(configuration.get('index_dir', '')))
    except app_service.ServiceError as exc:
        st.caption(str(exc))

st.title('🏰 迪士尼游客 RAG 助手')
st.caption('门票、酒店、乐园项目、餐饮、邮轮和游客安全指引。涉及票价与时刻请以官方渠道当日信息为准。')

for message in st.session_state['messages']:
    with st.chat_message(str(message['role'])):
        st.markdown(str(message['content']))
        stored = message.get('answer')
        if isinstance(stored, dict):
            _render_extras(Answer.from_dict(stored))

question = st.chat_input('输入你的问题，例如：一日票多少钱？')
if question:
    _handle_question(str(question).strip())
