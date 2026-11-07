import pytest

from config import ConfigurationError, Settings, load_settings


def test_settings_env_override():
    env = {
        'DEEPSEEK_API_KEY': 'deepseek-test-value',
        'DASHSCOPE_API_KEY': 'dashscope-test-value',
        'DEEPSEEK_MODEL': 'deepseek-flash',
        'EMBEDDING_MODEL': 'text-embedding-v4',
        'EMBEDDING_FALLBACK_MODEL': 'text-embedding-v1',
        'DISNEY_KB_ROOT': 'fixtures/kb',
        'RAG_DATA_DIR': 'fixtures/data',
        'RAG_INDEX_DIR': 'fixtures/index',
        'RETRIEVAL_TOP_K': '9',
        'DENSE_TOP_K': '11',
        'BM25_TOP_K': '13',
        'RERANK_ENABLED': 'yes',
        'CHUNK_TARGET_CHARS': '550',
        'CHUNK_MIN_CHARS': '400',
        'CHUNK_MAX_CHARS': '600',
        'CHUNK_OVERLAP_CHARS': '100',
    }
    settings = Settings.from_env(env, load_env_file=False)
    assert settings.deepseek_model == 'deepseek-flash'
    assert settings.retrieval_top_k == 9
    assert settings.dense_top_k == 11
    assert settings.bm25_top_k == 13
    assert settings.rerank_enabled is True
    assert settings.index_dir.name == 'index'
    assert settings.chunk_target_chars == 550


def test_missing_keys_error_does_not_echo_values():
    env = {'DEEPSEEK_API_KEY': 'secret-deepseek-value'}
    with pytest.raises(ConfigurationError) as captured:
        load_settings(env, load_env_file=False, require_api_keys=True)
    message = str(captured.value)
    assert 'DASHSCOPE_API_KEY' in message
    assert 'secret-deepseek-value' not in message


def test_repr_and_summary_hide_key_values():
    settings = Settings.from_env({'DEEPSEEK_API_KEY': 'secret-a', 'DASHSCOPE_API_KEY': 'secret-b'}, load_env_file=False)
    assert 'secret-a' not in repr(settings)
    assert 'secret-b' not in repr(settings)
    summary = settings.public_summary()
    assert summary['deepseek_api_key_present'] is True
    assert 'secret-a' not in str(summary)


def test_invalid_chunk_parameters_rejected():
    with pytest.raises(ConfigurationError):
        Settings.from_env({'CHUNK_TARGET_CHARS': '700'}, load_env_file=False)
