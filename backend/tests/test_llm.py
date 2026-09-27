from app.llm.embeddings import _mock_embed, cosine
from app.llm.provider import parse_json


def test_parse_json_plain_and_fenced():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('설명\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('앞말 {"b": "x"} 뒷말') == {"b": "x"}


def test_mock_embedding_similarity():
    a = _mock_embed("백엔드 API 서버 개발", 384)
    b = _mock_embed("백엔드 API 개발", 384)
    c = _mock_embed("인스타그램 광고 캠페인", 384)
    assert cosine(a, b) > cosine(a, c)
