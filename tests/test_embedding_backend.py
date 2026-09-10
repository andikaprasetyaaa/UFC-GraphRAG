from src.embeddings import resolve_embedding_backend


def test_resolve_embedding_backend_defaults_to_sentence_transformers():
    assert resolve_embedding_backend("") == "sentence_transformers"
    assert resolve_embedding_backend("gemini") == "sentence_transformers"
    assert resolve_embedding_backend("GEMINI") == "sentence_transformers"


def test_resolve_embedding_backend_supports_sentence_transformers():
    assert resolve_embedding_backend("sentence_transformers") == "sentence_transformers"
    assert resolve_embedding_backend("sentence-transformers") == "sentence_transformers"
    assert resolve_embedding_backend("SentenceTransformer") == "sentence_transformers"
