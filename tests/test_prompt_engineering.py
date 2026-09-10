from src.rag_pipeline import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def test_system_prompt_has_strict_grounding_and_no_hallucination_rules():
    text = SYSTEM_PROMPT.lower()
    assert "hanya berdasarkan konteks" in text
    assert "jangan mengarang" in text or "no hallucination" in text
    assert "fakta graph eksak" in text
    assert "jawaban singkat" in text or "ringkas" in text


def test_user_prompt_sets_answer_contract():
    text = USER_PROMPT_TEMPLATE.lower()
    assert "{context}" in USER_PROMPT_TEMPLATE
    assert "pertanyaan:" in text
    assert "jawaban:" in text
    assert "dengan kalimat yang jelas" in text or "jelas" in text
