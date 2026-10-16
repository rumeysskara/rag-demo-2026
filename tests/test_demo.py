"""Sahnedeki beklenen sonuçları kontrol eder. Ollama açık ve indeks kurulu olmalı.

Çalıştırma:  pytest -s tests/test_demo.py
Beklenen sonuç çıkmazsa getirilen parçalar yazdırılır; veriyi veya config.V1_CHUNK_WORDS'ü ayarlayın.
"""

import ollama
import pytest

import config


def _ready():
    try:
        ollama.list()
    except Exception:
        return "Ollama çalışmıyor (ollama serve)"
    if not config.CHROMA_DIR.exists():
        return "İndeks yok (python build_index.py)"
    return None


pytestmark = pytest.mark.skipif(_ready() is not None, reason=_ready() or "")

from pipelines import v1_basic, v2_better, v3_sources  # noqa: E402

Q_REMOTE = "Onay almadan evden çalışabilir miyim?"
Q_UNKNOWN = "Şirketin hisse fiyatı ne?"


def explain(name, question, result):
    lines = [f"\n{name} | Soru: {question}", f"Cevap: {result['answer']}", "Getirilen parçalar:"]
    for chunk in result["chunks"]:
        lines.append(f"  - [{chunk['source']} {chunk['section']}] {chunk['text'][:200]!r}")
    report = "\n".join(lines)
    print(report)
    return report


def normalize(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def test_v1_misses_approval_requirement():
    result = v1_basic.answer(Q_REMOTE)
    answer = normalize(result["answer"])
    assert "evet" in answer and "onay" not in answer, explain("v1", Q_REMOTE, result)


@pytest.mark.parametrize("name,system", [("v2", v2_better), ("v3", v3_sources)])
def test_better_systems_require_approval(name, system):
    result = system.answer(Q_REMOTE)
    answer = normalize(result["answer"])
    assert "onay" in answer and "evet" not in answer, explain(name, Q_REMOTE, result)


def test_v3_says_i_dont_know():
    result = v3_sources.answer(Q_UNKNOWN)
    assert "bilmiyorum" in normalize(result["answer"]), explain("v3", Q_UNKNOWN, result)
