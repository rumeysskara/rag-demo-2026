"""Model gerektirmeyen testler: veri ve bölme mantığı demo senaryosuna uygun mu?"""

from pipelines.common import chunk_by_section, chunk_fixed, read_documents

POLICY = read_documents()["hr_policy.md"]
FAQ = read_documents()["hr_faq.md"]

REMOTE = "§4.2 Çalışanlar haftada 3 gün evden çalışabilir, ancak bunun için yöneticisinin yazılı onayı gerekir."
CARRYOVER = "§6.1 Kullanılmayan yıllık iznin en fazla 5 günü sonraki yıla devreder."


def test_required_sentences_exist():
    assert REMOTE in POLICY
    assert CARRYOVER in POLICY
    assert "Başka şehirden çalışacaksan İK'ya haber ver." in FAQ
    assert "Devreden izin Mart sonuna kadar kullanılmalı." in FAQ


def test_v1_splits_remote_work_rule():
    """v1'de 'evden çalışabilir' ile 'yöneticisinin yazılı onayı gerekir' FARKLI parçalara düşmeli."""
    chunks = chunk_fixed(POLICY)
    permission = [i for i, c in enumerate(chunks) if "evden çalışabilir" in c and "§4.2" in c]
    approval = [i for i, c in enumerate(chunks) if "yöneticisinin yazılı onayı gerekir" in c]
    assert len(permission) == 1 and len(approval) == 1
    assert permission[0] != approval[0], "§4.2 tek parçada kaldı; V1_CHUNK_WORDS veya §4.1 metnini ayarlayın"


def test_v2_keeps_remote_work_rule_together():
    chunks = chunk_by_section("hr_policy.md", POLICY)
    rule = [c for c in chunks if c["section"] == "§4.2"]
    assert len(rule) == 1
    assert "evden çalışabilir" in rule[0]["text"] and "yazılı onayı gerekir" in rule[0]["text"]
    assert rule[0]["text"].startswith("Bu parça")  # bağlam satırı
    assert "Uzaktan Çalışma" in rule[0]["text"].splitlines()[0]
    assert rule[0]["updated_at"] == "2026-01-15"


def test_v2_faq_chunks_keep_question_and_answer():
    chunks = chunk_by_section("hr_faq.md", FAQ)
    city = [c for c in chunks if "başka bir şehirden" in c["text"]]
    assert len(city) == 1 and "İK'ya haber ver" in city[0]["text"]
