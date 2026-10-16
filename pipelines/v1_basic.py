"""v1 – Basic: sabit boyutlu parçalar + sadece embedding araması + düz prompt."""

import config
from pipelines.common import chat, vector_search

PROMPT = """Aşağıdaki bilgilere göre soruyu cevapla. Kısa ve Türkçe cevap ver.

Bilgiler:
{context}

Soru: {question}"""


def answer(question):
    chunks = vector_search(config.V1_COLLECTION, question, config.V1_TOP_K)
    context = "\n\n".join(c["text"] for c in chunks)
    reply = chat(PROMPT.format(context=context, question=question))
    sources = sorted({c["source"] for c in chunks})
    return {"answer": reply, "chunks": chunks, "sources": sources}
