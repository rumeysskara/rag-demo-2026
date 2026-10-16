"""v2 – Better search: madde bazlı parçalar + hybrid arama (embedding + BM25, RRF) + reranker."""

from functools import lru_cache

import config
from pipelines.common import bm25_search, chat, rerank, reciprocal_rank_fusion, vector_search

PROMPT = """Aşağıdaki bilgilere göre soruyu cevapla. Kısa ve Türkçe cevap ver.

Bilgiler:
{context}

Soru: {question}"""


@lru_cache  # v3 aynı aramayı kullanır; aynı soru için reranker iki kez çalışmasın
def search(question):
    vector_results = vector_search(config.V2_COLLECTION, question, config.VECTOR_TOP_K)
    bm25_results = bm25_search(config.V2_COLLECTION, question, config.BM25_TOP_K)
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return rerank(question, fused[:config.RERANK_CANDIDATES], config.FINAL_TOP_K)


def answer(question):
    chunks = search(question)
    context = "\n\n".join(c["text"] for c in chunks)
    reply = chat(PROMPT.format(context=context, question=question))
    sources = sorted({f"{c['source']} {c['section']}" for c in chunks})
    return {"answer": reply, "chunks": chunks, "sources": sources}
