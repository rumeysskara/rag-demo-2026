"""v3 – With sources: v2'nin araması + numaralı kaynaklar + 'Bilmiyorum' talimatı."""

from pipelines.common import chat
from pipelines.v2_better import search

SYSTEM = ("Sadece bu kaynaklara bakarak cevap ver. "
          "Cevap yoksa tam olarak şunu yaz: 'Bilmiyorum, bu bilgi kaynaklar arasında yer almıyor.' "
          "Her bilginin yanına kaynağını [1] gibi yaz. Kısa ve Türkçe cevap ver.")

PROMPT = """Kaynaklar:
{context}

Soru: {question}"""


def answer(question):
    chunks = search(question)
    sources = [f"[{i}] {c['source']} {c['section']} (güncelleme: {c['updated_at']})"
               for i, c in enumerate(chunks, start=1)]
    context = "\n\n".join(f"[{i}] (Kaynak: {c['source']} {c['section']})\n{c['text']}"
                          for i, c in enumerate(chunks, start=1))
    reply = chat(PROMPT.format(context=context, question=question), system_prompt=SYSTEM)
    return {"answer": reply, "chunks": chunks, "sources": sources}
