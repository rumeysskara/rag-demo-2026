"""Üç sistemin ortak parçaları: dokümanları okuma, bölme, embedding, arama, reranker, LLM çağrısı."""

import json
import re
from functools import lru_cache

import chromadb
import ollama
from rank_bm25 import BM25Okapi

import config


# ---------------------------------------------------------------------------
# Dokümanlar
# ---------------------------------------------------------------------------

def read_documents():
    """data/ klasöründeki .md dosyalarını {dosya_adı: metin} olarak döndürür."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(config.DATA_DIR.glob("*.md"))}


def get_updated_at(text):
    """Doküman içindeki 'Son güncelleme: YYYY-AA-GG' satırını bulur."""
    match = re.search(r"Son güncelleme:\s*(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else "bilinmiyor"


# ---------------------------------------------------------------------------
# Bölme (chunking)
# ---------------------------------------------------------------------------

def chunk_fixed(text, words_per_chunk=config.V1_CHUNK_WORDS):
    """v1: Metni anlamına bakmadan, sabit kelime sayısına göre böler (overlap yok)."""
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def chunk_by_section(source, text):
    """v2/v3: Markdown başlıklarına ve madde numaralarına göre böler.

    - '§4.2 ...' ile başlayan her paragraf tek bir parçadır.
    - '### Soru' başlığı ve altındaki cevap tek bir parçadır (SSS).
    - Her parçaya metadata ve kısa bir bağlam satırı eklenir.
    """
    doc_title = ""
    section_title = ""
    updated_at = get_updated_at(text)
    chunks = []
    current = None

    def close():
        if current and current["body"]:
            chunks.append(current)

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("Son güncelleme:"):
            continue

        if paragraph.startswith("# "):
            doc_title = paragraph[2:].strip()
        elif paragraph.startswith("## "):
            close()
            current = None
            section_title = paragraph[3:].strip()
        elif paragraph.startswith("### "):  # SSS sorusu
            close()
            question = paragraph[4:].strip()
            current = {"section": section_title, "body": [question]}
        elif paragraph.startswith("§"):  # madde numarası
            close()
            number = paragraph.split()[0]
            current = {"section": number, "body": [paragraph]}
        elif current is not None:  # önceki parçanın devamı
            current["body"].append(paragraph)
        else:  # bölüm giriş paragrafı
            current = {"section": section_title, "body": [paragraph]}
    close()

    result = []
    for chunk in chunks:
        context = f"Bu parça '{doc_title}' dokümanının '{_heading_for(text, chunk['section'])}' bölümünden."
        body = "\n".join(chunk["body"])
        result.append({
            "text": f"{context}\n{body}",
            "source": source,
            "section": chunk["section"],
            "title": doc_title,
            "updated_at": updated_at,
        })
    return result


def _heading_for(text, section):
    """'§4.2' gibi bir madde numarası için ait olduğu '## 4. ...' başlığını bulur."""
    if not section.startswith("§"):
        return section
    major = section[1:].split(".")[0]
    match = re.search(rf"^## ({major}\..*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else section


# ---------------------------------------------------------------------------
# Embedding (EmbeddingGemma 2: soru ve doküman için farklı önek)
# ---------------------------------------------------------------------------

def embed_documents(texts, titles=None):
    titles = titles or ["none"] * len(texts)
    inputs = [config.DOC_PREFIX.format(title=title) + text for text, title in zip(texts, titles)]
    return ollama.embed(model=config.EMBED_MODEL, input=inputs, keep_alive=config.KEEP_ALIVE).embeddings


def embed_query(question):
    query = config.QUERY_PREFIX + question
    return ollama.embed(model=config.EMBED_MODEL, input=query, keep_alive=config.KEEP_ALIVE).embeddings[0]


# ---------------------------------------------------------------------------
# Vektör veritabanı (Chroma)
# ---------------------------------------------------------------------------

def get_client():
    # Telemetri kapalı: demo internetsiz çalışır, dışarıya veri gönderilmez
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                     settings=chromadb.Settings(anonymized_telemetry=False))


@lru_cache
def get_collection(name):
    return get_client().get_collection(name)


def vector_search(collection_name, question, top_k):
    """Embedding araması. Parçaları benzerlik sırasıyla döndürür."""
    result = get_collection(collection_name).query(
        query_embeddings=[embed_query(question)], n_results=top_k
    )
    return [
        {"id": id_, "text": doc, **meta}
        for id_, doc, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0])
    ]


# ---------------------------------------------------------------------------
# Kelime araması (BM25)
# ---------------------------------------------------------------------------

def tokenize(text):
    """Türkçe için basit tokenizer: küçük harf + Türkçe karakter sadeleştirme + ilk 5 harf.

    'çalışabilir' ve 'calissam' -> 'calis', 'izmirden' -> 'izmir'.
    Klavyede Türkçe karakter kullanmayan ve ek alan kelimelerle başa çıkmanın en basit yolu.
    """
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = text.translate(str.maketrans("çğıöşü", "cgiosu"))
    return [word[:5] for word in re.findall(r"\w+", text)]


@lru_cache
def _bm25_index(collection_name):
    data = get_collection(collection_name).get()
    chunks = [{"id": id_, "text": doc, **meta}
              for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])]
    return BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks


def bm25_search(collection_name, question, top_k):
    bm25, chunks = _bm25_index(collection_name)
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k] if scores[i] > 0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(result_lists, k=config.RRF_K):
    """Her listedeki sıraya göre puan: 1 / (k + sıra). Puanlar toplanır."""
    scores, chunks = {}, {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (k + rank)
            chunks[chunk["id"]] = chunk
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [chunks[i] for i in ranked_ids]


# ---------------------------------------------------------------------------
# LLM (gemma4)
# ---------------------------------------------------------------------------

def chat(user_prompt, system_prompt=None, json_output=False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = ollama.chat(
        model=config.CHAT_MODEL,
        messages=messages,
        options=config.CHAT_OPTIONS,
        think=config.CHAT_THINK,
        keep_alive=config.KEEP_ALIVE,
        format="json" if json_output else None,
    )
    return response.message.content.strip()


RERANK_PROMPT = """Soru: {question}

Parça:
{chunk}

Bu parça soruyu cevaplamak için ne kadar alakalı? 0 (hiç alakasız) ile 10 (soruyu doğrudan cevaplıyor) arasında puan ver.
Sadece şu formatta JSON döndür: {{"puan": <sayı>}}"""


def rerank_score(question, chunk_text):
    """gemma4'e 0–10 arası alaka puanı verdirir. Hata olursa 0."""
    try:
        reply = chat(RERANK_PROMPT.format(question=question, chunk=chunk_text), json_output=True)
        score = float(json.loads(reply)["puan"])
        return max(0.0, min(10.0, score))
    except Exception:
        return 0.0


def rerank(question, chunks, top_k):
    scored = [{**chunk, "score": rerank_score(question, chunk["text"])} for chunk in chunks]
    # sorted() kararlıdır: eşit puanda RRF sırası korunur
    return sorted(scored, key=lambda c: c["score"], reverse=True)[:top_k]
