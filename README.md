# Local RAG Demo

**Aynı soru, aynı model, üç farklı RAG sistemi. İnternet yok, bulut yok, tamamen lokal.**

Bu DevFest demosu, bir RAG sisteminin neden yanlış cevap verdiğini ve üç adımda nasıl
güvenilir hâle geldiğini canlı gösterir: daha iyi bölme, daha iyi arama ve kaynak gösterme.
Modeller Ollama'da, vektör veritabanı diskte çalışır.

| Sistem | Bölme | Arama | Prompt |
|---|---|---|---|
| **v1 Basic** | Sabit 70 kelime | Sadece embedding, ilk 3 | "Aşağıdaki bilgilere göre cevapla" |
| **v2 Better search** | Madde/başlık bazlı + bağlam satırı + metadata | Embedding + BM25 → RRF → gemma4 reranker | Aynı düz prompt |
| **v3 With sources** | v2 ile aynı | v2 ile aynı | Numaralı kaynaklar + "Bilmiyorum" talimatı |

## Kurulum

Gereksinimler: Python 3.11+, güncel [Ollama](https://ollama.com/download) (eski sürümler bu modelleri indiremez).

```bash
# 1. Modeller
ollama pull gemma4:e4b
ollama pull embeddinggemma-2:270m

# 2. Python ortamı
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. İndeks (bir kez; sonuç chroma_db/ klasörüne kaydedilir)
python build_index.py

# 4. Testler
pytest -s tests/

# 5. Demo
streamlit run app.py
```

Model adları `config.py` içindedir; `ollama list` çıktısına göre değiştirebilirsiniz.
Model veya `data/` değişirse `python build_index.py` komutunu yeniden çalıştırın.

## Demo soruları ve beklenen sonuçlar

| Soru | v1 | v2 | v3 |
|---|---|---|---|
| Onay almadan evden çalışabilir miyim? | "Evet" (onay şartını kaçırır) | "Hayır, yazılı onay gerekir" | "Hayır, onay gerekir [1]" |
| ay sonu izmirden calissam olur mu?? | Bilgi bulamaz | "Evet, İK'ya haber ver" | Aynısı + kaynak |
| Kullanmadığım izinler seneye devreder mi? | Yanlış/eksik | "Evet, bir kısmı" | "En fazla 5 gün [1]" |
| Şirketin hisse fiyatı ne? | "Belirtilmemiş" | "Belirtilmemiş" | "Bilmiyorum, bu bilgi kaynaklar arasında yer almıyor." |

**Neden v1 yanılıyor?** §4.2 maddesi ("…evden çalışabilir, ancak … yazılı onayı gerekir") 70 kelimelik
sınırda ikiye bölünüyor. v1 sadece "evden çalışabilir" yarısını getiriyor. `tests/test_chunking.py`
bu bölünmeyi doğrular. `data/hr_policy.md`'de §4.2'den önceki metni değiştirirseniz bölünme kayabilir,
testi tekrar çalıştırın.

**Neden v2 "izmirden calissam"ı buluyor?** BM25 tokenizer'ı Türkçe karakterleri sadeleştirip kelimelerin
ilk 5 harfini alıyor: `izmirden` → `izmir`, `calissam` / `çalışabilir` → `calis`.

`tests/test_demo.py` beklenen sonuç çıkmazsa getirilen parçaları yazdırır.

## Proje yapısı

```
config.py              model adları, chunk boyutu, top-k, RRF k, embedding önekleri
build_index.py         data/ → chroma_db/ (v1_basic ve v2_better koleksiyonları)
app.py                 Streamlit arayüzü (üç sütun)
pipelines/common.py    bölme, embedding, BM25, RRF, reranker, LLM çağrısı
pipelines/v1_basic.py  answer(question) -> {answer, chunks, sources}
pipelines/v2_better.py
pipelines/v3_sources.py
data/                  kurgusal şirketin İK politikası ve SSS'si
tests/                 chunking testi (modelsiz) + demo testi (Ollama ile)
eval/                  RAGAS ölçümü (opsiyonel)
```

**EmbeddingGemma 2 önekleri** ([model config'i](https://huggingface.co/google/embeddinggemma-2/blob/main/config_sentence_transformers.json)):
soru için `task: search result | query: `, doküman için `title: {başlık} | text: `.

**Tekrarlanabilirlik:** Tüm LLM çağrıları `temperature=0`, `seed=42` ve gemma4'ün düşünme modu
kapalı (`think=False`) olarak yapılır. Düşünme modu açıkken reranker çağrısı ~15 sn, kapalıyken ~1 sn sürer.

## Ölçüm (opsiyonel)

```bash
python eval/run_eval.py --limit 5   # hızlı deneme
python eval/run_eval.py             # 30 soru × 3 sistem (~45–60 dk)
```

`eval/questions.jsonl` içinde 30 soru ve referans cevap var (5'inin cevabı dokümanlarda yok).
RAGAS ile **context precision** ve **faithfulness** hesaplanır. Hakem model lokal gemma4'tür (langchain_ollama).
Not: RAGAS 0.4.3'te langchain modelleri yalnızca eski `ragas.metrics` metrikleriyle çalışıyor.
Bu yüzden çalışırken deprecation uyarıları görebilirsiniz. RAGAS ve Chroma telemetrisi kapalıdır.
