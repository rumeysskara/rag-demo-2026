"""Demo ayarları. Model adlarını `ollama list` çıktısına göre buradan değiştirin."""

from pathlib import Path

# --- Modeller (Ollama) ---
CHAT_MODEL = "gemma4:e4b"
EMBED_MODEL = "embeddinggemma-2:270m"

# --- EmbeddingGemma 2 görev önekleri ---
# Kaynak: huggingface.co/google/embeddinggemma-2 -> config_sentence_transformers.json
# Soru:     "task: search result | query: {metin}"
# Doküman:  "title: {başlık veya none} | text: {metin}"
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: {title} | text: "

# --- Yollar ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
V1_COLLECTION = "v1_basic"
V2_COLLECTION = "v2_better"

# --- v1: sabit boyutlu bölme ---
V1_CHUNK_WORDS = 70  # her parçadaki kelime sayısı (overlap yok)
V1_TOP_K = 3

# --- v2 / v3: hybrid arama + reranker ---
VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RERANK_CANDIDATES = 10  # RRF sonrası kaç parça gemma4'e puanlatılsın
FINAL_TOP_K = 3

# --- Üretim ayarları (tekrarlanabilirlik için) ---
CHAT_OPTIONS = {"temperature": 0, "seed": 42}
KEEP_ALIVE = "2h"  # modeller sunum boyunca bellekte kalsın (ilk yükleme ~30 sn)
CHAT_THINK = False  # gemma4'ün "düşünme" modu kapalı: daha hızlı ve kısa cevap
