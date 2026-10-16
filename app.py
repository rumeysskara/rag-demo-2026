"""DevFest RAG demosu: aynı soru, aynı model, 3 farklı RAG yöntemi.

Çalıştırma:  streamlit run app.py
"""

import html
import time

import streamlit as st

import config
from pipelines import v1_basic, v2_better, v3_sources

SYSTEMS = [
    {"key": "v1", "title": "v1 · Basic", "color": "#F28B82", "module": v1_basic,
     "tags": ["Sabit 70 kelime", "Sadece embedding", "Düz prompt"]},
    {"key": "v2", "title": "v2 · Better search", "color": "#FDD663", "module": v2_better,
     "tags": ["Madde bazlı parça", "Embedding + BM25", "RRF", "Reranker"]},
    {"key": "v3", "title": "v3 · With sources", "color": "#81C995", "module": v3_sources,
     "tags": ["v2 araması", "Numaralı kaynaklar", "“Bilmiyorum” kuralı"]},
]

PRESET_QUESTIONS = [
    "Onay almadan evden çalışabilir miyim?",
    "ay sonu izmirden calissam olur mu??",
    "Kullanmadığım izinler seneye devreder mi?",
    "Şirketin hisse fiyatı ne?",
]

st.set_page_config(page_title="RAG Demo", layout="wide")

# --- Stil: koyu tema üstüne sahne için büyük, sade kartlar ---
css = """
<style>
.block-container { padding-top: 2rem; max-width: 1600px; }
h1 { font-size: 2.6rem !important; margin-bottom: 0 !important; }
.subtitle { color: #9AA0A6; font-size: 1rem; margin-bottom: 1.2rem; }
.pill { display: inline-block; padding: 0.15rem 0.7rem; margin-right: 0.4rem; border-radius: 999px;
        background: #1A1D27; border: 1px solid #2D3140; font-size: 0.85rem; color: #BDC1C6; }
.pill.offline { border-color: #81C995; color: #81C995; }

.asked { font-size: 1.5rem; margin: 0.4rem 0 1rem; color: #E8EAED; }
.asked span { color: #9AA0A6; font-size: 1rem; display: block; margin-bottom: 0.2rem; }

.sys-card { border-top: 4px solid var(--c); background: #1A1D27; border-radius: 0.8rem;
            padding: 0.9rem 1.1rem; margin-bottom: 0.8rem; min-height: 8.6rem; }
.sys-card h3 { margin: 0 0 0.5rem; padding: 0; color: var(--c); font-size: 1.5rem; }
.tag { display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.55rem; margin: 0 0.3rem 0.3rem 0;
       border-radius: 999px; background: #0E1117; color: #BDC1C6; border: 1px solid #2D3140; }

.timer { font-size: 0.85rem; color: #9AA0A6; margin: -0.3rem 0 0.6rem; text-align: right; }

.sources { font-size: 1rem; margin-top: 0.4rem; }
.sources div { padding: 0.35rem 0.7rem; margin-top: 0.3rem; border-radius: 0.5rem;
               background: #1A1D27; border-left: 3px solid #81C995; }

.chunk { background: #0E1117; border: 1px solid #2D3140; border-radius: 0.6rem;
         padding: 0.7rem 0.9rem; margin-bottom: 0.7rem; }
.chunk-meta { font-size: 0.8rem; color: #9AA0A6; margin-bottom: 0.4rem; display: flex;
              justify-content: space-between; gap: 0.5rem; }
.chunk-text { font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; color: #DADCE0; }
.score { display: inline-block; width: 60px; height: 6px; border-radius: 3px; background: #2D3140;
         vertical-align: middle; margin-left: 0.3rem; overflow: hidden; }
.score i { display: block; height: 100%; background: #FDD663; }

/* Hazır soru butonları */
div[data-testid="stButton"] button { border-radius: 999px; min-height: 3rem; }
div[data-testid="stButton"] button p { font-size: 0.95rem; white-space: normal; line-height: 1.25; }
"""
for s in SYSTEMS:  # her sistemin cevap kutusu kendi renginde
    css += f"""
.st-key-answer_{s['key']} {{ border-left: 5px solid {s['color']} !important; background: #1A1D27;
    border-radius: 0.8rem; min-height: 9rem; }}
.st-key-answer_{s['key']} p, .st-key-answer_{s['key']} li {{ font-size: 1.55rem; line-height: 1.45; }}
"""
st.markdown(css + "</style>", unsafe_allow_html=True)


# --- Başlık ---
st.title("Aynı soru, 3 RAG yöntemi")
st.markdown(
    f'<div class="subtitle"><span class="pill">Cevap: {config.CHAT_MODEL}</span>'
    f'<span class="pill">Embedding: {config.EMBED_MODEL}</span>'
    f'<span class="pill offline">● Local</span></div>',
    unsafe_allow_html=True,
)


def set_question(text):
    st.session_state.question = text
    st.session_state.ask = True


# --- Soru kutusu ---
with st.form("ask_form", border=False):
    input_col, button_col = st.columns([6, 1])
    input_col.text_input("Sorunuz", key="question", label_visibility="collapsed",
                         placeholder="İK dokümanlarına bir soru sorun…")
    if button_col.form_submit_button("Sor", type="primary", use_container_width=True):
        st.session_state.ask = True

preset_columns = st.columns(len(PRESET_QUESTIONS))
for column, text in zip(preset_columns, PRESET_QUESTIONS):
    column.button(text, on_click=set_question, args=(text,), use_container_width=True)


# --- Sonuç gösterimi ---
def render_result(system, result, seconds):
    st.markdown(f'<div class="timer">⏱ {seconds:.1f} sn</div>', unsafe_allow_html=True)
    with st.container(border=True, key=f"answer_{system['key']}"):
        st.markdown(result["answer"])

    if system["module"] is v3_sources:
        # Sadece cevapta [1], [2] diye atıf yapılan kaynakları göster
        cited = [s for i, s in enumerate(result["sources"], start=1) if f"[{i}]" in result["answer"]]
        items = "".join(f"<div>{html.escape(s)}</div>" for s in cited) or "<div>Kaynak kullanılmadı</div>"
        st.markdown(f'<div class="sources"><b>Kaynaklar</b>{items}</div>', unsafe_allow_html=True)

    with st.expander(f"Getirilen parçalar ({len(result['chunks'])})"):
        for i, chunk in enumerate(result["chunks"], start=1):
            score = ""
            if "score" in chunk:
                width = chunk["score"] * 10
                score = f'reranker {chunk["score"]:.0f}/10<span class="score"><i style="width:{width}%"></i></span>'
            st.markdown(
                f'<div class="chunk"><div class="chunk-meta">'
                f'<span>{i}. {html.escape(chunk["source"])} · {html.escape(chunk["section"])}</span>'
                f'<span>{score}</span></div>'
                f'<div class="chunk-text">{html.escape(chunk["text"])}</div></div>',
                unsafe_allow_html=True,
            )


question = st.session_state.get("question", "").strip()
run_now = st.session_state.pop("ask", False) and question
if run_now:
    st.session_state.results = {}
    st.session_state.asked = question

if st.session_state.get("asked"):
    st.markdown(f'<div class="asked"><span>Soru</span>{html.escape(st.session_state.asked)}</div>',
                unsafe_allow_html=True)

columns = st.columns(3, gap="medium")
for column, system in zip(columns, SYSTEMS):
    with column:
        tags = "".join(f'<span class="tag">{t}</span>' for t in system["tags"])
        st.markdown(f'<div class="sys-card" style="--c:{system["color"]}"><h3>{system["title"]}</h3>{tags}</div>',
                    unsafe_allow_html=True)

        if run_now:
            with st.spinner("Düşünüyor…"):
                start = time.perf_counter()
                result = system["module"].answer(question)
                st.session_state.results[system["key"]] = (result, time.perf_counter() - start)

        # Sonuçlar session_state'te durur: başka bir yere tıklayınca kaybolmaz
        if system["key"] in st.session_state.get("results", {}):
            render_result(system, *st.session_state.results[system["key"]])
