"""Üç sistemi eval/questions.jsonl ile çalıştırıp RAGAS ile ölçer.

Metrikler:
- context precision: getirilen parçalar doğru cevap için ne kadar alakalı? (referans cevapla)
- faithfulness: üretilen cevap, getirilen parçalara ne kadar sadık?
Hakem model: lokal gemma4 (langchain_ollama üzerinden).

Not: Kurulu RAGAS 0.4.3'te langchain modelleri yalnızca `evaluate()` ile kullanılabiliyor
(yeni `ragas.metrics.collections` metrikleri langchain kabul etmiyor). Bu yüzden
`ragas.metrics` içindeki metrikler kullanıldı; RAGAS bunlar için deprecation uyarısı verir.

Çalıştırma:  python eval/run_eval.py            (30 soru × 3 sistem, uzun sürer)
             python eval/run_eval.py --limit 5  (hızlı deneme)
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # RAGAS telemetrisini kapat (offline çalışma)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama  # noqa: E402
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference  # noqa: E402

import config  # noqa: E402
from pipelines import v1_basic, v2_better, v3_sources  # noqa: E402

SYSTEMS = {"v1 Basic": v1_basic, "v2 Better search": v2_better, "v3 With sources": v3_sources}
QUESTIONS_FILE = Path(__file__).parent / "questions.jsonl"


def load_questions(limit=None):
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def collect(system, questions):
    samples = []
    for i, row in enumerate(questions, start=1):
        result = system.answer(row["question"])
        samples.append({
            "user_input": row["question"],
            "response": result["answer"],
            "retrieved_contexts": [c["text"] for c in result["chunks"]],
            "reference": row["reference"],
        })
        print(f"  {i}/{len(questions)} {row['question']}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="sadece ilk N soruyu kullan")
    args = parser.parse_args()

    judge = ChatOllama(model=config.CHAT_MODEL, temperature=0, seed=42,
                       reasoning=False, num_ctx=8192, keep_alive=config.KEEP_ALIVE)
    metrics = [LLMContextPrecisionWithReference(), Faithfulness()]
    questions = load_questions(args.limit)

    scores = {}
    for name, system in SYSTEMS.items():
        print(f"\n{name}: cevaplar üretiliyor…")
        dataset = EvaluationDataset.from_list(collect(system, questions))
        print(f"{name}: RAGAS ölçüyor…")
        result = evaluate(dataset, metrics=metrics, llm=judge,
                          run_config=RunConfig(max_workers=2, timeout=600))
        df = result.to_pandas()
        scores[name] = {
            "context_precision": df["llm_context_precision_with_reference"].mean(),
            "faithfulness": df["faithfulness"].mean(),
        }

    print(f"\nSonuçlar ({len(questions)} soru, hakem: {config.CHAT_MODEL})\n")
    print(f"{'Sistem':<20} {'Context precision':>18} {'Faithfulness':>14}")
    print("-" * 54)
    for name, s in scores.items():
        print(f"{name:<20} {s['context_precision']:>18.2f} {s['faithfulness']:>14.2f}")


if __name__ == "__main__":
    main()
