"""Ölçüm A: Dört demo sorusunda her Gemma çağrısının Ollama metrikleri + uçtan uca süre.

Çalıştırma:  python bench/measure_demo.py   ->  results/demo_calls.csv
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipelines import common, v1_basic, v2_better, v3_sources  # noqa: E402

QUESTIONS = [
    "Onay almadan evden çalışabilir miyim?",
    "ay sonu izmirden calissam olur mu??",
    "Kullanmadığım izinler seneye devreder mi?",
    "Şirketin hisse fiyatı ne?",
]
SYSTEMS = [("v1", v1_basic), ("v2", v2_better), ("v3", v3_sources)]
NS = 1e9

# ollama.chat'i sarıp her yanıtın metriklerini kaydet
calls = []
_original_chat = common.ollama.chat


def recording_chat(*args, **kwargs):
    response = _original_chat(*args, **kwargs)
    calls.append({
        "kind": "rerank" if kwargs.get("format") == "json" else "answer",
        "prompt_eval_count": response.prompt_eval_count or 0,
        "prompt_eval_s": (response.prompt_eval_duration or 0) / NS,
        "eval_count": response.eval_count or 0,
        "eval_s": (response.eval_duration or 0) / NS,
        "total_s": (response.total_duration or 0) / NS,
    })
    return response


common.ollama.chat = recording_chat


def main():
    print(f"Model: {config.CHAT_MODEL} | num_ctx={config.NUM_CTX} | think={config.CHAT_THINK}")
    print("Isıtma…")
    common.chat("Merhaba")
    common.embed_query("Merhaba")
    calls.clear()

    rows, e2e = [], []
    for qi, question in enumerate(QUESTIONS, start=1):
        for name, system in SYSTEMS:
            start_index = len(calls)
            start = time.perf_counter()
            system.answer(question)
            e2e.append((qi, name, time.perf_counter() - start, len(calls) - start_index))
            for c in calls[start_index:]:
                rows.append({"question": qi, "system": name, **c})

    header = f"{'S':>2} {'Sis':<3} {'Çağrı':<7} {'prompt tok':>10} {'prompt sn':>9} {'üretilen tok':>12} {'üretim sn':>9} {'toplam sn':>9}"
    print("\nHer Gemma çağrısı\n" + header + "\n" + "-" * len(header))
    for r in rows:
        print(f"{r['question']:>2} {r['system']:<3} {r['kind']:<7} {r['prompt_eval_count']:>10} "
              f"{r['prompt_eval_s']:>9.3f} {r['eval_count']:>12} {r['eval_s']:>9.3f} {r['total_s']:>9.3f}")

    print("\nUçtan uca süre (embedding + arama + tüm Gemma çağrıları)")
    print(f"{'S':>2} {'Sistem':<6} {'Gemma çağrısı':>13} {'süre sn':>8}")
    for qi, name, seconds, n in e2e:
        print(f"{qi:>2} {name:<6} {n:>13} {seconds:>8.2f}")
    for name, _ in SYSTEMS:
        avg = sum(s for _, n, s, _ in e2e if n == name) / len(QUESTIONS)
        print(f"   {name} ortalama: {avg:.2f} sn")

    longest = max(rows, key=lambda r: r["prompt_eval_count"])
    print(f"\nEn uzun prompt: {longest['prompt_eval_count']} token, okuma {longest['prompt_eval_s']:.3f} sn "
          f"(soru {longest['question']}, {longest['system']} {longest['kind']})")
    total_tok = sum(r["prompt_eval_count"] for r in rows)
    total_s = sum(r["prompt_eval_s"] for r in rows)
    print(f"Tüm çağrılarda okuma hızı: {total_tok} token / {total_s:.2f} sn = {total_tok / total_s:.0f} token/sn")

    out = Path(__file__).resolve().parent.parent / "results" / "demo_calls.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
