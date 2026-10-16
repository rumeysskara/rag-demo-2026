"""Ölçüm B: Uzun metinde okuma (prefill) hızı.

Çalıştırma:  python bench/measure_prefill.py 8000 32000 64000
Sonuç:       results/prefill.csv  (token sayısı | okuma süresi | saniyede token)
"""

import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipelines.common import read_documents  # noqa: E402

RUNS = 3
TIMEOUT_S = 900  # tek çalıştırma 15 dakikayı geçerse o boyut atlanır
NS = 1e9
OUT = Path(__file__).resolve().parent.parent / "results" / "prefill.csv"


def machine_info():
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    ram_gb = int(sysctl("hw.memsize")) / 1024**3
    return f"{sysctl('machdep.cpu.brand_string')}, {sysctl('hw.ncpu')} çekirdek, {ram_gb:.0f} GB RAM, macOS {platform.mac_ver()[0]}"


def build_text(target_tokens, tokens_per_char):
    """data/ dokümanlarını tekrar ederek hedef token sayısına yakın bir metin oluşturur."""
    docs = "\n\n".join(read_documents().values())
    chars = int(target_tokens / tokens_per_char)
    return (docs * (chars // len(docs) + 1))[:chars]


def run_once(client, text, num_ctx, tag):
    # Önbelleği atlatmak için her denemede metnin başına farklı bir satır
    prompt = f"Deneme {tag} {time.time_ns()}\n{text}\n\nBu metnin konusu ne? Tek cümle."
    r = client.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_ctx": num_ctx, "num_predict": 20},
                    think=False, keep_alive="10m")
    return r.prompt_eval_count, r.prompt_eval_duration / NS


def main():
    targets = [int(t) for t in sys.argv[1:]] or [8000, 32000, 64000, 100000]
    client = ollama.Client(timeout=TIMEOUT_S)
    print(f"Makine: {machine_info()}")
    print(f"Model: {config.CHAT_MODEL} | think=False | num_predict=20 | her boyut {RUNS} kez\n")

    # Token/karakter oranını kalibre et
    sample = "\n\n".join(read_documents().values())
    count = client.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": sample}],
                        options={"num_ctx": 16384, "num_predict": 1}, think=False).prompt_eval_count
    tokens_per_char = count / len(sample)

    results = []
    for target in targets:
        num_ctx = int(target * 1.1) + 512
        text = build_text(target, tokens_per_char)
        print(f"~{target} token (num_ctx={num_ctx}): ısıtılıyor…", flush=True)
        try:
            client.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": "Merhaba"}],
                        options={"num_ctx": num_ctx, "num_predict": 1}, think=False, keep_alive="10m")
            runs = []
            for i in range(RUNS):
                tokens, seconds = run_once(client, text, num_ctx, f"{target}-{i}")
                runs.append((tokens, seconds))
                print(f"   deneme {i + 1}: {tokens} token, {seconds:.1f} sn, {tokens / seconds:.0f} token/sn", flush=True)
        except Exception as e:  # bellek yetmezse / zaman aşımı
            print(f"   ATLANDI: {type(e).__name__}: {e}", flush=True)
            continue
        tokens = round(sum(t for t, _ in runs) / RUNS)
        seconds = sum(s for _, s in runs) / RUNS
        results.append({"tokens": tokens, "prefill_s": round(seconds, 2), "tokens_per_s": round(tokens / seconds)})

    print(f"\n{'Token':>8} | {'Okuma (sn)':>10} | {'Token/sn':>8}")
    for r in results:
        print(f"{r['tokens']:>8} | {r['prefill_s']:>10.1f} | {r['tokens_per_s']:>8}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tokens", "prefill_s", "tokens_per_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
