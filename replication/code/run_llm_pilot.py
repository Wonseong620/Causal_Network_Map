# -*- coding: utf-8 -*-
"""Phase 4 pilot: zero-shot LLM sector classification via the Claude API.

Implements the GPT-OSS upgrade path specified in
PRD_iran_v3_io_matrix_generation.docx section 9, using prompt.txt verbatim as
the instruction block, claude-haiku-4-5, and structured outputs (50-key JSON
schema) so parsing never fails. Post-processing follows the PRD: clip to
[0,1], missing keys -> 0, renormalize to sum 1, uniform fallback + low-evidence
flag on zero mass.

Stratified pilot sample: N_PER_LANG articles per language, stratified by
type_risk. Compares against the TF-IDF baseline (top-sector agreement, cosine
similarity, entropy, uniform-row reduction).

Usage:  python run_llm_pilot.py [n_per_lang]   (default 100; smoke test: 2)
Output: figures/llm_pilot/llm_probs_<lang>.csv + pilot_summary.csv
"""
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "llm_pilot"
OUT.mkdir(parents=True, exist_ok=True)

N_PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL = "claude-haiku-4-5"
MAX_WORKERS = 8
PROB = [f"IO_V1_{i}" for i in range(1, 51)]
LANGS = {
    "arabic": "iran_ar_v3_with_io_matrix.csv",
    "chinese": "iran_ch_v3_with_io_matrix.csv",
    "english": "iran_en_v3_with_io_matrix.csv",
    "persian": "iran_pe_v3_with_io_matrix.csv",
}
TEXT_FIELDS = ["title", "economic_subtopic", "type_risk", "fact_eng",
               "opinion_eng", "fact", "opinion", "content"]

# ---- prompt: instruction block from prompt.txt, article fields appended ----
PROMPT_TEMPLATE = io.open(BASE / "prompt.txt", encoding="utf-8").read()
marker = "Article fields:"
INSTRUCTIONS = PROMPT_TEMPLATE.split(marker)[0].rstrip()

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in PROB},
    "required": PROB,
    "additionalProperties": False,
}

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env


def build_user_message(row: pd.Series) -> str:
    parts = ["Article fields:"]
    for f in TEXT_FIELDS:
        v = row.get(f)
        if pd.isna(v) or str(v).strip() == "":
            continue
        text = str(v)
        if f == "content" and len(text) > 6000:
            text = text[:6000]
        parts.append(f"{f}: {text}")
    return "\n".join(parts)


def classify(row: pd.Series) -> dict:
    """One article -> post-processed 50-sector probability dict + flags."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},
        }],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": build_user_message(row)}],
    )
    text = next(b.text for b in msg.content if b.type == "text")
    raw = json.loads(text)
    # PRD post-processing: clip, fill missing, renormalize / uniform fallback
    vec = np.array([float(raw.get(k, 0.0)) for k in PROB])
    vec = np.clip(vec, 0.0, 1.0)
    s = vec.sum()
    low_evidence = False
    if s > 0:
        vec = vec / s
    else:
        vec = np.full(50, 0.02)
        low_evidence = True
    usage = msg.usage
    return {
        "vec": vec, "low_evidence": low_evidence,
        "in_tok": usage.input_tokens,
        "out_tok": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens or 0,
    }


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    grp = df.groupby("type_risk", dropna=False, group_keys=False)
    take = grp.apply(lambda g: g.sample(
        n=max(1, int(round(n * len(g) / len(df)))), random_state=seed))
    if len(take) > n:
        take = take.sample(n=n, random_state=seed)
    return take


summary = []
for lang, fname in LANGS.items():
    t0 = time.time()
    df = pd.read_csv(BASE / fname, low_memory=False)
    sample = stratified_sample(df, N_PER_LANG)
    print(f"=== {lang}: sampled {len(sample)} articles ===")

    results: dict[int, dict] = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(classify, row): idx
                   for idx, row in sample.iterrows()}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:  # noqa: BLE001 - log and continue pilot
                errors += 1
                print(f"  ERROR idx={idx}: {type(e).__name__}: {e}")
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(sample)} ({time.time()-t0:.0f}s)")

    ok_idx = list(results.keys())
    L = np.vstack([results[i]["vec"] for i in ok_idx])           # LLM probs
    T = sample.loc[ok_idx, PROB].to_numpy(dtype=float)           # TF-IDF probs

    # diagnostics
    top_agree = float((L.argmax(1) == T.argmax(1)).mean())
    cos = float(np.mean(np.einsum("ij,ij->i", L, T)
                        / (np.linalg.norm(L, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)))
    def entropy(M):
        P = np.clip(M, 1e-12, 1)
        return float(np.mean(-(P * np.log(P)).sum(1) / np.log(50)))
    uni_tfidf = float(((T.max(1) - T.min(1)) < 1e-9).mean())
    uni_llm = float(np.mean([results[i]["low_evidence"] for i in ok_idx]))
    in_tok = sum(results[i]["in_tok"] for i in ok_idx)
    out_tok = sum(results[i]["out_tok"] for i in ok_idx)
    cache_rd = sum(results[i]["cache_read"] for i in ok_idx)
    cost = in_tok / 1e6 * 1.0 + out_tok / 1e6 * 5.0

    out_df = pd.DataFrame(L, columns=PROB)
    out_df.insert(0, "orig_index", ok_idx)
    out_df.insert(1, "low_evidence", [results[i]["low_evidence"] for i in ok_idx])
    out_df.to_csv(OUT / f"llm_probs_{lang}.csv", index=False)

    row = {
        "language": lang, "n": len(ok_idx), "errors": errors,
        "top1_agreement": round(top_agree, 3),
        "mean_cosine": round(cos, 3),
        "entropy_llm": round(entropy(L), 3),
        "entropy_tfidf": round(entropy(T), 3),
        "uniform_share_llm": round(uni_llm, 3),
        "uniform_share_tfidf": round(uni_tfidf, 3),
        "input_tokens": in_tok, "output_tokens": out_tok,
        "cache_read_tokens": cache_rd,
        "cost_usd": round(cost, 3),
        "elapsed_s": round(time.time() - t0),
    }
    summary.append(row)
    print("  " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "language"))

s = pd.DataFrame(summary)
s.to_csv(OUT / "pilot_summary.csv", index=False)
print("\n===== PILOT SUMMARY =====")
print(s.to_string(index=False))
print(f"\ntotal cost: ${s.cost_usd.sum():.2f}")
print("DONE ->", OUT)
