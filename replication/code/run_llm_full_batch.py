# -*- coding: utf-8 -*-
"""Phase 4 full run: classify all 58,530 articles into 50-sector probability
distributions via the Message Batches API (50% discount).

Same specification as run_llm_pilot.py (prompt.txt instructions, Haiku 4.5,
structured outputs, prompt caching, PRD post-processing). Submits per-language
batches in chunks, writes a manifest for recovery, polls to completion,
post-processes, and saves llm_probs_full_<lang>.csv keyed by the original
CSV row index.

Usage: python run_llm_full_batch.py            # submit + poll + collect
       python run_llm_full_batch.py --resume   # skip submit, poll manifest
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "llm_full"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "batches_manifest.json"

MODEL = "claude-haiku-4-5"
CHUNK = 2500
PROB = [f"IO_V1_{i}" for i in range(1, 51)]
LANGS = {
    "arabic": "iran_ar_v3_with_io_matrix.csv",
    "chinese": "iran_ch_v3_with_io_matrix.csv",
    "english": "iran_en_v3_with_io_matrix.csv",
    "persian": "iran_pe_v3_with_io_matrix.csv",
}
TEXT_FIELDS = ["title", "economic_subtopic", "type_risk", "fact_eng",
               "opinion_eng", "fact", "opinion", "content"]

INSTRUCTIONS = io.open(BASE / "prompt.txt", encoding="utf-8").read().split("Article fields:")[0].rstrip()
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in PROB},
    "required": PROB,
    "additionalProperties": False,
}

client = anthropic.Anthropic()


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


def make_params(row: pd.Series) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 2048,
        "system": [{
            "type": "text",
            "text": INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},
        }],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": build_user_message(row)}],
    }


def submit_all() -> list[dict]:
    manifest = []
    for lang, fname in LANGS.items():
        df = pd.read_csv(BASE / fname, low_memory=False)
        n = len(df)
        print(f"=== {lang}: {n} articles ===")
        for start in range(0, n, CHUNK):
            chunk = df.iloc[start:start + CHUNK]
            requests = [
                {"custom_id": f"{lang}_{idx}", "params": make_params(row)}
                for idx, row in chunk.iterrows()
            ]
            for attempt in range(3):
                try:
                    batch = client.messages.batches.create(requests=requests)
                    break
                except anthropic.APIStatusError as e:
                    print(f"  submit retry {attempt+1}: {e.status_code} {e.message}")
                    time.sleep(30)
            else:
                raise RuntimeError(f"failed to submit chunk {lang}:{start}")
            manifest.append({"id": batch.id, "lang": lang,
                             "start": start, "n": len(requests),
                             "status": "submitted"})
            MANIFEST.write_text(json.dumps(manifest, indent=1))
            print(f"  batch {batch.id} [{lang} {start}:{start+len(requests)}]")
    return manifest


def poll(manifest: list[dict]) -> None:
    t0 = time.time()
    pending = {m["id"] for m in manifest}
    while pending:
        time.sleep(120)
        for m in manifest:
            if m["id"] not in pending:
                continue
            b = client.messages.batches.retrieve(m["id"])
            if b.processing_status == "ended":
                pending.discard(m["id"])
                m["status"] = "ended"
                MANIFEST.write_text(json.dumps(manifest, indent=1))
        done = len(manifest) - len(pending)
        print(f"[{(time.time()-t0)/60:.0f}m] batches ended: {done}/{len(manifest)}")


def collect(manifest: list[dict]) -> None:
    by_lang: dict[str, dict[int, dict]] = {l: {} for l in LANGS}
    err_ids: list[str] = []
    usage_in = usage_out = usage_cache = 0

    for m in manifest:
        for result in client.messages.batches.results(m["id"]):
            cid = result.custom_id
            lang, idx = cid.rsplit("_", 1)
            idx = int(idx)
            if result.result.type != "succeeded":
                err_ids.append(cid)
                continue
            msg = result.result.message
            usage_in += msg.usage.input_tokens
            usage_out += msg.usage.output_tokens
            usage_cache += msg.usage.cache_read_input_tokens or 0
            try:
                text = next(b.text for b in msg.content if b.type == "text")
                raw = json.loads(text)
                vec = np.clip(np.array([float(raw.get(k, 0.0)) for k in PROB]), 0, 1)
            except Exception:
                err_ids.append(cid)
                continue
            s = vec.sum()
            low = s <= 0
            by_lang[lang][idx] = {
                "vec": (vec / s) if s > 0 else np.full(50, 0.02),
                "low": low,
            }
        print(f"collected batch {m['id']} ({m['lang']} {m['start']})")

    for lang, d in by_lang.items():
        if not d:
            continue
        idxs = sorted(d)
        M = np.vstack([d[i]["vec"] for i in idxs])
        out = pd.DataFrame(M, columns=PROB)
        out.insert(0, "orig_index", idxs)
        out.insert(1, "low_evidence", [d[i]["low"] for i in idxs])
        out.to_csv(OUT / f"llm_probs_full_{lang}.csv", index=False)
        print(f"{lang}: saved {len(idxs)} rows "
              f"(low-evidence {sum(d[i]['low'] for i in idxs)})")

    (OUT / "failed_ids.json").write_text(json.dumps(err_ids))
    # batch pricing = 50% of standard
    cost = (usage_in / 1e6 * 1.0 + usage_out / 1e6 * 5.0 + usage_cache / 1e6 * 0.1) * 0.5
    print(f"\nerrors={len(err_ids)}  input_tok={usage_in:,}  output_tok={usage_out:,}  "
          f"cache_read={usage_cache:,}  est. cost=${cost:.2f}")


def main() -> None:
    if "--resume" in sys.argv and MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        print(f"resuming with {len(manifest)} batches from manifest")
    else:
        manifest = submit_all()
    poll(manifest)
    collect(manifest)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
