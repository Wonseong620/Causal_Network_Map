# -*- coding: utf-8 -*-
"""Retry only the requests that failed in run_llm_full_batch.py (billing
interruption), then merge the recovered rows into the existing
llm_probs_full_<lang>.csv outputs.

Usage: python run_llm_retry_failed.py            # submit + poll + collect
       python run_llm_retry_failed.py --resume   # skip submit, poll manifest
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
MANIFEST = OUT / "retry_manifest.json"

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


def load_failed_by_lang() -> dict[str, list[int]]:
    ids = json.loads((OUT / "failed_ids.json").read_text())
    by_lang: dict[str, list[int]] = {l: [] for l in LANGS}
    for cid in ids:
        lang, idx = cid.rsplit("_", 1)
        by_lang[lang].append(int(idx))
    return by_lang


def submit_all() -> list[dict]:
    by_lang = load_failed_by_lang()
    manifest = []
    for lang, fname in LANGS.items():
        idxs = sorted(by_lang[lang])
        if not idxs:
            continue
        df = pd.read_csv(BASE / fname, low_memory=False)
        subset = df.loc[idxs]
        print(f"=== {lang}: retrying {len(subset)} failed articles ===")
        for start in range(0, len(subset), CHUNK):
            chunk = subset.iloc[start:start + CHUNK]
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
                             "start": start, "n": len(requests)})
            MANIFEST.write_text(json.dumps(manifest, indent=1))
            print(f"  batch {batch.id} [{lang} {start}:{start+len(requests)}]")
    return manifest


def poll(manifest: list[dict]) -> None:
    t0 = time.time()
    pending = {m["id"] for m in manifest}
    while pending:
        time.sleep(60)
        for m in manifest:
            if m["id"] not in pending:
                continue
            b = client.messages.batches.retrieve(m["id"])
            if b.processing_status == "ended":
                pending.discard(m["id"])
        done = len(manifest) - len(pending)
        print(f"[{(time.time()-t0)/60:.0f}m] batches ended: {done}/{len(manifest)}")


def collect(manifest: list[dict]) -> None:
    by_lang: dict[str, dict[int, dict]] = {l: {} for l in LANGS}
    still_failed: list[str] = []
    usage_in = usage_out = usage_cache = 0
    err_detail: dict[str, int] = {}

    for m in manifest:
        for result in client.messages.batches.results(m["id"]):
            cid = result.custom_id
            lang, idx = cid.rsplit("_", 1)
            idx = int(idx)
            if result.result.type != "succeeded":
                still_failed.append(cid)
                key = str(getattr(result.result, "error", result.result.type))[:100]
                err_detail[key] = err_detail.get(key, 0) + 1
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
                still_failed.append(cid)
                continue
            s = vec.sum()
            low = s <= 0
            by_lang[lang][idx] = {
                "vec": (vec / s) if s > 0 else np.full(50, 0.02),
                "low": low,
            }
        print(f"collected retry batch {m['id']} ({m['lang']} {m['start']})")

    for lang, d in by_lang.items():
        if not d:
            continue
        idxs = sorted(d)
        new_df = pd.DataFrame(np.vstack([d[i]["vec"] for i in idxs]), columns=PROB)
        new_df.insert(0, "orig_index", idxs)
        new_df.insert(1, "low_evidence", [d[i]["low"] for i in idxs])

        existing_path = OUT / f"llm_probs_full_{lang}.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset="orig_index", keep="last")
            merged = merged.sort_values("orig_index").reset_index(drop=True)
        else:
            merged = new_df
        merged.to_csv(existing_path, index=False)
        print(f"{lang}: merged +{len(idxs)} recovered rows -> total {len(merged)}")

    (OUT / "still_failed_ids.json").write_text(json.dumps(still_failed))
    cost = (usage_in / 1e6 * 1.0 + usage_out / 1e6 * 5.0 + usage_cache / 1e6 * 0.1) * 0.5
    print(f"\nstill_failed={len(still_failed)}  input_tok={usage_in:,}  "
          f"output_tok={usage_out:,}  cache_read={usage_cache:,}  est. cost=${cost:.2f}")
    if err_detail:
        print("error breakdown:")
        for k, v in sorted(err_detail.items(), key=lambda x: -x[1])[:5]:
            print(f"  {v:6d} :: {k}")


def main() -> None:
    if "--resume" in sys.argv and MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        print(f"resuming with {len(manifest)} retry batches from manifest")
    else:
        manifest = submit_all()
    poll(manifest)
    collect(manifest)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
