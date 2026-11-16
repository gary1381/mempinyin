"""Offline reranker with a masked LM (bert-base-chinese ONNX): pseudo-log-likelihood over libime N-best.

Full PLL masks every character of a candidate in turn; diff PLL masks only the positions where the
reranked candidates disagree (one batched ONNX call per sentence either way).
Usage: python rerank_mlm.py nbest_16.jsonl model_dir onnx_file out.json
"""
import json
import math
import sys
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

nbest_path, model_dir, onnx_file, out_path = sys.argv[1:5]
rows = [json.loads(l) for l in open(nbest_path, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(model_dir)
so = ort.SessionOptions()
so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
if len(sys.argv) > 5 and sys.argv[5] == "noopt":  # q4f16 fails ORT CPU graph fusion
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
t0 = time.time()
sess = ort.InferenceSession(f"{model_dir}/onnx/{onnx_file}", so, providers=["CPUExecutionProvider"])
load_s = time.time() - t0
input_names = {i.name for i in sess.get_inputs()}
MASK = tok.mask_token_id


def encode(text):
    ids = tok(text + "。")["input_ids"]  # [CLS] chars 。 [SEP]
    assert len(ids) == len(text) + 3, f"non 1:1 char tokenization: {text}"
    return ids


def pll(texts, positions=None):
    """Sum of log P(char_i | rest) over char positions (all, or the given char indices) for each text."""
    batch, owners = [], []
    for k, t in enumerate(texts):
        ids = encode(t)
        for i in (positions if positions is not None else range(len(t))):
            masked = list(ids)
            masked[i + 1] = MASK
            batch.append(masked)
            owners.append((k, i + 1, ids[i + 1]))
    if not batch:
        return [0.0] * len(texts)
    width = max(map(len, batch))
    inp = np.zeros((len(batch), width), dtype=np.int64)
    att = np.zeros_like(inp)
    for r, x in enumerate(batch):
        inp[r, : len(x)], att[r, : len(x)] = x, 1
    feed = {"input_ids": inp, "attention_mask": att}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(inp)
    logits = sess.run(None, feed)[0].astype(np.float32)
    scores = [0.0] * len(texts)
    for r, (k, pos, gold) in enumerate(owners):
        row = logits[r, pos]
        scores[k] += float(row[gold] - (row.max() + np.log(np.exp(row - row.max()).sum())))
    return scores


def diff_positions(texts):
    if len({len(t) for t in texts}) != 1:
        return None  # unequal lengths: fall back to full PLL
    return [i for i in range(len(texts[0])) if len({t[i] for t in texts}) > 1]


def mcnemar(base, new):
    b = sum(x and not y for x, y in zip(base, new))
    c = sum(y and not x for x, y in zip(base, new))
    m, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return b, c, p


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


full_all, diff_all, full_top3, diff_top3 = [], [], [], []
ms = {"full_all": [], "diff_all": [], "full_top3": [], "diff_top3": []}
for r in rows:
    texts = [c["text"] for c in r["cands"]]
    for name, store, cands, diff in (("full_all", full_all, texts, False), ("diff_all", diff_all, texts, True),
                                     ("full_top3", full_top3, texts[:3], False),
                                     ("diff_top3", diff_top3, texts[:3], True)):
        a = time.perf_counter()
        store.append(pll(cands, diff_positions(cands) if diff else None))
        ms[name].append((time.perf_counter() - a) * 1e3)

base = [r["cands"][0]["text"] == r["ref"] for r in rows]


def evaluate(name, scores, lam=0.0):
    chosen = []
    for r, s in zip(rows, scores):
        cands = r["cands"][: len(s)]
        # libime scores are log10; convert to natural log before mixing with PLL
        tot = [s[i] + lam * c["score"] * math.log(10) for i, c in enumerate(cands)]
        chosen.append(cands[max(range(len(cands)), key=lambda i: (tot[i], -i))]["text"])
    ok = [c == r["ref"] for c, r in zip(chosen, rows)]
    b, c, p = mcnemar(base, ok)
    print(f"{name}: top1={sum(ok)}/{len(ok)}={100 * sum(ok) / len(ok):.1f}%  broken={b} fixed={c} McNemar p={p:.4f}")
    return chosen, ok


print(f"model={onnx_file} load_s={load_s:.2f} n={len(rows)} libime_top1={sum(base)}/{len(base)}")
for k, v in ms.items():
    print(f"rerank_ms {k}: p50={pct(v, .5):.1f} p95={pct(v, .95):.1f} max={max(v):.1f}  (2 threads)")
evaluate("MLM-only, all N-best (full PLL)", full_all)
evaluate("MLM-only, all N-best (diff PLL)", diff_all)
evaluate("MLM-only, top-3 (full PLL)", full_top3)
evaluate("MLM-only, top-3 (diff PLL)", diff_top3)
results = {}
for lam in (0.5, 1.0):
    evaluate(f"MLM + {lam}*libime, all N-best (full PLL)", full_all, lam)
    results[lam] = evaluate(f"MLM + {lam}*libime, top-3 (full PLL)", full_top3, lam)
    evaluate(f"MLM + {lam}*libime, top-3 (diff PLL)", diff_top3, lam)

chosen, ok = evaluate("detail: MLM-only, all N-best (full PLL)", full_all)
for r, c, o, b in zip(rows, chosen, ok, base):
    if o != b or not o:
        print(f"{'FIXED ' if o and not b else 'BROKEN' if b and not o else 'STILL '} ref={r['ref']} "
              f"libime={r['cands'][0]['text']} rerank={c}")
json.dump({"model": onnx_file, "ms": ms, "full_all": full_all, "diff_all": diff_all, "full_top3": full_top3, "diff_top3": diff_top3},
          open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
