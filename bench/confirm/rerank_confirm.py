"""Confirmation test (see PREREGISTRATION.md): libime top-1 vs bert-base-chinese int8 diff-PLL reranking with context.

Usage: python rerank_confirm.py testset.jsonl nbest_test.jsonl model_dir onnx_file out.json
"""
import json
import math
import sys
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

test_path, nbest_path, model_dir, onnx_file, out_path = sys.argv[1:6]
items = [json.loads(l) for l in open(test_path, encoding="utf-8")]
nbest = [json.loads(l) for l in open(nbest_path, encoding="utf-8")]
assert len(items) == len(nbest) and all(i["ref"] == n["ref"] for i, n in zip(items, nbest))

tok = AutoTokenizer.from_pretrained(model_dir)
so = ort.SessionOptions()
so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
sess = ort.InferenceSession(f"{model_dir}/onnx/{onnx_file}", so, providers=["CPUExecutionProvider"])
use_tt = "token_type_ids" in {i.name for i in sess.get_inputs()}
CLS, SEP, MASK = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
PERIOD = tok.convert_tokens_to_ids("。")


def char_ids(text):
    ids = tok.convert_tokens_to_ids(list(text))
    return [i if i is not None else tok.unk_token_id for i in ids]


def pll(texts, context, positions):
    """Sum over the given char positions of log P(char | rest), each text as [CLS] context text 。 [SEP]."""
    ctx = tok(context, add_special_tokens=False)["input_ids"] if context else []
    batch, owners = [], []
    for k, t in enumerate(texts):
        ids = [CLS] + ctx + char_ids(t) + [PERIOD, SEP]
        off = 1 + len(ctx)
        for i in (positions if positions is not None else range(len(t))):
            masked = list(ids)
            masked[off + i] = MASK
            batch.append(masked)
            owners.append((k, off + i, ids[off + i]))
    if not batch:
        return [0.0] * len(texts)
    width = max(map(len, batch))
    inp = np.zeros((len(batch), width), dtype=np.int64)
    att = np.zeros_like(inp)
    for r, x in enumerate(batch):
        inp[r, : len(x)], att[r, : len(x)] = x, 1
    feed = {"input_ids": inp, "attention_mask": att}
    if use_tt:
        feed["token_type_ids"] = np.zeros_like(inp)
    logits = sess.run(None, feed)[0]
    scores = [0.0] * len(texts)
    for r, (k, pos, gold) in enumerate(owners):
        row = logits[r, pos].astype(np.float64)
        m = row.max()
        scores[k] += float(row[gold] - m - math.log(np.exp(row - m).sum()))
    return scores


def diff_positions(texts):
    if len({len(t) for t in texts}) != 1:
        return None  # unequal lengths: full PLL
    return [i for i in range(len(texts[0])) if len({t[i] for t in texts}) > 1]


def rerank(cands, context):
    texts = [c["text"] for c in cands]
    if len(texts) < 2:
        return texts[0] if texts else ""
    s = pll(texts, context, diff_positions(texts))
    return texts[max(range(len(texts)), key=lambda i: (s[i], -i))]


def mcnemar(base, new):
    b = sum(x and not y for x, y in zip(base, new))
    c = sum(y and not x for x, y in zip(base, new))
    m, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return b, c, p


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


systems = {"S1": (3, True), "S2": (None, True), "S1_noctx": (3, False), "S2_noctx": (None, False)}
chosen = {k: [] for k in systems}
ms = {k: [] for k in systems}
for it, nb in zip(items, nbest):
    for name, (k, with_ctx) in systems.items():
        a = time.perf_counter()
        chosen[name].append(rerank(nb["cands"][:k] if k else nb["cands"], it["context"] if with_ctx else ""))
        ms[name].append((time.perf_counter() - a) * 1e3)

refs = [it["ref"] for it in items]
b0 = [bool(nb["cands"]) and nb["cands"][0]["text"] == r for nb, r in zip(nbest, refs)]
reach = sum(any(c["text"] == r for c in nb["cands"]) for nb, r in zip(nbest, refs))
n = len(items)
print(f"N={n}  ref in libime N-best16: {reach}/{n}")
print(f"B0 libime top-1: {sum(b0)}/{n} = {100 * sum(b0) / n:.1f}%")
ok = {k: [c == r for c, r in zip(v, refs)] for k, v in chosen.items()}
tests = {}
for name in systems:
    b, c, p = mcnemar(b0, ok[name])
    tests[name] = (b, c, p)
    print(f"{name}: {sum(ok[name])}/{n} = {100 * sum(ok[name]) / n:.1f}%  broken={b} fixed={c} raw p={p:.2e}  "
          f"rerank ms p50={pct(ms[name], .5):.1f} p95={pct(ms[name], .95):.1f}")

# Holm over the two primary hypotheses
prim = sorted(["S1", "S2"], key=lambda k: tests[k][2])
holm, running = {}, 0.0
for rank, k in enumerate(prim):
    running = max(running, min(1.0, (len(prim) - rank) * tests[k][2]))
    holm[k] = running
for k in ("S1", "S2"):
    b, c, _ = tests[k]
    verdict = "SIGNIFICANTLY BETTER" if holm[k] < 0.05 and c > b else "not significant"
    print(f"PRIMARY {k} vs B0: Holm-adjusted p={holm[k]:.2e} -> {verdict}")

for src in ("dialogue", "narrative"):
    idx = [i for i, it in enumerate(items) if it["source"] == src]
    if not idx:
        continue
    row =f"{src} (n={len(idx)}): B0 {100 * sum(b0[i] for i in idx) / len(idx):.1f}%"
    for k in systems:
        row += f"  {k} {100 * sum(ok[k][i] for i in idx) / len(idx):.1f}%"
    print(row)

json.dump({"items": [{"ref": r, "context": it["context"], "source": it["source"], "libime": nb["cands"][0]["text"] if nb["cands"] else "",
                      **{k: chosen[k][i] for k in systems}} for i, (r, it, nb) in enumerate(zip(refs, items, nbest))],
           "ms": ms}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
