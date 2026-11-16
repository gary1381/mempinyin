"""Offline P1 reranker: rescore libime N-best sentences with a small causal LM.

Primary config is fixed in P1_preregistration.md; everything else is printed as exploratory.
Usage: python rerank_lm.py nbest_16.jsonl Qwen/Qwen3-0.6B out.json
"""
import json
import math
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)
nbest_path, model_id, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(nbest_path, encoding="utf-8")]

tok = AutoTokenizer.from_pretrained(model_id)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
load_s = time.time() - t0
prefix = tok.convert_tokens_to_ids("<|endoftext|>")


@torch.inference_mode()
def lm_logprob(texts):
    """Total natural-log probability of each text + '。' given the <|endoftext|> prefix."""
    ids = [[prefix] + tok(t + "。", add_special_tokens=False)["input_ids"] for t in texts]
    width = max(map(len, ids))
    inp = torch.full((len(ids), width), prefix)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    logp = torch.log_softmax(model(input_ids=inp, attention_mask=mask).logits[:, :-1].float(), -1)
    tgt = inp[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
    return tok_lp.sum(-1).tolist()


def mcnemar(base, new):
    b = sum(x and not y for x, y in zip(base, new))  # base right, new wrong
    c = sum(y and not x for x, y in zip(base, new))  # new right, base wrong
    m, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return b, c, p


lm_lp_all, ms_all, ms_top3 = [], [], []
lm_lp_all.clear()
for r in rows:
    texts = [c["text"] for c in r["cands"]]
    a = time.perf_counter()
    lps = lm_logprob(texts)
    ms_all.append((time.perf_counter() - a) * 1e3)
    a = time.perf_counter()
    lm_logprob(texts[:3])
    ms_top3.append((time.perf_counter() - a) * 1e3)
    lm_lp_all.append(lps)

base = [r["cands"][0]["text"] == r["ref"] for r in rows]


def pick(r, lps, k=None, lam=0.0):
    cands = r["cands"][:k] if k else r["cands"]
    # libime scores are log10; convert to natural log before mixing with LM log-probs
    s = [lps[i] + lam * c["score"] * math.log(10) for i, c in enumerate(cands)]
    best = max(range(len(cands)), key=lambda i: (s[i], -i))
    return cands[best]["text"]


def evaluate(name, k=None, lam=0.0):
    chosen = [pick(r, lps, k, lam) for r, lps in zip(rows, lm_lp_all)]
    ok = [c == r["ref"] for c, r in zip(chosen, rows)]
    b, c, p = mcnemar(base, ok)
    print(f"{name}: top1={sum(ok)}/{len(ok)}={100 * sum(ok) / len(ok):.1f}%  "
          f"broken={b} fixed={c} McNemar p={p:.4f}")
    return chosen, ok


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


print(f"model={model_id} load_s={load_s:.1f} n={len(rows)} libime_top1={sum(base)}/{len(base)}")
print(f"rerank_ms all-cands p50={pct(ms_all, .5):.0f} p95={pct(ms_all, .95):.0f}; "
      f"top3 p50={pct(ms_top3, .5):.0f} p95={pct(ms_top3, .95):.0f}  (2 CPU threads)")
chosen, ok = evaluate("PRIMARY  LM-only, all N-best")
evaluate("explore  LM-only, top-3", k=3)
for lam in (0.25, 0.5, 1.0):
    evaluate(f"explore  LM + {lam}*libime, all N-best", lam=lam)
evaluate("explore  LM + 1.0*libime, top-3", k=3, lam=1.0)

detail = []
for r, c, o, b, lps in zip(rows, chosen, ok, base, lm_lp_all):
    ref_i = next(i for i, x in enumerate(r["cands"]) if x["text"] == r["ref"])
    detail.append({"ref": r["ref"], "libime": r["cands"][0]["text"], "rerank": c, "libime_ok": b, "rerank_ok": o,
                   "ref_rank_libime": ref_i, "lm_ref": round(lps[ref_i], 2), "lm_chosen": round(max(lps), 2)})
    if o != b or not o:
        print(f"{'FIXED ' if o and not b else 'BROKEN' if b and not o else 'STILL '} ref={r['ref']} "
              f"libime={r['cands'][0]['text']} rerank={c}")
json.dump({"model": model_id, "rows": detail, "rerank_ms": ms_all, "rerank_top3_ms": ms_top3},
          open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
