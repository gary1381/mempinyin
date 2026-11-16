"""Recompute the headline tables from the stored outputs (no engines, no models, standard library only).

1. Four open-source engines on the 59-sentence dev set (docs/DESIGN.md section 5.1), from bench/results/results_*.txt
   written by libime_bench / gpy_bench / lpy_bench / lpy_nbest / rime_bench. Top-1 accuracy and the paired tests are
   recomputed from the per-sentence lines; latency and memory are read from each program's summary lines (the raw
   per-keystroke timings were not stored).
2. Pre-registered P1 dev-set reranking with Qwen3-0.6B (section 5.2), from bench/results/p1/rerank_qwen3-0.6b.json.
3. Pre-registered confirmation test on 360 new sentences (section 5.3), from bench/confirm/confirm_items.json
   (per-item correctness flags and per-item rerank times written by confirm/rerank_confirm.py).

Statistics: exact two-sided McNemar test on the discordant pairs; Holm correction over the two primary hypotheses
(H1: S1 > B0, H2: S2 > B0), as pre-registered in bench/confirm/PREREGISTRATION.md.
Exits non-zero if a recomputed number differs from the one reported in README.md / docs/DESIGN.md.

Usage: python3 bench/reproduce_tables.py
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
failures = []


def mcnemar(base, new):
    """(only base right, only new right, exact two-sided p), as in confirm/rerank_confirm.py."""
    b = sum(x and not y for x, y in zip(base, new))
    c = sum(y and not x for x, y in zip(base, new))
    m, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return b, c, p


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def check(label, got, want):
    """Exact match for ints / tuples / strings; for a number given as a string, equal to the string's number of decimals."""
    if isinstance(want, str) and isinstance(got, float):
        d = len(want.split(".")[1]) if "." in want else 0
        ok = abs(got - float(want)) <= 0.5 * 10**-d + 1e-9
    else:
        ok = got == want
    if not ok:
        failures.append(f"{label}: got {got}, documented {want}")
    return "" if ok else f"   <-- MISMATCH (documented: {want})"


def summary(tag):
    """key=value pairs from a benchmark's non-sentence lines."""
    kv = {}
    for line in open(os.path.join(RES, f"results_{tag}.txt"), encoding="utf-8"):
        if line.startswith(("S\t", "W\t")):
            continue
        prefix = line.split("_n=")[0] if line.startswith(("keystroke_n=", "fullsent_n=")) else ""
        for k, v in re.findall(r"(\w+)=([\d.]+)", line):
            timing = prefix and (k.startswith("p") or k == "max_us")
            kv[f"{prefix}.{k}" if timing else k] = float(v)
    return kv


# 1. four engines, 59 dev sentences
print("== 1. Four engines, 59 dev sentences (bench/sentences.tsv), whole-sentence top-1 ==")
engines = {"libgooglepinyin": "googlepinyin", "libpinyin": "libpinyin", "libime": "libime", "Rime": "rime"}
doc_correct = {"libgooglepinyin": 37, "libpinyin": 38, "libime": 50, "Rime": 34}
ok = {}
for name, tag in engines.items():
    rows = [l.split("\t") for l in open(os.path.join(RES, f"results_{tag}.txt"), encoding="utf-8") if l.startswith("S\t")]
    ok[name] = [r[1] == "OK" for r in rows]
    n, k = len(rows), sum(ok[name])
    print(f"  {name:16s} {k:3d}/{n} = {100 * k / n:.1f}%" + check(f"{name} top-1", k, doc_correct[name]))
for other, want in (("libgooglepinyin", "0.004"), ("libpinyin", "0.004"), ("Rime", "0.0004")):
    b, c, p = mcnemar(ok[other], ok["libime"])
    print(f"  libime vs {other:16s} only libime right {c:2d}, only {other} right {b:2d}, exact McNemar p = {p:.2g}"
          + check(f"libime vs {other} p", float(f"{p:.1g}"), want))

print("\n   latency and memory (from the stored summaries)")
# documented: load ms, RSS MB, keystroke p50/p95/p99/max, whole sentence p50/p99 (units as in the table)
doc = {
    "libgooglepinyin": ("0.9", "5.6", ("4", "32", "139", "0.15"), ("36", "76"), ("us", "us", "us", "ms"), "us"),
    "libime": ("92", "39", ("0.30", "1.11", "1.34", "2.05"), ("0.98", "2.29"), ("ms",) * 4, "ms"),
    "Rime": ("20", "51", ("0.12", "1.27", "1.89", "2.86"), ("6.1", "25.3"), ("ms",) * 4, "ms"),
    "libpinyin": ("1.2", "14", ("0.70", "1.62", "1.99", "2.37"), None, ("ms",) * 4, None),
}
for name, (load, rss, key, sent, kunits, sunit) in doc.items():
    s = summary(engines[name])
    scale = {"us": 1.0, "ms": 1e-3}
    got_key = [s[f"keystroke.{q}_us"] * scale[u] for q, u in zip(("p50", "p95", "p99", "max"), kunits)]
    line = f"  {name:16s} load {s['open_ms']:.2f} ms  RSS {s['rss_end_kb'] / 1000:.1f} MB  keystroke p50/p95/p99/max " \
           + " / ".join(f"{g:.3g} {u}" for g, u in zip(got_key, kunits))
    line += check(f"{name} load", s["open_ms"], load) + check(f"{name} RSS", s["rss_end_kb"] / 1000, rss)
    for q, g, w in zip(("p50", "p95", "p99", "max"), got_key, key):
        line += check(f"{name} keystroke {q}", g, w)
    if sent:
        got_s = [s[f"fullsent.{q}_us"] * scale[sunit] for q in ("p50", "p99")]
        line += f"  sentence p50/p99 {got_s[0]:.3g} / {got_s[1]:.3g} {sunit}"
        for q, g, w in zip(("p50", "p99"), got_s, sent):
            line += check(f"{name} sentence {q}", g, w)
    print(line)
top3 = {"libime": summary("libime")["in_top3"], "Rime": summary("rime")["in_top3"],
        "libpinyin": summary("libpinyin_nbest")["in_top3"]}
print("  correct sentence in top 3: " + ", ".join(f"{k} {v:.1f}%" for k, v in top3.items())
      + check("libime top3", top3["libime"], "94.9") + check("Rime top3", top3["Rime"], "57.6")
      + check("libpinyin top3", top3["libpinyin"], "71.2"))

# 2. P1: Qwen3-0.6B on the dev set (pre-registered primary configuration)
print("\n== 2. P1 dev-set reranking, pre-registered primary: Qwen3-0.6B, LM score only, all 16-best ==")
p1 = json.load(open(os.path.join(RES, "p1", "rerank_qwen3-0.6b.json"), encoding="utf-8"))
base = [r["libime_ok"] for r in p1["rows"]]
new = [r["rerank_ok"] for r in p1["rows"]]
b, c, p = mcnemar(base, new)
print(f"  libime {sum(base)}/59 -> Qwen3-0.6B {sum(new)}/59 = {100 * sum(new) / 59:.1f}%  fixed {c}  broken {b}  p = {p:.2f}"
      + check("P1 primary (correct, fixed, broken)", (sum(new), c, b), (51, 8, 7)) + check("P1 p", p, "1.0"))
print(f"  rerank ms p50 all candidates {pct(p1['rerank_ms'], .5):.0f}, top 3 {pct(p1['rerank_top3_ms'], .5):.0f}")
bert = json.load(open(os.path.join(RES, "p1", "rerank_bert_model_int8.json"), encoding="utf-8"))["ms"]
print(f"  (exploratory) BERT int8 diff-PLL top 3 rerank ms p50 {pct(bert['diff_top3'], .5):.1f} / p95 {pct(bert['diff_top3'], .95):.1f}"
      + check("BERT dev top3 p50", pct(bert["diff_top3"], .5), "48") + check("BERT dev top3 p95", pct(bert["diff_top3"], .95), "84"))

# 3. confirmation test, 360 sentences
print("\n== 3. Confirmation test, 360 new sentences (PerLTQA zh), pre-registered ==")
d = json.load(open(os.path.join(HERE, "confirm", "confirm_items.json"), encoding="utf-8"))
items = d["items"]
n = len(items)
col = {s: [it["correct"][s] for it in items] for s in d["systems"]}
b0 = col["libime"]
print(f"  B0 libime top-1 (no context)   {sum(b0)}/{n} = {100 * sum(b0) / n:.1f}%" + check("B0", sum(b0), 262))
doc = {"S1": (288, 31, 5, "260", "440"), "S2": (302, 47, 7, "3180", "6670"),
       "S1_noctx": (275, 27, 14, "48", "92"), "S2_noctx": (277, 34, 19, "604", "1306")}
tests = {}
for s, (k_doc, c_doc, b_doc, p50_doc, p95_doc) in doc.items():
    b, c, p = mcnemar(b0, col[s])
    tests[s] = (b, c, p)
    k = sum(col[s])
    p50, p95 = pct(d["ms"][s], .5), pct(d["ms"][s], .95)
    # documented latencies are the 0.1 ms values of confirm_results.txt rounded again (3179.6 -> 3180, 603.5 -> 604):
    # allow half a unit of the last kept digit
    near = lambda x, w: abs(round(x, 1) - float(w)) <= 0.5 * 10 ** (len(w) - len(w.rstrip("0"))) + 1e-9
    print(f"  {s:9s} {k}/{n} = {100 * k / n:.1f}%  fixed {c:2d}  broken {b:2d}  raw p = {p:.2e}  "
          f"rerank ms p50 {p50:.1f} / p95 {p95:.1f}"
          + check(f"{s} (correct, fixed, broken)", (k, c, b), (k_doc, c_doc, b_doc))
          + check(f"{s} p50 ~ {p50_doc}", near(p50, p50_doc), True) + check(f"{s} p95 ~ {p95_doc}", near(p95, p95_doc), True))
prim = sorted(["S1", "S2"], key=lambda k: tests[k][2])
holm, running = {}, 0.0
for rank, k in enumerate(prim):
    running = max(running, min(1.0, (len(prim) - rank) * tests[k][2]))
    holm[k] = running
for k, want in (("S1", "1.3e-05"), ("S2", "4.6e-08")):
    b, c, _ = tests[k]
    verdict = "significantly better" if holm[k] < 0.05 and c > b else "not significant"
    print(f"  PRIMARY {k} vs B0: Holm-adjusted p = {holm[k]:.1e} -> {verdict}" + check(f"Holm {k}", f"{holm[k]:.1e}", want))
doc_src = {"dialogue": ["79.0", "84.0", "89.5", "83.5", "83.0"], "narrative": ["65.0", "75.0", "76.9", "67.5", "69.4"]}
for src, wants in doc_src.items():
    idx = [i for i, it in enumerate(items) if it["source"] == src]
    row = f"  {src} (n={len(idx)}):"
    for s, w in zip(d["systems"], wants):
        v = 100 * sum(col[s][i] for i in idx) / len(idx)
        row += f"  {'B0' if s == 'libime' else s} {v:.1f}%" + check(f"{src} {s}", f"{v:.1f}", w)
    print(row)

if failures:
    print("\nFAIL:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("\nOK: all recomputed numbers match the documented ones.")
