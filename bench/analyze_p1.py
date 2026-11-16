"""List sentences whose correctness changed after reranking, with libime ranks of the chosen and reference candidates."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
for path in sys.argv[2:]:
    print("==", path)
    for r, x in zip(rows, json.load(open(path, encoding="utf-8"))["rows"]):
        if x["libime_ok"] == x["rerank_ok"]:
            continue
        texts = [c["text"] for c in r["cands"]]
        pos = next(i for i, (a, b) in enumerate(zip(x["rerank"], x["ref"])) if a != b) if not x["rerank_ok"] else "-"
        print("FIXED " if x["rerank_ok"] else "BROKEN", x["ref"], "->", x["rerank"],
              f"chosen_rank={texts.index(x['rerank'])} ref_rank={x['ref_rank_libime']} first_diff_pos={pos}")
