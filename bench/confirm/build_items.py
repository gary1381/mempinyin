"""Sample confirmation-test items (target clause + preceding context) from PerLTQA zh (CC BY-NC 4.0).

Deterministic: seed 20261027; 200 items from dialogues, 160 from event narratives, at most one item per
dialogue/event. Target = a clause of 4-16 chars, all CJK; context = the last <=64 chars before it
(at least 8). Speaker labels are dropped from dialogue turns.
Usage: python build_items.py perltmem.json dev_sentences.tsv items_raw.jsonl [exclude.jsonl seed]
exclude.jsonl: an earlier item set; its dialogues/events and target clauses are not used again (a fresh test set).
"""
import json
import random
import re
import sys

src, dev_path, out_path = sys.argv[1:4]
old = [json.loads(l) for l in open(sys.argv[4], encoding="utf-8")] if len(sys.argv) > 4 else []
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20261027
data = json.load(open(src, encoding="utf-8"))
dev = {l.split("\t")[1].strip() for l in open(dev_path, encoding="utf-8")}
CJK = re.compile(r"^[一-鿿]{4,16}$")
SPLIT = re.compile(r"[，。！？；：、“”‘’（）《》…—,.!?;:()\"'\s~·]+")
SPEAKER = re.compile(r"^[^:：]{1,12}[:：]")


def clause_spans(text):
    """(start, clause) for every clause between punctuation marks."""
    pos = 0
    for part in SPLIT.split(text):
        start = text.find(part, pos) if part else pos
        if part:
            yield start, part
            pos = start + len(part)


def candidates(text):
    return [(s, c) for s, c in clause_spans(text) if CJK.match(c) and s >= 8 and c not in dev]


dialogues, events = [], []
for p_i, person in enumerate(data):
    for d_id, dlg in person["dialogues"].items():
        flat = [u for v in dlg["contents"].values() for t in v for u in (t if isinstance(t, list) else [t])]
        turns = [SPEAKER.sub("", u).strip() for u in flat]
        dialogues.append((f"p{p_i}/d{d_id}", "".join(turns)))
    for e_id, ev in person["events"].items():
        events.append((f"p{p_i}/e{e_id}", ev["content"]))

rng = random.Random(seed)
items, seen, used = [], {o["ref"] for o in old}, {o["id"] for o in old}  # whole keys (until 11-06 "#n" was cut: 18 dialogues ended up in both sets)
for source, pool, want in (("dialogue", dialogues, 200), ("narrative", events, 160)):
    order = list(range(len(pool)))
    rng.shuffle(order)
    got = 0
    for idx in order:
        if got == want:
            break
        uid, text = pool[idx]
        if uid in used:
            continue
        cands = [c for c in candidates(text) if c[1] not in seen]
        if not cands:
            continue
        start, clause = rng.choice(cands)
        seen.add(clause)
        items.append({"id": uid, "source": source, "context": text[max(0, start - 64):start], "ref": clause})
        got += 1

with open(out_path, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(items), {s: sum(i["source"] == s for i in items) for s in ("dialogue", "narrative")})
