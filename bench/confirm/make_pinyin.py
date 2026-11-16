"""Hanzi -> toneless pinyin for test refs, using libime's own dictionary dump (sc_dict.txt).

1. Segment each ref into dictionary words (longest words preferred) and take each word's reading.
2. Characters outside multi-char words use their most common reading across the dictionary,
   with a short override list for function characters.
3. Reachability: the ref must appear among libime's full-sentence N-best (NBest=64) for that pinyin.
   If not, try switching one polyphonic character at a time to its other readings (forced alignment).
Only reachability is checked here; libime's top-1 ranking is never looked at.
Usage: python make_pinyin.py sc_dict.txt items_raw.jsonl ./libime_nbest lm sc.dict testset.jsonl [--sylls]
"""
import collections
import json
import subprocess
import sys
import tempfile

dict_path, items_path, nbest_bin, lm, scdict, out_path = sys.argv[1:7]
KEEP_SYLLS = "--sylls" in sys.argv  # also write "p": the syllables, space-separated
words = collections.defaultdict(list)
char_counts = collections.defaultdict(collections.Counter)
for line in open(dict_path, encoding="utf-8"):
    parts = line.split()
    if len(parts) < 2:
        continue
    w, py = parts[0], parts[1]
    sylls = py.split("'")
    if len(sylls) != len(w):
        continue
    words[w].append(py)
    if len(w) > 1:
        for ch, s in zip(w, sylls):
            char_counts[ch][s] += 1
    else:
        char_counts[ch := w][py] += 0  # make sure every single-char reading is known

OVERRIDE = {"的": "de", "了": "le", "着": "zhe", "地": "de", "得": "de", "还": "hai", "都": "dou", "和": "he",
            "为": "wei", "长": "chang", "行": "xing", "便": "bian", "只": "zhi", "没": "mei", "大": "da",
            "种": "zhong", "少": "shao", "乐": "le", "重": "zhong", "觉": "jue", "处": "chu", "要": "yao",
            "好": "hao", "会": "hui", "那": "na", "哪": "na", "呢": "ne", "吧": "ba", "啊": "a", "个": "ge",
            "中": "zhong", "一": "yi", "不": "bu", "与": "yu", "给": "gei", "看": "kan", "发": "fa", "相": "xiang",
            "将": "jiang", "更": "geng", "间": "jian", "当": "dang", "差": "cha", "数": "shu", "么": "me", "什": "shen"}


def char_reading(ch):
    if ch in OVERRIDE:
        return OVERRIDE[ch]
    c = char_counts.get(ch)
    return c.most_common(1)[0][0] if c else None


def word_reading(w):
    # among a word's dictionary readings, prefer the one whose syllables are most common for each char
    return max(words[w], key=lambda py: sum(char_counts[ch][s] for ch, s in zip(w, py.split("'"))))


def to_pinyin(text):
    """Return list of syllables, or None if a char is unknown. DP maximizes sum(len(word)^2)."""
    n = len(text)
    best = [(-1, None)] * (n + 1)
    best[0] = (0, None)
    for i in range(n):
        if best[i][0] < 0:
            continue
        for j in range(i + 1, min(n, i + 8) + 1):
            w = text[i:j]
            if (j - i == 1 and char_reading(w)) or (j - i > 1 and w in words):
                score = best[i][0] + (j - i) ** 2
                if score > best[j][0]:
                    best[j] = (score, i)
    if best[n][0] < 0:
        return None
    sylls, j = [], n
    while j > 0:
        i = best[j][1]
        w = text[i:j]
        sylls[:0] = [char_reading(w)] if j - i == 1 else word_reading(w).split("'")
        j = i
    return sylls


def reachable(pairs):
    """pairs: list of (pinyin, ref) -> list of bool (ref among libime full-sentence N-best 64)."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", encoding="utf-8", delete=False) as f:
        f.writelines(f"{py}\t{ref}\n" for py, ref in pairs)
    out = subprocess.run([nbest_bin, lm, scdict, f.name, "64"], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    return [any(c["text"] == json.loads(l)["ref"] for c in json.loads(l)["cands"]) for l in out]


items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
status = collections.Counter()
kept = []
for it in items:
    s = to_pinyin(it["ref"])
    if s is None:
        status["unknown_char_dropped"] += 1
        continue
    it["sylls"] = s
    kept.append(it)

ok = reachable([("".join(it["sylls"]), it["ref"]) for it in kept])
todo = []
for it, r in zip(kept, ok):
    it["pinyin_status"] = "default" if r else "unreachable"
    if not r:
        for pos, ch in enumerate(it["ref"]):
            for alt in char_counts.get(ch, {}):
                if alt != it["sylls"][pos]:
                    todo.append((it, it["sylls"][:pos] + [alt] + it["sylls"][pos + 1:]))
if todo:
    ok2 = reachable([("".join(s), it["ref"]) for it, s in todo])
    for (it, s), r in zip(todo, ok2):
        if r and it["pinyin_status"] == "unreachable":
            it["sylls"], it["pinyin_status"] = s, "aligned"
with open(out_path, "w", encoding="utf-8") as f:
    for it in kept:
        sylls = it.pop("sylls")
        it["py"] = "".join(sylls)
        if KEEP_SYLLS:
            it["p"] = " ".join(sylls)
        status[it["pinyin_status"]] += 1
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(kept), dict(status))
for it in kept:
    if it["pinyin_status"] != "default":
        print(it["pinyin_status"], it["ref"], it["py"])
