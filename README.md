# MemPinyin: benchmarking open-source pinyin engines and reranking libime with context-aware BERT

Reranking the Fcitx5 pinyin engine's (libime) N-best sentences with an int8 `bert-base-chinese` that also sees the
text before the cursor raises whole-sentence top-1 accuracy on 360 new test sentences. Accuracy goes from 72.8% to
80.0% when the top 3 are reranked and to 83.9% when all ~16 candidates are reranked. The test was pre-registered, and
both gains are significant.

Design and full results: [docs/DESIGN.md](docs/DESIGN.md) · [中文简介](README.zh-CN.md)

## Problem

Pinyin input is a decoding problem: one string such as `tamenzhengzaitaolun` maps to many hanzi sentences.
Open-source Linux engines decode with small statistical models:

| Engine | Model |
|---|---|
| libime | KenLM trigram |
| libgooglepinyin | unigram |
| Rime luna_pinyin | word frequencies |

They miss three kinds of case:
- homophones that need meaning (的/得, 响应/相应);
- segmentation ambiguity (`yue` = 月 or 余额);
- context: the text before the cursor is not used at all.

Every wrong first choice costs the user a correction. This project asks:
- Which open engine is the best starting point?
- Can a neural reranker on top of it beat it significantly, and how fast is it?

## Method

1. **Benchmark four open-source engines** (libime, librime, libpinyin, libgooglepinyin) on one machine, each pinned to
   the same idle core. The harnesses measure top-1 accuracy, selection actions, per-keystroke latency and memory
   (`bench/*_bench.c(pp)`). The dev set is 59 synthetic sentences written for this project (`bench/sentences.tsv`).
2. **Take libime's N-best.** `bench/libime_nbest.cpp` dumps full-sentence candidates at NBest = 16.
3. **Rerank with BERT.** `bert-base-chinese` runs as an int8 ONNX model on 2 CPU threads. Candidates are scored by
   pseudo-log-likelihood. *Diff-PLL* masks only the positions where the candidates differ. The input is
   `[CLS] context + candidate + 。 [SEP]`, where *context* is up to 64 characters before the target
   (`bench/confirm/rerank_confirm.py`).
4. **Pre-register.** Hypotheses, systems and the test were written before running on a new 360-item test set
   (`bench/confirm/PREREGISTRATION.md`).
   - Test set: 360 clauses from PerLTQA (Chinese), 200 dialogue and 160 narrative, each with its preceding context.
   - Test: exact two-sided McNemar with Holm correction, α = 0.05.

## Results

### Four open-source engines, 59 dev sentences

| Metric | libgooglepinyin | libpinyin | **libime** | Rime |
|---|---|---|---|---|
| Whole-sentence top-1 | 62.7% (37/59) | 64.4% (38) | **84.7% (50)** | 57.6% (34) |
| Correct sentence in top 3 | — | 71.2% | **94.9%** | 57.6% |
| Actions per sentence (incl. commit) | 2.25 (3 unreachable) | — | **1.31** | 2.34 |
| New words / abbreviations top-1 (24) | 62.5% | — | **100%** | 66.7% |

| Paired comparison (exact McNemar) | only first right | only second right | p |
|---|---|---|---|
| libime vs libgooglepinyin | 16 | 3 | **0.004** |
| libime vs libpinyin | 14 | 2 | **0.004** |
| libime vs Rime | 18 | 2 | **0.0004** |

| Engine | Load | Resident memory | Per keystroke p50 / p95 / p99 / max | Whole sentence p50 / p99 |
|---|---|---|---|---|
| libgooglepinyin | 0.9 ms | 5.6 MB | 4 µs / 32 µs / 139 µs / 0.15 ms | 36 µs / 76 µs |
| libime | 92 ms | 39 MB | 0.30 / 1.11 / 1.34 / 2.05 ms | 0.98 / 2.29 ms |
| Rime | 20 ms (first deploy 0.6 s) | 51 MB | 0.12 / 1.27 / 1.89 / 2.86 ms | 6.1 / 25.3 ms |
| libpinyin | 1.2 ms | 14 MB | 0.70 / 1.62 / 1.99 / 2.37 ms | — |

libime is significantly better than the other three. Its correct sentence is in its top 3 for 94.9% of inputs, 10
points above its top-1: that gap is the room for a reranker.

On the dev set, the pre-registered Qwen3-0.6B reranker did **not** beat libime: 86.4%, fixed 8 and broke 7,
p = 1.0. Exploratory BERT runs looked much better, which led to the confirmation test below (details in
[DESIGN §5.2](docs/DESIGN.md#52-reranking-on-the-dev-set)).

### Confirmation test: 360 new sentences, pre-registered

| System | top-1 | fixed | broken | Holm-adjusted p | verdict |
|---|---|---|---|---|---|
| B0: libime top-1 (no context, as Fcitx5 today) | 72.8% (262/360) | — | — | — | — |
| **S1: BERT int8, rerank top 3, with context** | **80.0% (288)** | 31 | 5 | **1.3e-5** | **significantly better** |
| **S2: BERT int8, rerank all ~16 candidates, with context** | **83.9% (302)** | 47 | 7 | **4.6e-8** | **significantly better** |

| System | top-1 | fixed / broken | rerank time p50 / p95 (2 CPU threads) |
|---|---|---|---|
| S1 with context | 80.0% | 31 / 5 | 260 / 440 ms |
| S1 without context | 76.4% | 27 / 14 | 48 / 92 ms |
| S2 with context | 83.9% | 47 / 7 | 3180 / 6670 ms |
| S2 without context | 76.9% | 34 / 19 | 604 / 1306 ms |

The rows without context are secondary, descriptive results with no significance claim.

- **Context matters, mostly by preventing damage.** S2 breaks 19 items without context and 7 with it.
- **Candidate recall is the ceiling.** Only 322 of 360 references are in libime's 16-best, mostly missing names.
  Within those 322, S2 is right 302 times (93.8%).
- **The two baselines come from the same engine.** 72.8% here and 84.7% on the dev set are the same libime with the
  same settings; the test text is harder, with more names and a more written style. All gains are paired against
  libime on the same items.

## Reproduce

### Recompute every table above (no engines, no models)

```bash
python3 bench/reproduce_tables.py
```

The script uses the standard library only. It recomputes the following from the stored per-sentence outputs in
`bench/results/` and `bench/confirm/confirm_items.json`:
- accuracy;
- fixed/broken counts;
- exact McNemar p-values and the Holm correction;
- per-source rows;
- rerank-time percentiles.

It also reads the engines' latency and memory summaries, then checks everything against the documented numbers. It
exits non-zero on any mismatch.

### Re-run from scratch (Linux x86-64; the original run used Ubuntu 24.04)

1. **Engines.** Install `libimecore-dev libimepinyin-dev libfcitx5utils-dev libime-bin libime-data librime-dev
   librime-data libboost-dev` from the distribution. Get libgooglepinyin and libpinyin from their upstream releases.
   The tested versions are libime 1.1.5, librime 1.10.0, libpinyin 2.6.1 and libgooglepinyin 0.1.2.
2. **Four-engine benchmark** (in `bench/`):
   ```bash
   L=/usr/lib/x86_64-linux-gnu; I="-I/usr/include/LibIME -I/usr/include/Fcitx5/Utils"
   g++ -O2 -std=c++17 $I -o libime_bench libime_bench.cpp -lIMEPinyin -lIMECore -lFcitx5Utils
   g++ -O2 -std=c++17 $I -o libime_nbest libime_nbest.cpp -lIMEPinyin -lIMECore -lFcitx5Utils
   gcc -O2 -o rime_bench rime_bench.c -lrime
   taskset -c 1 ./libime_bench $L/libime/zh_CN.lm /usr/share/libime/sc.dict sentences.tsv words.tsv
   taskset -c 1 ./rime_bench /usr/share/rime-data ./rime_user luna_pinyin_simp sentences.tsv words.tsv
   ```
   `gpy_bench.c`, `lpy_bench.c` and `lpy_nbest.c` build the same way against libgooglepinyin and libpinyin. Each
   declares the C API it uses at the top of the file, and you link with `-Wl,-rpath` to the library.
3. **Models** (not included):
   ```bash
   pip install onnxruntime transformers numpy torch huggingface_hub
   hf download Xenova/bert-base-chinese --local-dir models/bert-base-chinese-onnx   # onnx/model_int8.onnx + tokenizer
   ```
   This is a community ONNX conversion of `google-bert/bert-base-chinese` (Apache-2.0). The P1 baseline
   `bench/rerank_lm.py` loads `Qwen/Qwen3-0.6B` (Apache-2.0) from the Hugging Face Hub.
4. **Dev-set reranking:**
   ```bash
   ./libime_nbest $L/libime/zh_CN.lm /usr/share/libime/sc.dict sentences.tsv 16 > nbest_16.jsonl
   taskset -c 1,2 python3 rerank_lm.py nbest_16.jsonl Qwen/Qwen3-0.6B results/p1/rerank_qwen3-0.6b.json
   taskset -c 1,2 python3 rerank_mlm.py nbest_16.jsonl ../models/bert-base-chinese-onnx model_int8.onnx results/p1/rerank_bert_model_int8.json
   ```
5. **Confirmation test.** The test set is not redistributed (see below). Download PerLTQA (Chinese,
   `Dataset/zh/perltmem.json`) from its authors, then rebuild the same 360 items:
   ```bash
   cd bench/confirm
   libime_pinyindict -d /usr/share/libime/sc.dict sc_dict.txt
   python3 build_items.py /path/to/perltmem.json ../sentences.tsv items_raw.jsonl
   python3 make_pinyin.py sc_dict.txt items_raw.jsonl ../libime_nbest $L/libime/zh_CN.lm /usr/share/libime/sc.dict testset.jsonl
   python3 -c 'import json; [print(j["py"], j["ref"], sep="\t") for j in map(json.loads, open("testset.jsonl"))]' > testset.tsv
   ../libime_nbest $L/libime/zh_CN.lm /usr/share/libime/sc.dict testset.tsv 16 > nbest_test.jsonl
   taskset -c 1,2 python3 rerank_confirm.py testset.jsonl nbest_test.jsonl ../../models/bert-base-chinese-onnx model_int8.onnx confirm_results.json
   ```
   `confirm_items.json` stores each item's PerLTQA id (`perltqa_id`), so a rebuilt set can be checked item by item.

## Limitations

- **Offline evaluation only.** Stored libime N-best lists are reranked by a Python script. The reranker is not
  integrated into an input method, and end-to-end latency was not measured. B0 uses no context because libime 1.1.5
  has no API for it.
- **Latency is far from interactive.** With context, S1 takes p50 260 ms and S2 p50 3.2 s per item on 2 CPU threads,
  against a ~20 ms target. Without context, top-3 reranking takes p50 48 ms. [DESIGN §8](docs/DESIGN.md#8-whats-next)
  lists the next steps.
- **The test sets are small.**
  - The dev set has 59 sentences (±12-point CI) and was reused for exploration, so only its engine comparison and
    the pre-registered P1 result count as evidence.
  - The confirmation set has 360 clauses of LLM-written fictional text, with more names than everyday typing.
  - Pinyin was generated automatically, with 30 items spot-checked.
  - No engine had user history.
- **Recall caps reranking.** 38 of 360 references are not in libime's 16-best.
- **Engine settings.** Rime was tested only with the distribution's `luna_pinyin_simp`, without a grammar model;
  libime used its default settings.

## Repository layout

| Path | What |
|---|---|
| `bench/libime_bench.cpp`, `rime_bench.c`, `lpy_bench.c`, `lpy_nbest.c`, `gpy_bench.c` | four-engine benchmark harnesses |
| `bench/libime_nbest.cpp` | libime full-sentence N-best dumper |
| `bench/sentences.tsv`, `bench/words.tsv` | 59-sentence dev set and 24 new words / abbreviations |
| `bench/P1_preregistration.md`, `bench/rerank_lm.py`, `bench/rerank_mlm.py`, `bench/analyze_p1.py` | dev-set reranking (pre-registered Qwen3 P1; exploratory BERT) |
| `bench/results/` | raw engine outputs (`results_*.txt`) and dev-set reranking outputs (`p1/`) |
| `bench/confirm/` | confirmation-test pre-registration, test-set builder, pinyin generator, BERT reranker, per-item results |
| `bench/reproduce_tables.py` | recomputes all tables from stored outputs |
| `docs/DESIGN.md` | design, method, results, limitations, next steps |

The pre-registrations are the Chinese originals, written before the runs. `docs/DESIGN.md` §4 summarizes them in
English.

## Data and licenses

- **Code:** MIT, see [LICENSE](LICENSE). The programs link dynamically against libime and fcitx5-utils (LGPL-2.1+),
  librime (BSD-3), libgooglepinyin (Apache-2.0) and libpinyin (GPL-3.0+). No source from them is included. A binary
  you build against libpinyin is covered by the GPL. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Dev set** (`bench/sentences.tsv`, `bench/words.tsv`): written for this project, MIT.
- **PerLTQA** (CC BY-NC 4.0) is not redistributed. The stored results keep only per-item correctness flags, lengths,
  timings and item ids. DESIGN.md quotes a few short fragments as error examples.
- **Models** are not included. bert-base-chinese and Qwen3 are Apache-2.0.

---

Built with AI coding agents; experiment design, evaluation and review by Gary Xiao.
