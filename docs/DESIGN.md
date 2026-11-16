# MemPinyin: design and results

This document covers how four open-source pinyin engines compare, and whether a context-aware BERT reranker on top of
the best one (libime) measurably improves whole-sentence accuracy. All numbers are reproduced by
`python3 bench/reproduce_tables.py` from the stored outputs in `bench/results/` and `bench/confirm/`.

## 1. Problem

A pinyin input method turns a letter string such as `tamenzhengzaitaolun` into hanzi. Every syllable has many
characters and the letters can be split in more than one way, so the engine has to decode a sentence. Every wrong
first choice costs the user extra keystrokes to correct.

Open-source Linux engines decode with small statistical models. They fail in three predictable ways:

- **Homophones that need meaning or grammar:** 的/得, 响应/相应, 超支/超值.
- **Segmentation ambiguity:** `yue` can be 月 or 余额; `anhuichejian` can be 按回车键 or 安徽车间.
- **No context:** the engines decode the current input alone. The text already in the document, before the cursor,
  is ignored, so a sentence-initial 他/她 or 买/卖 is a guess.

Two questions follow:

1. Which open engine is the best starting point?
2. Can a neural reranker on its N-best list beat it significantly, and how fast?

## 2. Engines compared

| Engine | What it is | License | Version tested | Model |
|---|---|---|---|---|
| **libime** | pinyin engine of Fcitx5 (fcitx5-chinese-addons) | LGPL-2.1+ | 1.1.5 | KenLM trigram + segmentation graph, beam Viterbi, N-best |
| **librime** | Rime engine, schema `luna_pinyin_simp` | BSD-3 | 1.10.0 | word frequencies; no grammar model installed |
| **libpinyin** | engine behind ibus-libpinyin | GPL-3.0+ | 2.6.1 | n-gram with default dictionary |
| **libgooglepinyin** | core of the 2009 Android PinyinIME | Apache-2.0 | 0.1.2 | unigram over ~65k words, 40-letter limit |

Setup:
- All four ran on the same x86-64 Linux machine, pinned to the same idle core (`taskset`).
- None had any user learning history.
- libime used the fcitx5-chinese-addons defaults (NBest 2, beam 20, frame 10) plus the fuzzy options Inner,
  InnerShort and PartialFinal.
- libpinyin used `IS_PINYIN | PINYIN_INCOMPLETE | DYNAMIC_ADJUST`.

The harnesses are `bench/libime_bench.cpp`, `bench/rime_bench.c`, `bench/lpy_bench.c` (+ `lpy_nbest.c` for top-3)
and `bench/gpy_bench.c`. For every sentence and word they report:
- top-1 exact match;
- the rank of the reference;
- selection actions;
- per-keystroke latency (including fetching the first 10 candidates);
- whole-sentence latency;
- load time and resident memory.

**Dev set** (`bench/sentences.tsv`): 59 synthetic sentences written for this project, covering daily life, tech,
places and names, idioms and polyphones. `bench/words.tsv` adds 24 new words and abbreviations.

## 3. Method: libime N-best + BERT reranking

1. **Candidates.** `bench/libime_nbest.cpp` dumps libime's full-sentence candidates with their scores, using
   NBest = 16 and otherwise the same settings. Only candidates that spell the whole input are kept, deduplicated, in
   libime's order. On the dev set this gives about 17 candidates per sentence, and every reference is among them
   (56/59 with NBest = 2). Decoding takes p50 1.08 ms.
2. **Scorer.** `bert-base-chinese` (Apache-2.0, ~100M parameters, 12 layers, one token per character) runs as the
   community int8 ONNX conversion (`Xenova/bert-base-chinese`, `model_int8.onnx`). It runs in onnxruntime on 2 CPU
   threads.
3. **Score = pseudo-log-likelihood (PLL).** BERT is bidirectional, so it cannot give a left-to-right sentence
   probability. Instead each candidate character is masked in turn and the log-probabilities of recovering it are
   summed.
   - *Full PLL* masks every character.
   - *Diff PLL* masks only the positions where the reranked candidates differ, which is much faster. It falls back
     to full PLL when the candidates differ in length.
   - All masked copies for one item go in a single batched ONNX call.
4. **Context.** The input is `[CLS] context + candidate + 。 [SEP]`. *Context* is up to 64 characters of the text
   before the target. Only candidate characters are masked.
5. **Choice.** The candidate with the highest MLM score wins (λ = 0, no libime score). Ties keep libime's order.

Two scripts implement this:
- `bench/rerank_mlm.py`: the dev-set version, without context.
- `bench/confirm/rerank_confirm.py`: the confirmation-test version, with and without context.

A causal-LM baseline, `bench/rerank_lm.py`, scores *log P(candidate + 。)* with Qwen3.

## 4. Pre-registration

The 59 dev sentences were used to look at errors and pick configurations, so they cannot also be the evidence. Two
experiments were registered in writing before their results were seen. The Chinese originals are in the repo.

**P1, dev set** (`bench/P1_preregistration.md`):
- Primary configuration: Qwen3-0.6B, fp32, CPU 2 threads, LM score only, all 16-best.
- Primary metric: top-1 vs libime's 50/59, exact two-sided McNemar, p < 0.05.
- Other settings were declared exploratory: Qwen3-1.7B, top-3 only, and LM + λ·libime with λ ∈ {0.25, 0.5, 1}, all
  of them reported.

**Confirmation test** (`bench/confirm/PREREGISTRATION.md`):
- **Test set.** 360 clauses from PerLTQA (Chinese; CC BY-NC 4.0; LLM-written fictional dialogues and narratives,
  released in 2024, after bert-base-chinese was trained). They were sampled by `confirm/build_items.py` with seed
  20261027:
  - 200 from dialogues and 160 from narratives, at most one per dialogue or event;
  - targets are 4–16 hanzi (mean 10.5), each with up to 64 characters of preceding context (mean 59);
  - no overlap with the dev set.
- **Pinyin** was generated from libime's dictionary by `confirm/make_pinyin.py`. A spot check of 30 items found all 30
  correct. While building the set, the only check made was whether the reference is reachable in libime's 64-best.
  libime's ranking on the test set was not looked at.
- **Systems** all share one libime 16-best list (mean 16.9 candidates, decoding p50 1.5 ms):
  - **B0:** libime top-1. No context, because libime 1.1.5 has no API for it. This is what Fcitx5 does today.
  - **S1:** BERT int8, diff PLL, rerank the top 3 only, MLM score only, with context.
  - **S2:** the same as S1 but reranking all candidates.
- **Hypotheses:** H1: S1 > B0; H2: S2 > B0 on top-1 exact match.
- **Test:** exact two-sided McNemar on the paired items, with Holm correction over H1 and H2, family-wise
  α = 0.05.
- **"Significantly better"** was defined as Holm-adjusted p < 0.05 *and* more items fixed than broken.
- **Secondary results** (descriptive only): S1/S2 without context, a per-source breakdown, and rerank time p50/p95.
- A 3-layer distilled model (hfl/rbt3) was dropped before the test. Its masked-LM head is not usable as a language
  model (it predicts noise even in PyTorch), so it took no slot in the Holm correction.

## 5. Results

### 5.1 Four engines (dev set, 59 sentences)

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
| the other three, pairwise | — | — | 0.45–1.0 |

| Engine | Load | Resident memory | Per keystroke p50 / p95 / p99 / max | Whole sentence p50 / p99 |
|---|---|---|---|---|
| libgooglepinyin | 0.9 ms | 5.6 MB | 4 µs / 32 µs / 139 µs / 0.15 ms | 36 µs / 76 µs |
| libime | 92 ms | 39 MB | 0.30 / 1.11 / 1.34 / 2.05 ms | 0.98 / 2.29 ms |
| Rime | 20 ms (first deploy 0.6 s) | 51 MB | 0.12 / 1.27 / 1.89 / 2.86 ms | 6.1 / 25.3 ms |
| libpinyin | 1.2 ms | 14 MB | 0.70 / 1.62 / 1.99 / 2.37 ms | — |

**libime is the best open-source starting point.**
- It is significantly better than the other three, and the other three cannot be told apart.
- All four engines take under 3 ms per keystroke in the worst case, so decoding speed is not what users notice.
  Corrections are.
- libime's correct sentence is in its top 3 for 94.9% of inputs, 10 points above its top-1. That gap is the room for
  a reranker.
- With n = 59, a single accuracy carries a 95% CI of roughly ±12 points.

**Errors.**
- libgooglepinyin makes 22 errors: 14 from missing context, 5 segmentation, 2 missing dictionary words and 1 from the
  40-letter cap.
- Rime makes the same kinds of errors.
- libime makes only 9 errors: 7 that need context (发展的→得, 相应→响应, 超值→超支, 及其→机器, 合理死→和李四, 最长的和→河,
  狠辣→很辣), 1 segmentation (握住在→我住在) and 1 variant character (帐户/账户).
- Six sentences defeat all four engines. They need grammar (动词 + 得 + 补语), collocation semantics (预算…超支,
  火锅很辣), world knowledge (长江 is a 河) or name recognition. A trigram cannot see these, which motivates a
  whole-sentence neural reranker.

### 5.2 Reranking on the dev set

**P1, pre-registered (Qwen3-0.6B).**
- It does not significantly beat libime: 86.4% (51/59), fixed 8 and broke 7, p = 1.0.
- Every sentence it broke was changed in the first one or two characters, where there is no left context. It
  promoted rare characters libime ranks low (他→祂, 我→喔, 把→吧).
- Qwen3-1.7B was no better (84.7%).
- Rerank time on 2 threads was p50 1.10 s for 16 candidates and 0.28 s for the top 3.

**BERT int8, exploratory** (not pre-registered; 11 configurations tried on an already-analysed dev set, so the
p-values are signals only):

| Configuration (no context) | top-1 | fixed | broken | p (indicative) | rerank p50 (2 threads) |
|---|---|---|---|---|---|
| libime baseline | 84.7% | — | — | — | — |
| MLM only, all 16, full PLL | 98.3% | 9 | 1 | 0.022 | 1111 ms |
| MLM only, all 16, diff PLL | 96.6% | 9 | 2 | 0.065 | 625 ms |
| MLM only, top 3, diff PLL | 93.2% | 6 | 1 | 0.125 | **48 ms** (p95 84 ms) |

Bidirectional scoring is the difference:
- It uses the following characters: 发展__非常快 → 得.
- It almost never rewrites the sentence start.
- It fixed all 9 of libime's errors.

A q4f16 ONNX variant was within one sentence of int8 but 2.3× slower on CPU, so the CPU choice is int8. These
results set up the confirmation test below.

### 5.3 Confirmation test (360 new sentences, pre-registered)

| System | top-1 | fixed | broken | Holm-adjusted p | verdict |
|---|---|---|---|---|---|
| B0: libime top-1 (no context) | 72.8% (262/360) | — | — | — | — |
| **S1: BERT int8, top 3, with context** | **80.0% (288)** | 31 | 5 | **1.3e-5** | **significantly better** |
| **S2: BERT int8, all ~16 candidates, with context** | **83.9% (302)** | 47 | 7 | **4.6e-8** | **significantly better** |

**Both hypotheses hold:** +7.2 points with the top 3 and +11.1 points with all candidates.

Secondary results (descriptive):

| System | top-1 | fixed / broken | rerank p50 / p95 |
|---|---|---|---|
| S1 with context | 80.0% | 31 / 5 | 260 / 440 ms |
| S1 without context | 76.4% | 27 / 14 | 48 / 92 ms |
| S2 with context | 83.9% | 47 / 7 | 3180 / 6670 ms |
| S2 without context | 76.9% | 34 / 19 | 604 / 1306 ms |

| Source | B0 | S1 | S2 | S1 no context | S2 no context |
|---|---|---|---|---|---|
| Dialogue (200) | 79.0% | 84.0% | 89.5% | 83.5% | 83.0% |
| Narrative (160) | 65.0% | 75.0% | 76.9% | 67.5% | 69.4% |

What the numbers say:
- **Context matters, mostly by preventing damage.** Without context S2 breaks 19 items; with context it breaks 7,
  and the number fixed rises from 34 to 47.
- **He/she/it is part of it, not most of it.** Of the 47 items S2 fixes, 10 differ only in 他/她/它, and so do 3 of
  the 7 it breaks. Counting 他/她/它 as one character, B0 is 76.9% and S2 is 86.7%, a +9.8-point gain, about the same.
  The other fixes are real character choices, for example a name mentioned in the context.
- **Candidate recall is the ceiling.** The reference is in libime's 16-best for only 322 of 360 items (89.4%). Most
  misses are person and product names. Within those 322, S2 picks correctly 302 times (93.8%) and S1 288 times.
- **Remaining errors:**
  - 的/得 (坚定地, 做得);
  - 吗/嘛;
  - homophonous names (李明→黎明);
  - 他/她 that the context cannot settle.
- **72.8% vs 84.7%.** These come from the same libime with the same settings, on different text.
  - The dev set is short common sentences (95% CI ~74–92%).
  - The test set is LLM-written prose with many names (95% CI ~68–77%).
  - Excluding the 38 unreachable items, libime gets 81.4% (262/322).
  - Every gain above is a paired comparison against libime on the same items.

## 6. Latency

| Path | Time per item (2 CPU threads) |
|---|---|
| libime decode, 16-best | p50 1.5 ms |
| BERT int8, top 3, diff PLL, no context | p50 48 ms, p95 92 ms |
| BERT int8, top 3, diff PLL, **with context** (S1) | p50 260 ms, p95 440 ms |
| BERT int8, all candidates, **with context** (S2) | p50 3180 ms, p95 6670 ms |

The target for interactive use is about 20 ms per keystroke, and both registered systems are far above it. Context
makes each masked copy about 70 tokens instead of about 12. And because BERT is bidirectional, the context has to be
re-encoded for every masked copy.

## 7. Limitations

- **This is an offline evaluation.** Stored N-best lists are reranked by a Python script. It is not integrated into
  an input method, and there is no end-to-end latency (app → candidate window).
- **The test sets are small and specific.**
  - The dev set has 59 hand-written sentences and was used for exploration.
  - The confirmation set has 360 clauses of LLM-generated fictional text, with more names than everyday typing. It
    may favour language models.
  - Pinyin was generated automatically, and only 30 items were spot-checked.
  - No engine had user history. Real use with learning would shift every number.
- **Engine coverage.** Rime was tested only with the distribution's `luna_pinyin_simp` and no grammar model.
  libime ran at default settings with cloud pinyin off.
- **Recall cap.** 38 of 360 references are not in libime's 16-best, so no reranker can fix them.
- **Latency.** 260 ms (S1) and 3.2 s (S2) per item on 2 CPU threads are not usable while typing.

## 8. What's next

- **Speed, by order of expected payoff:**
  - encode the context once per commit, with candidate-only attention to cached context states, or a causal model
    with a KV cache;
  - shorter context (16–32 characters instead of 64);
  - rerank only when the user pauses (~150 ms without a key) and replace the first candidate only when it is stable;
  - distil a 3–6-layer Chinese student from bert-base-chinese on libime N-best errors. Simply trimming a multilingual
    model's vocabulary was fast but lost too much Chinese.
- **Recall:** a larger N-best or lattice reranking, plus name and new-word handling, to reach the 38 unreachable
  items.
- **Integration:** pass the text before the cursor (Fcitx5/IBus SurroundingText) to the reranker inside a real input
  method and measure end to end.
- **A fresh test** on a new, independently built set for every configuration change, pre-registered the same way.
