# Third-party notices

This repository contains no third-party source code or data files. Its programs call the following software through
public APIs. Each is a dependency that you install separately:

| Dependency | License | Used by |
|---|---|---|
| libime, fcitx5-utils (Fcitx5 project) | LGPL-2.1-or-later | `bench/libime_bench.cpp`, `bench/libime_nbest.cpp` (dynamic linking) |
| librime | BSD-3-Clause | `bench/rime_bench.c` |
| libgooglepinyin | Apache-2.0 | `bench/gpy_bench.c` |
| libpinyin | GPL-3.0-or-later | `bench/lpy_bench.c`, `bench/lpy_nbest.c`. A binary linked against libpinyin is covered by the GPL. The source here stays MIT. |
| onnxruntime | MIT | `bench/rerank_mlm.py`, `bench/confirm/rerank_confirm.py` |
| Hugging Face transformers, PyTorch, NumPy | Apache-2.0, BSD-3-Clause, BSD-3-Clause | Python rerankers |

## Not included

- **PerLTQA** (Chinese; CC BY-NC 4.0) is the source of the 360-item confirmation test set. Rebuild the set from your
  own download with `bench/confirm/build_items.py`. The results stored here keep only per-item correctness flags,
  lengths, timings and item ids.
- **Models:** `bert-base-chinese` (Apache-2.0; ONNX conversion `Xenova/bert-base-chinese`) and Qwen3-0.6B / 1.7B
  (Apache-2.0). Download them from their publishers.
- **Engine data:** libime's `zh_CN.lm` and `sc.dict` and Rime's `luna_pinyin` schema come from the distribution
  packages.
