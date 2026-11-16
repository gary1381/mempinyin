# MemPinyin：开源拼音引擎评测，以及带上文的 BERT 重排

用 int8 的 `bert-base-chinese`，结合光标前的上文，对 Fcitx5 拼音引擎（libime）的整句候选重新排序。在 360 句新测试集上，整句首选正确率从 72.8% 提到 80.0%（只重排前 3 名）和 83.9%（重排全部约 16 个候选）。这次测试预先登记，两项提升都显著。

English: [README.md](README.md) · 设计与完整结果（英文）：[docs/DESIGN.md](docs/DESIGN.md)

## 做法

1. **四个开源引擎同机同核实测**：libime、librime、libpinyin、libgooglepinyin。测准确率、选词次数、每键延迟和内存（`bench/*_bench.c(pp)`）。开发集是 59 句为本项目写的合成句（`bench/sentences.tsv`）。
2. **libime 出 N-best**：NBest=16（`bench/libime_nbest.cpp`）。
3. **BERT 重排**：`bert-base-chinese` int8 ONNX，2 个 CPU 线程。打分用差异伪对数似然（只遮候选之间不同的字）。输入是 `[CLS] 上文 + 候选 + 。 [SEP]`（`bench/confirm/rerank_confirm.py`）。
4. **预先登记的确认测试**：运行前写好假设和检验（`bench/confirm/PREREGISTRATION.md`）。测试集是从 PerLTQA 中文版抽的 360 个分句，每句带上文。检验用 McNemar 精确检验和 Holm 校正。

## 主要结果

| 系统 | top-1 | 修好 | 弄错 | Holm 校正后 p |
|---|---|---|---|---|
| B0：libime top-1（不用上文） | 72.8%（262/360） | — | — | — |
| **S1：BERT int8，前 3 名，带上文** | **80.0%（288）** | 31 | 5 | **1.3e-5** |
| **S2：BERT int8，全部约 16 个候选，带上文** | **83.9%（302）** | 47 | 7 | **4.6e-8** |

- **耗时**：每条重排的耗时（2 线程，p50 / p95）如下：S1 带上文 260 / 440 ms；S2 带上文 3180 / 6670 ms；S1 不带上文 48 / 92 ms。
- **四引擎对比**（59 句开发集）：libime 84.7%（50/59），libgooglepinyin 62.7%，libpinyin 64.4%，Rime 57.6%。libime 显著优于另外三个，配对 McNemar 的 p 分别是 0.004、0.004、0.0004。

## 复现

- 不需要引擎和模型，可以从存档的逐句结果重算所有表：`python3 bench/reproduce_tables.py`。
- 从头复现的依赖、模型下载和命令见 [README.md](README.md#reproduce)。

## 局限

- 结果来自离线脚本，没有接进输入法。
- 带上文时延迟离 20 ms 的目标还很远。
- 开发集只有 59 句。测试集 360 句，是大模型写的虚构文本。

## 许可

代码采用 MIT（见 [LICENSE](LICENSE)）。PerLTQA（CC BY-NC 4.0）和模型都不随仓库发布。第三方依赖见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

Built with AI coding agents; experiment design, evaluation and review by Gary Xiao.
