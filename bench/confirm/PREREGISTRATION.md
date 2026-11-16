# 确认测试预先登记

> **Note (English):** original pre-registration, written 2026-10-27 before any reranker was run on the test set. Edited for publication only: a later supplementary registration covering models outside this release was removed, and one sentence about who proposed the rbt3 variant was reworded. Hypotheses, systems and decision rules are unchanged. An English summary is in `docs/DESIGN.md` §4.

写于 2026-10-27。此时测试集已经建好，但还**没有**在测试集上运行任何重排器，也没有看过 libime 在测试集上的 top-1 结果。
（建测试集时只检查了"正确答案是否在 libime 的 64 个候选里"，用来核对拼音，见 `make_pinyin.py`。）

## 测试集

- **来源**：PerLTQA 中文版（`perltmem.json`，CC BY-NC 4.0，2024 年发布，由大模型生成的虚构人物对话和事件叙述）。
  - 它在 bert-base-chinese（2018 年）之后发布，所以不存在训练数据泄漏。
  - 和开发集（`../sentences.tsv` 的 59 句）没有重复。
- **抽样**：`build_items.py`，随机种子 20261027。
  - 200 条来自对话，160 条来自叙述；每个对话或事件最多抽 1 条。
  - 目标是 4–16 个汉字的分句（纯汉字）；上文是它前面最多 64 个字，至少 8 个字。
- **拼音**：`make_pinyin.py`，用 libime 词库自动生成。
  - 抽查 30 条，30 条全对。
  - 360 条全部保留，其中 31 条 libime 的 64 个候选里没有正确答案（几乎都是人名或产品名）。这些对所有系统都算错，不影响配对检验，只拉低绝对准确率。
- **最终 N = 360**，文件是 `testset.jsonl`。

## 系统

所有系统都用同一份 libime N-best：NBest=16，其他参数同 `libime_bench.cpp`，只保留拼写等于输入的整句候选，去重。

| 代号 | 系统 |
|---|---|
| **B0** | libime top-1（基线）。libime 1.1.5 没有传入上文的接口，所以不用上文，和现在的 Fcitx5 一样 |
| **S1** | bert-base-chinese `model_int8.onnx`（Xenova 转换版），**差异 PLL，只重排前 3 名**，只用 MLM 分数（λ=0），**带上文** |
| **S2** | 同 S1，但**重排全部候选**（约 16 个） |

- **差异 PLL**：只遮候选之间不同的位置；如果候选长度不一样，就退回完整 PLL。
- **带上文的输入格式**：`[CLS] 上文 + 候选 + 。 [SEP]`；只遮候选里的字。
- **其他设置**：分数相同时按 libime 原顺序；onnxruntime CPU，2 个线程，`taskset -c 18,19`。
- **开发集结果**（59 句，不带上文，供参考）：S1 配置 93.2%，S2 配置 96.6%，libime 84.7%。

## 主要假设和判定规则

- **H1**：S1 的 top-1 完全正确率高于 B0。
- **H2**：S2 的 top-1 完全正确率高于 B0。
- **检验方法**：配对 McNemar 精确双侧检验，两个检验一起做 Holm 校正，族错误率 α = 0.05。
- **"显著超过"的定义**：Holm 校正后 p < 0.05，并且修好的句子多于弄错的句子。

## 次要结果（只描述，不做显著性结论）

- S1 和 S2 不带上文时的结果，用来看上文的作用；
- 按来源（对话 / 叙述）分别统计；
- 每条的重排耗时 p50 和 p95。

## 和原计划的差别：去掉了 rbt3

原计划把 hfl/rbt3 作为速度变体。运行前在开发集上检查时，它没通过：
- **转换没有问题**：转成 fp32 ONNX 后和 PyTorch 结果一致，最大误差 4e-5；int8 量化会严重损坏它，top-1 和 fp32 只有 47–51% 一致。
- **模型本身的问题**：在 PyTorch 里直接预测被遮住的字也是乱码，例如"今天天[MASK]真好"预测出"寝/雄/有"。这说明 hfl 发布的 rbt3 只适合当微调起点，它的预测字的输出层不能当语言模型用。
- **开发集结果**：重排 16 个候选只有 18.6%，速度也没比 bert-base-chinese int8 快多少（前 3 名 + 差异 PLL：39 ms 对 48 ms）。

所以它不进入确认测试，也不占 Holm 校正的名额。
