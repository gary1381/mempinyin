# P1 主方案（先写下，再运行）

> **Note (English):** original pre-registration of the dev-set reranking experiment (Qwen3-0.6B), written 2026-10-27 before any reranking result was seen; unchanged. English summary: `docs/DESIGN.md` §4.

写于 2026-10-27，在看到任何重排结果之前。

- **候选**：libime NBest=16（其余参数同 `libime_bench.cpp`），只保留拼写等于整串输入的整句候选，去重，保持 libime 原顺序。
- **模型**：Qwen3-0.6B（四个可用模型里最小、最接近延迟目标），fp32，CPU 2 线程。
- **打分**：log P(候选 + "。" | `<|endoftext|>`)，即整句加句号的总对数概率；不用 libime 分数，不调任何参数。
- **选择**：取分数最高的候选；分数相同时按 libime 原顺序。
- **主指标**：59 句整句 top-1 完全正确率，与 libime top-1（50/59 = 84.7%）做配对 McNemar 精确双侧检验，显著性阈值 p < 0.05。
- **事先说明的统计功效**：libime 只错 9 句。最好情况是修好 9 句、不弄错任何一句，此时 p = 0.004；修好 6 句、弄错 0 句时 p = 0.031；修好 6 句、弄错 1 句时 p = 0.125。也就是说，**只有几乎不弄错原本正确的句子，才可能显著**。
- **探索项（只报告、不据此下结论）**：Qwen3-1.7B；只重排前 3 名；LM 分数与 libime 分数加权（λ ∈ {0.25, 0.5, 1}，全部报告，不挑最好的）。
