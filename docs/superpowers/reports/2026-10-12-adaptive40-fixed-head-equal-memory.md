# Adaptive40 与固定分类头的等回放内存正式对照

日期：2026-10-12（Asia/Shanghai）。ISOLET、UPMC Food-101 clean image、CIFAR-100 各以 seed 42/43/44 比较 Adaptive40、固定 Full40、固定 Bias40。固定头共有 18 次正式测试，全部完成、发布并复核。

## 问题与实验身份

这次实验检验：**每类都保留 40 个原始训练样本时，自适应双分支分类头是否优于直接固定使用其中一个分支？** 它与先前 20→40 样本实验回答不同的问题。固定头的每次运行只更改分类头模式（Bias40 还使用对应预先锁定的学习率与步数）和输出路径；种子、任务顺序、数据划分、训练骨干、损失权重 `proto_lambda_a=0.05`、`distill_weight=0.10`、`feat_distill_weight=0.02` 以及每类回放 40 个样本均与同种子 Adaptive40 对齐。

固定头源代码提交为 `65279b855d0ea4a07bafbe4b12703fbb758999d0`，Adaptive40 已发表结果来自 `2d441589d96b55d8cb6b9511f4a227abf52bb8ec`。18 份配置在正式测试前写入[锁定清单](data/adaptive40-fixed-head-locked-manifest-20261011-v2.json)，SHA-256 为 `6d478be4a329eabfa2d7a2d0c86272ef61fce6f655f334d359e688b3b4df1f4a`。两台主机在运行前核对配置、源代码、数据哈希与同种子 Adaptive40 发布结果；运行后再次核对数据哈希和源码工作树。

`AA_final` 为最后一个增量任务结束后、跨已见任务的类增量准确率平均值。表中差值均以同一数据集、同一种子 Adaptive40 为基准，单位为百分点（pp）。

## 逐种子 AA_final

| 数据集 | Seed | Adaptive40 | Full40 | Full−Adaptive | Bias40 | Bias−Adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISOLET | 42 | 91.66% | 91.72% | +0.06 pp | 75.10% | −16.56 pp |
| ISOLET | 43 | 91.08% | 91.08% | 0.00 pp | 74.78% | −16.30 pp |
| ISOLET | 44 | 91.53% | 91.47% | −0.06 pp | 73.63% | −17.90 pp |
| UPMC clean image | 42 | 82.57% | 82.54% | −0.03 pp | 78.58% | −3.99 pp |
| UPMC clean image | 43 | 82.75% | 82.73% | −0.02 pp | 78.37% | −4.38 pp |
| UPMC clean image | 44 | 82.55% | 82.66% | +0.11 pp | 78.45% | −4.10 pp |
| CIFAR-100 | 42 | 37.83% | 37.38% | −0.45 pp | 34.19% | −3.64 pp |
| CIFAR-100 | 43 | 37.54% | 36.50% | −1.04 pp | 34.24% | −3.30 pp |
| CIFAR-100 | 44 | 37.76% | 36.72% | −1.04 pp | 33.94% | −3.82 pp |

## 三种子汇总

下表为均值 ± 样本标准差（SD，n=3）；“配对差”是三个同种子差值的均值 ± 样本 SD。

| 数据集 | Adaptive40 AA_final | Full40 AA_final | Full 配对差 | Bias40 AA_final | Bias 配对差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ISOLET | 91.42% ± 0.30 pp | 91.42% ± 0.32 pp | 0.00 ± 0.06 pp | 74.50% ± 0.77 pp | −16.92 ± 0.86 pp |
| UPMC clean image | 82.62% ± 0.11 pp | 82.64% ± 0.10 pp | +0.02 ± 0.08 pp | 78.47% ± 0.11 pp | −4.16 ± 0.20 pp |
| CIFAR-100 | 37.71% ± 0.15 pp | 36.87% ± 0.46 pp | −0.84 ± 0.34 pp | 34.12% ± 0.16 pp | −3.59 ± 0.26 pp |

ISOLET 与 UPMC 的 Adaptive40 与 Full40 基本相同。CIFAR-100 的 Adaptive40 在三个种子上均高于 Full40，差值为 +0.45/+1.04/+1.04 pp，均值 +0.84 pp。Bias40 在全部九个种子上明显低于 Adaptive40 与 Full40。这是三个种子的描述性比较，不作为显著性结论。

## 遗忘、回放内存与耗时

| 数据集 | Adaptive / Full / Bias 平均 BWT | 每次运行原始回放容量 | Adaptive / Full / Bias 观察到的平均墙钟时间 |
| --- | ---: | ---: | ---: |
| ISOLET | +81.56 / +81.56 / +66.24 pp | 2.45 MiB | 3.1 / 6.0 / 6.4 min |
| UPMC clean image | +66.78 / +66.81 / +62.77 pp | 23.67 MiB | 2.5 / 2.3 / 2.4 min |
| CIFAR-100 | −34.24 / −35.30 / −37.51 pp | 46.88 MiB | 213.8 / 207.5 / 206.5 min |

每次固定头的最终检查点均确认每类 40 个原始回放样本；Full40 与 Bias40 在同一数据集上持久原始回放字节数相同，也与 Adaptive40 的原始回放容量相同。这里的字节数只统计原始样本张量，未包含模型、优化器、事件检查点或运行时显存。墙钟时间由运行日志创建至退出标记的时间计算；训练日志中的逐事件时间也保存在[机器可读台账](data/adaptive40-fixed-head-equal-memory-20261012.json)。主机有并发外部作业，且 Adaptive40 与固定头使用不同代码提交，因此这些耗时不宜解释为分类头的孤立计算开销。

## 完整性与可比性核对

- 两台主机的 ISOLET、UPMC、CIFAR 顺序队列均以退出码 0 结束。18 次运行逐一确认 `exit.code=0`、`FORMAL_STATE_FROZEN.json`、`FORMAL_EVALUATION_PUBLISHED.json`、`results.json` 和最终检查点存在；发布记录中的结果及检查点哈希与文件一致。结束后再次运行正式结果校验器，18 次输出与首次保存的校验结果逐字节一致。
- 固定头最终检查点确认分类头模式、每类 40 个原始回放样本、仅一次最终头拟合、无测试样本参与拟合，且回放数量与类别数匹配。两台源代码工作树保持干净并锁定在上述固定头提交；运行后的数据 SHA-256 与锁定清单及 Adaptive40 对照一致。
- 对每个固定头运行，都直接比较了它与同种子 Adaptive40 的**前一事件训练器状态、最终底部网络状态、原始回放张量、对应 Full 或 Bias 分支的分类头状态**；18 项比较的四类状态均完全相等。这使相同回放容量下的分类头对照比单看配置更可靠。
- 第一版固定头根目录曾在测试集授权前因阶段约束失败，未产生已发布测试；修复后使用全新的 v2 结果根运行。本报告仅使用 v2 正式结果。磁盘紧张时，仅删除了四个已发布 CIFAR 运行中与 `event_9_CIL.pt` 逐字节相同、且发布文件不引用的 `resume_latest.pt`；保留所有正式事件和最终检查点，清理记录在 3080 v2 结果根的 `CLEANUP_DUPLICATE_RESUME_20261012.tsv`。

[机器可读台账](data/adaptive40-fixed-head-equal-memory-20261012.json)记录18次逐种子指标、配对差、配置和发布哈希、GPU、回放字节数、耗时及同种子 Adaptive40 对照。台账 SHA-256：`e2415fee598f4a814e4f163cdcdf36e2540b5af3d5680d7db159cee032ba9c4c`。Adaptive40 相比原每类20个样本的结果见[前一份报告](2026-10-11-adaptive40-three-dataset-aa-final.md)。

结果根：3080 `/home/c3080/YangXiaoXiang/VF-CL/results/adaptive40-fixed-controls-formal-20261011-v2`；4090 `/home/chase/Yangxx/VF-CL/results/adaptive40-fixed-controls-formal-20261011-v2`。

## 研究决策

当前自适应头在 **CIFAR-100** 上显示一致但小幅的等内存优势；在 **ISOLET、UPMC** 上没有观察到相对 Full40 的实质优势。因此，先前 20→40 样本带来的 ISOLET 与 UPMC 提升不能归因于自适应头选择；回放容量翻倍与源代码版本差异仍是解释因素。若目标是提高 ISOLET 准确率，下一步应先分析任务/类别错误与表征、回放覆盖情况，再针对具体瓶颈做一个受控修改。若目标是论证分类头创新，则需要解释并扩大 CIFAR-100 上的稳定优势，同时如实限定其适用范围。
