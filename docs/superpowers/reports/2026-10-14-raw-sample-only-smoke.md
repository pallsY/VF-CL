# 仅原始样本回放实现试跑

4090 GPU 被外部作业占用时，先在同机 CPU 做 ISOLET seed54、每任务 1 epoch 的**训练专用实现试跑**。这不是六组正式开发矩阵中的读数，更不是正式测试结果。前两个 GPU 等待器均在任何训练/读数出现前安全停止；v1 源码的留出校验仍强制要求原型，已在源码提交 `99e1da7e05e7eeb8b57587b6d3a75c6eaa490066` 修复。未碰外部作业。

最终 v3 CPU 试跑配置 SHA-256 为 `afa604eddbe9e3c1680eb2e457cb7013fbbe84d320f780e97a5db0cca5eddb8b`。运行退出码 0、冻结状态和开发留出读数均存在；1,040 个留出样本的 Class-IL 为 0.882692、Task-IL 为 0.979808。最终检查点有 26 类、每类 20 个原始样本，`global_protos` 与 `prev_protos` 均为空，原型字节数 0。数据流审计没有正式测试 loader，且无正式测试发布/消耗或 `results.json`。

完整 49 文件结果保存在本机 `D:\VF-CL-ablation-archive-20261012\raw-only-isolet-seed54-smoke-v3.tar.gz`，压缩包 SHA-256 为 `3c2428073b0630d4b993d1c9fa13a6b8c30aa7d05fac86b6a5d305d36d6b2b37`，逐文件哈希已核对。远端原始运行目录只在本机备份核验通过后清理；配置、清单、日志和归档标记仍保留。六组正式开发运行会恢复原 50/20/50 epochs 及 GPU 协议，并使用独立结果根。
