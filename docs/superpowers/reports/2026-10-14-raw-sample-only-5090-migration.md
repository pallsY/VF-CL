# Adaptive20 仅原始样本回放：5090-1 迁移记录

## 范围

回放记忆仅保留每类 20 个原始样本。此模式关闭持久原型、原型损失、SDC 和特征蒸馏，仍保留 logit/party 蒸馏、FIM、BiC 与样本支持的最终双分支头。以下准确率均来自开发留出集，不是正式 `AA_final`；正式测试集未访问。

训练源码固定为 `99e1da7e05e7eeb8b57587b6d3a75c6eaa490066`。

## 首组完成与迁移

原 4090-1 等待器在迁移准备期间检测到 GPU 空闲，自动完成 `isolet_seed54_SAMPLE_ONLY`，退出码为 0。检查到运行目录已生成后，没有终止或重复启动该组。读数：ClassIL `0.8990384615`、TaskIL `0.9932692308`，留出样本 `1040`，原型缓存 `0` 字节，原始样本缓存 `20/class`。冻结状态和待读正式评估标记存在；访问记录只有开发验证集，未发布 `results.json`。

其 49 文件完整档案位于 `D:\VF-CL-ablation-archive-20261012\raw-only-isolet-seed54-full-4090-v1.tar.gz`，SHA-256 为 `e8087cde248bf4499baabb678149f354d18a2c7cba8a316ea90fb4a3ff9b4d94`，已按逐文件清单核验。本机读数 SHA-256 为 `5d1c7d3d426f977619bc7a1a01593fa98eafdcfd7d02cf58590f6d57139aa2a8`。远端运行目录只在本机完整核验后清理，保留 `ARCHIVED_isolet_seed54_SAMPLE_ONLY.txt`、读数和审计小文件。

其余五组在 `vf-cl-5090-1` 上依次运行，避免与 4090-1 重复。5090-1 的训练源码目录为 `/home/c5090/VF-CL-worktrees/raw-sample-replay-20261014`，由 SHA-256 为 `209fc2acfa96e558d92c5144a8e744bb5789d5bfc29115a2fa8a2551ed1855da` 的完整 Git bundle 恢复并检出上述固定提交。结果根为 `/home/c5090/VF-CL-results/adaptive20-sample-replay-only-dev-20261014-v2`。

5090-1 的 `/home` 分区约剩余 13 GiB；`/` 分区的 179 GiB 空间不计入训练输出容量。脚本要求表格数据集至少 2 GiB、CIFAR 至少 8 GiB 空闲，并要求 GPU 无任何计算进程。每次只运行一组，完整档案直接流式复制到本机 D 盘并逐文件核验，核验前保留远端运行目录。5090-1 使用 RTX 5090、驱动 580.173.02，基础 Python 环境 PyTorch 2.12.0+cu130。

迁移清单为 `adaptive20-sample-replay-only-dev-20261014-5090-v2-manifest.json`，SHA-256 为 `a1958b65242cfea90fb80591d117ab6fb4143c6c7f7380bd8827683f2bffeae3`。五组配置相对原 4090-1 配置仅更改数据和输出路径；训练参数、种子和留出协议保持一致。守卫脚本 `start_sample_replay_only_dev_5090.sh` 的 SHA-256 为 `d76cb54ae3b0bac7b2e3430ab35310aa6b406db1e02446f73acd622cc6989308`。

ISOLET、UPMC 和 CIFAR-100 数据已复制到 `/home/c5090/VF-CL-data`；五个原始数据文件的远端 SHA-256 均与迁移清单相同。文件核验完成时，`/home` 仍约有 13 GiB 可用空间。

5090-1 上的独立 CPU 实现烟测已退出 0：留出 `1040`、`raw20/class`、原型 `0` 字节、ClassIL `0.8826923077`、TaskIL `0.9798076923`。其 49 文件本机档案 `D:\VF-CL-ablation-archive-20261012\raw-only-isolet-seed54-cpu-smoke-5090-v1.tar.gz` 的 SHA-256 为 `dc1cda2c14008cdc82a68bafc2d57e6253a94c8711afbe5e7176894522216c91`，已逐文件核验。该烟测只验证环境与实现，不计入六组完整实验。

`isolet_seed55_SAMPLE_ONLY` 的守卫等待器 PID `2762707` 已通过源码、配置、ISOLET 数据及前组归档检查。2026-10-14 约 15:57 CST，5090-1 的外部 GPU 进程 PID `4160238` 仍占用约 25.8 GiB，因此尚未生成运行目录；等待器将在 GPU 无计算进程时自行启动。未向外部进程发送信号。

## 5090-2 的后续排队

2026-10-16 约 17:07 CST，确认 5090-1 的 `isolet_seed55_SAMPLE_ONLY` 仍没有运行目录、训练日志或退出读数后，安全停止其等待器 PID `2762707`，留下 `MIGRATED_BEFORE_TRAINING_isolet_seed55_SAMPLE_ONLY.txt`。本机也保存了该小证据文件。未触碰任何外部进程。

用户提供的新 5090-2 地址可连接，但其 GPU 仍有多个外部计算任务，显存仅剩约 6.3 GiB，磁盘约剩 7.4 GiB。因此只在 5090-2 排队 ISOLET seed55，并继续要求 GPU 无计算进程、该盘至少 2 GiB 空闲后才启动。其余 UPMC/CIFAR 组没有在 5090-2 启动；CIFAR 尤其不能按当前磁盘余量运行。

5090-2 的独立源码目录为 `/home/c518/VF-CL-worktrees/raw-sample-replay-20261016`，从相同 SHA 已核验的完整 Git bundle 检出固定训练提交 `99e1da7e05e7eeb8b57587b6d3a75c6eaa490066`。ISOLET 数据 SHA-256 `d34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514` 核对通过，Python 环境可以导入 PyTorch、实验入口及主要模块。结果根为 `/home/c518/VF-CL-results/adaptive20-sample-replay-only-dev-20261016-v3`。配置相对 5090-1 仅更改数据和输出路径；配置 SHA-256 `255ecc6150708e8abebad77bf16387baecf406c1e805524004a236b7255353bf`、清单 SHA-256 `6d341751cb929450080d402d9758e4c304f8c9991e92ea1af525a642ebfd2083`、守卫脚本 SHA-256 `bd4dcc6c4b60e7f30ca5e0bf9ca28124da17cd86aa8292af919ee9d3a1ac598f` 均在远端核对。守卫同时要求前组归档证据和 5090-1 未训练迁移证据。

5090-2 的新等待器 PID `1510676` 已通过源代码、配置、数据和前组归档检查（`PREFLIGHT_OK`）。截至 2026-10-16 17:11 CST，GPU 仍被外部任务占用，没有训练目录或读数；等待器将继续等待。5090-1 上该组等待器已停，不存在双重排队。

待全部六组完成后，最终台账须标注 seed54 与后续组分属不同 GPU，运行时间不得直接做跨机器速度比较。旧 P0R1 仅可作为机制不同的参考，不能冒充本模式的结果。
