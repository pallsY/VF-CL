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

## 同时观察 5090-2 与双卡 3080

按用户要求，持续观察 5090-2 GPU0 与双卡 3080 的 GPU0/1，比较谁先空闲。2026-10-16 19:49 CST，两台都仍有外部 GPU 进程；5090-2 的 ISOLET seed55 等待器保持唯一活动队列。双卡 3080 本地盘仅约 684 MiB，不使用曾两次被全局清空的 `/dev/shm`。它另有用户可写的持久 `/brf/users/yangxx` 挂载（约 18 TiB 空闲），可以作为经过核对后的备用结果位置。

备用源码在 `/brf/users/yangxx/VF-CL-worktrees/raw-sample-replay-20261016-standby`，从相同 SHA 已核验的 Git bundle 检出固定训练提交；ISOLET 数据仍使用已经 SHA 核对的 `/home/c3080/YangXiaoXiang/VF-CL/data/isolet/isolet_vfl.npz`。备用结果根 `/brf/users/yangxx/VFCL-results/adaptive20-sample-replay-only-dev-20261016-3080-standby-v1` 已放置配置、清单及未启动的守卫脚本。其配置 SHA-256 `b15c0d3477d5c5832d0f322d15d99ad7a40760c90ab577c634fbaf2c7bcce1eb`、清单 SHA-256 `98300cc5b310b072ddbe76439aba76ed56e9f22e65cf25eceb3c01ed4a7e120a`、守卫脚本 SHA-256 `063e9c4b6e5de2d834d3ef75390b8a1d231dd2c1b011265bd8f8de017029b282` 已远端核对。当前没有备用训练进程或结果目录。

如果 3080 GPU0 先空闲，必须先确认 5090-2 仍无训练目录、日志和退出读数，安全停止其唯一等待器，再将带有 `training_started=false` 的迁移证据送至 3080。3080 脚本在该证据、前组归档、固定源码与数据 SHA、GPU0 无外部计算进程、`/brf` 至少 8 GiB 空闲时才能启动，并将临时文件和输出写入 `/brf`。若 5090-2 已先开跑，则 3080 备用配置只保持未启动状态。

双卡 3080 的 GPU1 也可作为同等备用：`start_dev_gpu1.sh` 仅将可见设备与空闲检查从 GPU0 改为 GPU1，SHA-256 为 `59c26218d663176457d96386860930df8262a24c33f74e3b025bbe3bbd9db673`，已复制到同一备用根并通过语法检查。两脚本只能选一份启动，且启动前均须先完成 5090-2 唯一等待器的安全撤销。

## 3080 GPU1 实际启动与临时目录修复

2026-10-16 20:50 CST，3080 的 GPU1 先空闲。确认 5090-2 仍没有运行目录、训练日志或退出读数后，只停止它的等待器 PID `1510676`，留存 SHA-256 为 `8236afabb36a4fd1689597db10d378b264fe36a8637177a1f28f87203ded45cf` 的 `training_started=false` 迁移证据。3080 按该证据、前组归档、源码/配置/数据 SHA、GPU1 空闲及 `/brf` 空间检查后启动唯一任务。

首次 3080 备用根在 `event_0_CIL` 的数据加载阶段报告 `OSError: AF_UNIX path too long`。追查发现启动脚本把 `TMPDIR` 指向约 89 字符的持久结果路径；Python multiprocessing 再附加套接字名，超过 Unix 路径限制。该运行没有冻结文件或开发留出读数；只对自己的 pilot PID `136997` 发 `TERM`，wrapper 记录退出码 `143`。保留失败目录、日志和迁移证据，不将其计入六组。

在同一 Python 环境下用短 `/tmp` 目录创建 Unix multiprocessing listener 成功，监听地址长 `36` 字符。因此另建结果根 `/brf/users/yangxx/VFCL-results/adaptive20-sample-replay-only-dev-20261016-3080-v2`，仅把临时套接字目录设为 `/tmp`，完整运行输出仍落在持久 `/brf`；脚本还核对旧运行退出码 `143` 且无读数。新配置 SHA-256 `1d4de4d438051f62174bc66b8166685f63b07bcf7675fb48194609616208e434`、清单 SHA-256 `a0ed87dd346c9054b50c7cf41740bb272e98b31e25687df76bd55587bf488c76`、GPU1 脚本 SHA-256 `2d0c4ec9163bd4d812d7053354ab1528f0b4b530955e92a7e7aa1a2a55c4c69d` 已远端核对；训练源码仍固定不变。

新 wrapper PID `149657` 于约 20:52 CST 启动，通过预检且进入 `event_0_CIL`；此时尚无退出读数。继续监控日志及持久磁盘，成功后再做冻结/留出/零正式测试审计和本机全文件备份。

## ISOLET seed55 核验完成

修复后的 v2 运行于约 2026-10-16 20:55 CST 退出 `0`，冻结文件、开发留出读数及正式评估待办标记齐全。开发留出 `1040` 个样本，ClassIL `0.9173076923`、TaskIL `0.9923076923`；`variant=sample_replay_only`、原型缓存 `0` 字节、原始样本 `20/class`。留出清单 SHA-256 为 `d1bab580ecab91b054a4bd7f21fde9d30612270c845da52daed899dabb7e00d7`，与 seed54 一致。正式访问目录只有 `validation.consumed.json`，没有 `results.json` 或正式测试发布文件。

完整 49 文件档案在 `D:\VF-CL-ablation-archive-20261012\raw-only-isolet-seed55-full-3080-v2.tar.gz`，SHA-256 `fdb1b224f9213338508a463c68a8321017e79dc307991f646093f679841db410`，按逐文件清单 SHA-256 `3459364f6cba48f8665e9515a8d77d8255b128a204eac1772bf5ea991926af2c` 全部核对通过。本机读数位于 `C:\Users\30783\Documents\VF-CL-sample-only-results-20261014\ISOLET_seed55_SAMPLE_ONLY_HOLDOUT_READOUT_3080.json`，SHA-256 `65116cfc0fdf91e9e011531ffc1acba398d85cff4654ae8b4664135ec7dce710` 与远端相同。远端写入 `ARCHIVED_isolet_seed55_SAMPLE_ONLY.txt` 并保留运行目录作为额外副本。至此新模式完整科学运行 **2/6**，CPU 烟测与首次 event0 环境失败均不计数。

UPMC seed54/55 的迁移配置已单独登记在 `adaptive20-sample-replay-only-dev-20261016-3080-upmc-v1`，仍固定同一训练源码和原有 UPMC 留出协议。原始 UPMC 文件正复制到 `/brf/users/yangxx/VFCL-data/upmc_food101_image_clean_v1/`；在复制完成并校验 SHA-256 `dea0569f0d4195eaaecc9cab3a6cce533f07b8d88472e7dbca5113152a314339` 前不得启动。UPMC seed54 配置 SHA-256 `0d2e4735c5700323c70ac15239e6801e8cc09e283757d3dda489899e6eaffd0a`，seed55 配置 SHA-256 `c4c98118e5bde4d562d36ac8c9371a2daf47f8e50ef56b9eba6bf739f9a70efa`；清单 SHA-256 `0a388a144d5335455c06e5da1881271235671a5d34c1cf1b95f7ed553cbc1baf`。备用 GPU0/GPU1 脚本 SHA-256 分别为 `8d67dc0fad36041fbd2cf6236113693f2e9879991cbbbd046a33cfe39a9cbd35`、`df75a8ca90a3f6f1b8994c99d81f10ae5dd342ab6f24813d5fb2b17131a3f098`，脚本需前组 `ARCHIVED` 证据及目标 GPU 空闲，当前尚未启动。

CIFAR-100 seed54/55 也已在独立的 3080 持久根 `adaptive20-sample-replay-only-dev-20261016-3080-cifar-v1` 预备，尚无训练进程。它使用原有可读的 CIFAR train/test/meta，三文件 SHA-256 分别为 `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`、`4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864`、`a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9`，远端重新核对。seed54 配置 SHA-256 `3d85f6593f2b9d252073f676a0e6ce9be81ccb7e4f4571e2aa7460a5acaa57a3`、seed55 `dbe0fb55a393f1f203b524ebdc0eee6f3ffc4e06dbdea84f8798ff57ec6d0a7a`，清单 `a248bcfffe7cc4f369c2e8f6ab5175f9895bdf92c375a075ffb034d819efde55`；两张 GPU 的互斥守卫脚本分别为 `e91840d91241c6b55a73f31b776acd4a66b9c0991e406b0e421e838aee31bf7d` 和 `9f08d56aef4a3012017ffffb7014d6ea0ec38a5ba424e6ddcb35ade3571ef740`。启动顺序必须等待 UPMC seed55 完整归档并且目标 GPU 空闲，所有输出留在 `/brf`。

待全部六组完成后，最终台账须标注 seed54 与后续组分属不同 GPU，运行时间不得直接做跨机器速度比较。旧 P0R1 仅可作为机制不同的参考，不能冒充本模式的结果。
