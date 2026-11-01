# 3090 接入当前正式队列：交接可行性审查（未激活）

日期：2026-11-01。范围：只读代码与设计审查；本报告不授权任何远端操作。未发送 SSH 信号、未暂停/终止进程、未修改 live 代码/manifest/lock/ledger、未写 claim、未启动 worker。运行状态（brf3080 PID 3003183、f473、两个独立 session 的 CIFAR fim_off 42/44、seed43 两 arm 尚未开始）来自任务输入，未实时验证。

## 结论

当前代码不存在可直接启用的无中断多机交接路径。最小运行风险方案是保持原控制器运行，继续离线完成候选设计与测试；若 seed43 在准备期间被原控制器启动，则本轮候选失效，不能抢占、重启或重训。单纯等待两个当前训练完成不能形成可靠交接窗口：每条 lane 在 settlement 后自行取下一项，另一条 lane 无须同时完成。

如果坚持在本轮加入 3090，必须先完成并独立审查一种显式的旧控制器 fencing、既有进程收养与 finish 结算协议，再取得最终激活确认。它需要暂停/退役旧控制器以及变更新 worker 授权协议；这不是已支持的热接入，也不是修改一个 GPU allowlist。当前阻碍未解决前应保持未激活。

## 代码事实与约束

- `frozen_ablation_run.py:run_queue` 无 drain 文件、信号处理器或恢复入口。启动时拒绝任何已有 QUEUE_LOCK、ATTEMPTS、QUEUE_COMPLETE、QUEUE_FAILURE；不能把已有账本直接交给同一个 run_queue 重启。
- 两个线程各自持有 jobs 的内存快照。修改磁盘 manifest 或增加旁路 claim 不会改变旧 pending；旧控制器既不读旁路锁，也不参与新锁，两个独立 writer 会产生重复启动风险。
- 所有 start/finish 写入由该进程 mutex 保护，只是进程内锁。新 controller 即便使用 flock 也无法约束从未使用该 flock 的旧 controller。
- worker 启动时重新 load_jobs，验证 manifest 的固定路径/哈希、18项完整集合、代码 commit/clean、原 formal runtime、配置推导及 physical_gpu。随后要求 QUEUE_LOCK.pid == os.getppid()、唯一 start、精确 plan。此验证只在进入训练前执行一次，不会因之后 parent/lock 变化自行停止。
- 子进程 start_new_session=True。单独停止或终止 parent 不等于停止训练 session；停止整个 parent 进程组也不能假设覆盖它们。现有 fail_stop 只有旧 parent 内存 active 映射中的两个本地进程组。
- finish 逻辑依赖 Popen.wait 的精确 returncode、JOB_SUCCESS、FIM off 的完整 num_tasks 零冻结日志、日志/成功文件哈希、test consumed 状态。完成一个模型文件或看到 JOB_SUCCESS 都不足以替代正式 finish。
- `runtime_identity()` 包含 host，以及 Python、Torch、CUDA runtime、NumPy、Pillow、scikit-learn、torchvision。即使软件版本全部相等，3090 的 host 仍与原 runtime 不同；原代码会拒绝。驱动/GPU UUID 不在该 runtime 字典中，仍需独立绑定。

## 方案比较

1. **原队列继续、不在本轮接入：推荐的当前选择。** 不改变任何已启动 attempt 和单次测试证据，原 controller 仍唯一结算者；代价是 seed43 可能被原队列先启动，机会自然失效。
2. **并列启动第二 scheduler/remote worker：不可接受。** 无法保留唯一 writer/global fail-stop，旧内存 pending 不会被新 claim 排除；即使只分配两个 seed43，也存在竞态。
3. **更新 live 文件、manifest 或 QUEUE_LOCK：不可接受。** 旧 controller 已加载代码和 jobs；既破坏冻结证据又不能可靠 fence。改锁 PID 也只能改变后续 worker 授权，不能阻止旧 start 落账。
4. **等任务完成后 SIGSTOP parent 再切换：仍需专门协议。** 没有自然安全点；信号可能落在 plan/start/Popen/finish 的任一位置，不能假定冻结时两个孩子都在训练。
5. **最终批准后暂停旧 parent，收养既有 children，永不再运行旧调度循环：技术上可研究的候选。** 保留训练不重启，但必须解决精确退出状态、半写账本、启动窗口、授权代际和失败停机守护。没有这些实现与实机同构合成验证，不可准入。
6. **停止现有 children 并重启剩余队列：不满足不重训已启动 attempt，本轮不可采用。**

## 条件性交接设计（仅供后续实现审查）

### 单 controller 与 fencing

新控制器只读准备，不能在 fencing 前 append 或 GO。最终激活流程需绑定旧 PID、boot_id、/proc starttime、exe/cmdline、原 lock/manifest/ledger 哈希、所有线程与实际 worker session 身份。PID 数字本身不够。

最终批准后只能针对旧 controller 建立停止/退役屏障，不能向现有训练 session 发停止信号。须验证整个线程组已停止，并完成两次稳定的进程/文件盘点；停止瞬间的有限快照不能代替多线程及 fork/exec 竞态处理。原 parent 若可能再运行，则 fencing 不成立。外部新 lock 只能辅助排斥新实现，不能 fence 旧实现。

冻结后需要分类每个 key：完整 start+活 child；完整 start+已退出；完整 plan但无start；start已落账但未spawn；部分 JSON 行；启动中的 fork/exec 子进程；完整 finish；未知目录/测试访问。任何不确定项进入阻断，不截断或自动修复账本，不再调度其他项。start 已存在即已消耗 attempt，未产生模型也不得自动重启。孤立 plan 默认阻断，不能当未开始回收。

只有所有既有 start 精确绑定且旧 writer 永久 fenced 后，新 controller 才取得唯一 ledger 权限。交接记录需追加、不可重写原 lock/manifest；原 lock 作为 epoch0 原证据，新增 epoch1 handoff/lock 明确旧 writer 退役原因、新 scheduler commit、完整状态与文件哈希。旧 parent 不得恢复运行以“顺手结算”，否则将带着旧 pending 和旧 active 重新调度。

### 收养既有训练与 finish

保留相同 PID/session/训练目录/配置/attempt-01、原 start、JOB_RUNTIME、原 manifest SHA 和训练代码，不重跑 worker 入口，不移动目录，不重新授权测试。其启动时 parent 绑定已完成，新 controller 只是监督与结算接管，不宣称自己是历史 parent。

新 controller 不是这些进程的 parent，通常不能 waitpid 得到精确退出码；pidfd 可帮助可靠识别/等待退出，但不自动赋予非父进程 wait 权限。不得把“进程不见了 + JOB_SUCCESS”推断为 returncode=0。

一种需 Linux 同构试验的候选是保留停止的旧 parent，让退出 children 成为其未回收 zombie，从 /proc 身份绑定的退出状态读取原 wait status；还必须证明旧 parent 停止前未在 wait 中回收某个 child、不会失去状态，并覆盖多线程 Popen 的竞态。已被旧线程回收而 finish 尚未写出时，退出码可能仅在旧进程内存中，外部不能可靠重建。该窗口必须阻断而非猜测。ptrace 注入、修改旧进程内存、用调试器调用 finish 不是最小风险方案，不建议。

更稳健的后续架构是在下次从空队列启动时就使用可 drain 的 controller 与独立、持久退出回执的 supervisor；当前进程无法追溯安装这种回执。若无法证明本轮收养状态完整，则放弃本轮接入，保留证据。

抽取 finish 为纯验证/分类函数时，保持原 classification 顺序与全部判据；新 writer 每个 start 最多结算一次，保留原 start 不补写重复 start。既有成功 seal 必须只读验证，绝不再次调用训练/正式 test。队列总成功数需从完整 ledger 重建，不能只统计新 controller 的内存 outcomes。

### manifest / runtime / parent 授权

原 manifest 字节、配置与 runtime 保持原样，不能把 3090 的 host/UUID 写回原行。新增 admission/handoff 扩展锚点需明确列出仅 `cifar100/fim_off/43`、`cifar100/no_partykd/43` 的远端位置映射、逻辑 device 映射、原配置哈希、实际派生配置及精确允许差异、实际 runtime、GPU 型号/UUID/驱动、NAS来源、候选 scheduler/worker commit和全部准入证据。

原 config device 与双卡 CUDA_VISIBLE_DEVICES 的语义也不能照搬：单物理卡屏蔽后必须验证逻辑索引配置合法，输出目录仍为原 attempt 路径。所有路径/device 差异须提前封存，不能以实现方便默许。

新授权 worker 必须是隔离候选实现，用明确 epoch、host、broker PID/starttime、controller身份、assignment/READY/GO链条替代单个跨主机 PID 比较。不得伪造本机父 PID，或轮流改写原 QUEUE_LOCK.pid 以迁就 broker。已有 workers 不改；未来本地 workers 与远端 workers 都由 epoch1 统一规则授权。原 load_jobs 的 runtime=原formal及orchestration_commit要求也需显式兼容扩展设计，不能删除 host 检查后称完全一致。

### 全局 fail-stop 与断网

中央仅有一个 controller 写 plan/start/finish/claim；远端 broker 仅写控制目录 READY/heartbeat/EXIT/abort，不写账本。协议采用唯一 epoch+nonce+递增sequence，所有消息绑定原manifest、扩展锚点、job、attempt、配置、代码和主机/GPU身份。ASSIGN→READY→持久start→GO；start 后 GO 丢失也消耗 attempt，不能重派。

一把共享门串行化所有 lane 的 admission、start、GO与settlement；检测任何失败先关闭门，再停本研究所有已验证 session，最后结算。包括继承的两条本地 session，不能仅停新增 workers。禁止按用户名、GPU占用或宽泛 pkill 操作其他研究。

中央须同时监控 SSH进程退出/通道断开、递增heartbeat进度和所有 worker退出；任一失效都置不可逆停止，绝不因心跳恢复自动继续。heartbeat 只能使用接收方 monotonic 跟踪新sequence，不用mtime或远端时间判断；身份改变、重复旧消息、损坏JSON均失败。

仅中央通过 SSH 发 kill 无法满足断网时停远端。远端必须有独立监督者/看门狗，训练不阻塞其 lease 监控；需要双向 controller lease，EOF/SSH loss/过期会在远端停止本研究 session，controller同时停止本地。若 NAS失联或写heartbeat线程死亡，watchdog仍须独立存活并执行停机。明确心跳周期、最大lease、TERM/KILL宽限期及断网最大停止上界；分布式“同时”只能是有界停止，不能承诺原子同刻。

旧 broker 只有单向 heartbeat、阻塞 process.wait 和空闲时 STOP 轮询，不提供此双向 lease；不可直接复制后宣称满足要求。旧 HandoffGate 也要求所有调度写入经其门，现有 f473 并不满足前提。

## 激活前必须补足的审查与测试

- GPU四profile及3080逐位比较全部通过，软件/数据/NAS和候选GitHub/NAS锚点完成。
- 隔离合成进程/NAS目录，不触碰正式结果：冻结发生在mkdir/plan/start/fsync/Popen/授权/退出/wait/finish每个边界；线程组停止、未知子进程、半行与PID复用均默认阻断；证明不重复、不漏finish、不启动新attempt。
- 精确验证旧child退出状态可获取性；已回收未finish窗口不可恢复时明确拒绝，不放宽成功判据。
- 比对新旧finish对所有成功/失败/OOM/缺seal/FIM日志缺失/peer cancellation的结果；已消费测试凭据只读保留。
- 同时退出与失败期间零新start；SSH断链、网络分区、NAS挂起、中央死亡、broker死亡、heartbeat线程死亡、乱序/重放GO、lease过期都触发本地与远端有界停机；仅研究session受影响。
- 实际候选只在fence后的稳定账本重读后确定；任一seed43有start/plan/不确定证据就拒绝原分配并重新封存，不扩大白名单。
- 全部实现/测试/独立审查完成后，向用户展示具体操作、修改的证据协议、暂停/退役对象与回退边界，取得最终激活确认。回退到旧parent的普通SIGCONT不是安全回退。

本报告仅提出可行性和阻碍，不表示收养方案已实现、已测试或已获得执行许可。
