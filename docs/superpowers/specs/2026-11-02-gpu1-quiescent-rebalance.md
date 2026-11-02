# 同机GPU1未开始任务重分配

用户已请求将剩余任务分给GPU1以加速。范围仅brf3080现有两张3080；不接入3090、不改方法、配置、数据、种子、预算、正式测试次数或已完成结果。原18项manifest、原QUEUE_LOCK与历史ledger字节保持；后续ledger只追加原18项尚未开始工作的start/finish。当前正在运行的任务正常完成，不暂停、迁移或重训它。

## 决策与取舍

即时暂停父调度、收养活worker需要精确退出状态恢复，代价和风险高。旁路新scheduler会重复领取同一pending。采用当前任务完成后的空闲点接管：旧调度每次任务结算后有120秒资源门槛；新控制器只在所有已有start均已success finish、没有任何旧worker时停止并退役旧父进程，再唯一接管pending。没有活训练进程被终止或重新启动。不满足空闲点就继续等，旧队列照常跑；如果已经全部完成则不切换。

当前10成功、1运行、7未开始；候选上界仅 `cifar100/no_partykd/43` 与 ISOLET两arm×42/43/44。执行时与未开始集合取交集；有任何plan、目录、start或身份不确定则不可重领。不会追加候选上界外的新任务。

若CIFAR43仍未开始，GPU1优先跑它，GPU0接续全部ISOLET；若只有ISOLET，则按固定排序交替分配给0/1。GPU0/1各串行一项，GPU0>=3072MiB、GPU1>=5120MiB连续120秒且启动前复查，固定原UUID；外部共享继续允许。

若空闲点剩余不足2项，或固定分配不能产生双卡并行，则不切换，让旧队列继续；不为没有并行收益的情况退役调度器。

## 原证据与新的执行映射

原manifest SHA `bb10fa6c62719b68124e7b625a204defc6b3e396ac285c068849fff8866e8fca`，原调度代码f4735cacf51c41d63a9f87e8933c968cb268637b。原训练根e2a559d和b13ac24不改。新控制器调用干净f473模块的load_jobs复用全部原数据、配置、runtime、preflight验证，绝不伪装成旧parent或改QUEUE_LOCK.pid。

所有候选config.device均cuda:0；配置文件字节完全不变。GPU0 worker的CUDA_VISIBLE_DEVICES按[UUID0,UUID1]，GPU1按[UUID1,UUID0]，保持两卡可见数量，并显式记录逻辑cuda:0到实际物理UUID的映射。新执行GPU与原manifest物理分配的差异只在追加执行扩展中声明，不伪写原物理身份。两张卡同机、同驱动且已经被本队列使用，不使用3090跨机数值诊断作为本次准入依据。

扩展根固定为原结果根下 `gpu1-rebalance-v1`。ARMED.json记录候选上界、旧PID/boot_id/starttime/命令、原manifest与当时ledger SHA、固定代码/设备与调度规则。实际切换记录EXECUTION_EPOCH.json绑定成功前缀原始字节SHA、旧进程已退役证明、实际pending与GPU映射、新controller PID/starttime、新代码与ARMED哈希。原ledger不得插入epoch行，仍每项start/finish两行。

## 空闲点屏障

先只读观察旧ledger完整行/成功classification及seal/log字节SHA、进程树和根目录状态。没有失败、半行、未知目录、未结算start和任何直接旧child时，验证旧PID完整身份后才SIGSTOP整个旧父线程组；验证所有线程停止，重复两次稳定账本/无child盘点。检测到竞态则在没有任何新writer/worker/epoch前恢复旧父继续正常队列，保留拒绝快照；不能猜测退出码或补写finish。

确定空闲且快照稳定后退役旧父（SIGKILL，仅此已绑定进程，无活训练child），验证其终止，永久不再恢复。再创建新epoch并开始唯一写账本。旧进程在先前已完成作业的Popen.wait中得到的返回码已经由旧finish记录保存，不需要收养或重建。切换期间若有任何不确定性则失败关闭，不启动任何pending。

## 新worker与全局失败停机

新worker必须验证新parent PID/starttime/epoch SHA、原manifest与config SHA、唯一start、attempt1、plan、固定训练commit/runtime及物理映射。它先写READY并等待GO，父控制器确认身份后才GO；start已写但启动失败仍消耗该attempt，绝不自动重跑。

新worker在训练前启动独立线程监控父PID/starttime；父退出/身份变化则停止本worker自己的独立session，覆盖loader子进程，不依赖NAS心跳。在父控制器一侧，先处理全部已退出worker并完成seal/退出码/FIM日志分类，再允许新start/GO；检测任一失败后立即关闭调度并停止本研究所有活worker。不得按用户名/GPU/宽泛pkill终止外部任务。控制器正常异常路径要有finally fail-stop；未知退出状态、OOM、NaN、audit失败均保留并停止，不自动重跑。

worker训练与一次正式publication仍走冻结runner和verify_formal_seal。JOB_RUNTIME、JOB_SUCCESS和start/finish明确附加epoch与实际执行身份，不伪造原parent。新QUEUE_COMPLETE标记真实新orchestration_commit并显式绑定原orchestration_commit、execution extension SHA、36行ledger SHA、18项全部成功；若失败写失败记录并停止。

## 审计与发布门槛

独立audit须显式接收执行扩展路径和外部SHA，验证完整原成功前缀、限定pending、唯一writer、精确物理/逻辑映射、新旧metadata绑定及全部原seal检查。没有扩展参数时保持原审计行为；不得为了新映射放松旧路径的GPU/parent/来源验证。18项全成功且新审计通过前不读取或汇总准确率，原PartyKD和FIM解释规则不变。

在隔离branch完成TDD、Linux合成空闲点/退出竞态/重复claim/错PID/半行/未知目录/资源门槛/worker父死亡/同伴失败/单次publication审计测试及独立review，固定NAS/GitHub代码与armed锚点后，才能部署空闲点切换监视器。用户本次同机GPU1重分配请求覆盖此范围，不再重复索取同一授权。任何需要中断活训练、重跑已开始作业、修改冻结方法或接入其他主机的方案均超出范围，不执行。
