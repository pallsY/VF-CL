# 3090数值诊断：启动失败，未取得数值结论

## 已封存的执行

诊断代码 `7e9014d7c6b6e26c46825076f22b2f3b1d7fa767`；先验协议 SHA256 `de9b951795689416219ec2fb457a88a39a151fbb9eb90bb3e3d675cbbe0e78ae`。GitHub标签 `3090-fim-diagnosis-v1-7e9014d`，NAS `PRE_EXECUTION_ANCHOR.json` SHA256 `30ecbd8cfce942a37d0b4782560808f6a9ea6576149ff3be6979f2b72a1eec23`。

Windows运行23项，22项通过、1项Linux符号链接检查跳过；两台Linux服务器均23项全部通过。独立代码审查完成。准备期间3080首次clone抢在bundle传输完成前执行，失败日志保留；完整bundle SHA256核验后重新clone并完成CPU单元测试。该准备故障未启动诊断训练。

## 原因与边界

实际capture命令遗漏四个固定环境变量：`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONHASHSEED=42`。这是主执行者的启动命令错误。

原冻结代码 `determinism.configure_determinism` 正确拒绝了四项None。调用栈在 `runner.run_experiment` 的 `initialize_experiment_rng(args)` 处退出；该调用位于 `VFLDataset(args)`、`build_models(args)` 和训练之前。因此没有取得FIM张量、均值差异或准确率，未发生真实测试访问。3090的显存门槛和GPU初始化已执行，不把它描述为完全没有运行任何GPU准备。

未自动重跑。旧严格准入拒绝仍保持拒绝，本次记录只是启动失败，不能推断跨卡数值容差是否通过。

## 证据

本地服务器目录 `/home/nlp/YangXX/vfcl-env-setup/queue-admission/numerical-capture-7e9014d`；完整字节副本和日志保留在 NAS `/brf/users/yangxx/VFCL-admission/3090-20261101/numerical-diagnosis-v1/failed-capture-7e9014d` 及其同级 `.log`。NAS复制时不支持权限元数据保留；文件内容已通过原代码 `load_seal` 完整验证。`SEALED.json` SHA256：`3b94210bf8cf1ddf8213d07891b603c3d8dff0d752729378fa776f3fe9bb09dc`。

正式队列继续使用 `f4735ca`、PID3003183和原18项manifest；未修改代码、清单、账本、锁或训练配置。末次实际状态与哈希在后续校验记录中更新，不将本报告代替实时监控。

## 修复与下一次执行边界

准备一个固定环境的shell启动器，在Python进程开始前设置上述四项；诊断入口再检查环境，在任何GPU资源等待之前拒绝缺失/不符；提供只读CPU的 `check-environment` 命令验证启动链。原smoke、GPU admission、冻结训练与正式调度源码均不改；数值容差、选择完全相同的要求及结论范围不改。

修复只做CPU测试和环境检查。按原先验协议“失败则保留数值诊断证据，不调整本诊断阈值、不自动重跑”，新的capture必须使用新目录、保留本次失败并取得用户对重做一次短预检的确认；当前不启动。

## 已验证修复与待确认的具体操作

修复提交 `914818f5cac638cdb5b9cb1d13764b3bd906c7a6`。Windows运行26项，24项通过、2项Linux检查跳过；3090和3080各26项全部通过。两台服务器均通过固定环境启动器的 `check-environment`（退出码0），该子命令不读取数据、不查询GPU、不创建诊断目录。启动器SHA256为 `839ea4269f1e5d9edebe32abc7b84eeba0df08a999a13ebd2bcc09cd825928f2`，已加入后续诊断封存。独立审查确认环境变量在Python启动前设置，原冻结代码的确定性契约未放松。

只有用户确认重做后才执行3090：

```bash
cd /home/nlp/YangXX/vfcl-env-setup/queue-admission/diagnosis-code-914818f
bash run_frozen_fim_diagnosis.sh /home/nlp/YangXX/conda-envs/vfcl-formal-py311-cu124/bin/python capture \
  --code-root /home/nlp/YangXX/vfcl-env-setup/queue-admission/base \
  --historical-candidate /home/nlp/YangXX/vfcl-env-setup/queue-admission/candidate-7c2cc36 \
  --output /home/nlp/YangXX/vfcl-env-setup/queue-admission/numerical-capture-914818f
```

成功后将整个capture按字节复制到新的NAS目录，使用同一914818f代码验证seal，再在3080做归约重放。复制目录和replay目录均须不存在，任何失败保留、不自动再试。3080通过同一个固定环境启动器执行 `replay`，旧reference为 `/brf/users/yangxx/VFCL-admission/3090-20261101/7c2cc36/reference`，旧candidate为其同级 `candidate`；重放输出拟为 `/brf/users/yangxx/VFCL-admission/3090-20261101/numerical-diagnosis-v1/reference-replay-914818f`。

本次确认仅限再做一次无测试访问的短预检及其归约重放，不包含修改正式队列、额外正式对照、向3090分配正式作业或交接调度进程。
