# 左臂训练准备与 P0 检查（2026-10-16）

范围：先做 attention 一致性和源数据到合并数据的对应检查；2026-10-16 收到启动新训练的明确指示后，启动左臂 40k 训练。没有运行 P1/P2 或闭环评测。

## 左臂训练准备

- 原始数据：`datasets/cookie_multi_v1`，保留不改。
- 新数据：`datasets/cookie_multi_left8_v1`，200 集、140795 帧，state/action 为左臂 7 个关节及夹爪，共 8 维。
- 所有 140795 帧已验证：新 state/action 精确等于原字段前 8 维，其他 parquet 字段不变；统计量也对应裁剪。视频通过链接复用原始文件，不重复压缩。
- 训练 dry-run 配置：40k 步、batch 32、学习率 5e-5、warmup 500、衰减至 5e-6；从基础模型开始。相较上轮 20k 步，名义数据遍历次数从约 4.55 增加至 9.09。
- 两任务、三路相机、chunk_size 50、文本长度 128 保持现有配置。仅按本次要求修改 state/action 维度和训练步数。
- `left_training_plan.json` 保存完整配置和命令。新训练已经按用户后续指示启动；进度以 `outputs/cookie_multi_left8_40k_v1/status` 和 `train.log` 为准。Attention P0 的数值阈值尚未全部通过，这一限制仍成立。

## P0 数据对应：通过

- 两个源数据集到合并数据的全部 140795 帧：state、action、velocity、eef_pose、force、timestamp、frame_index 精确对应，episode/index/task 映射正确。
- 对 200 集各抽查 5 个位置（首帧、第二帧、中段、倒数第二帧、末帧），通过 LeRobot/PyAV 解码；共 1000 对样本、3000 对相机图像，源数据与合并数据像素最大差均为 0。
- 从实际行重算左臂均值/标准差，与元数据最大误差小于 7.4e-7。
- 边界：本检查证明合并过程保持了被检查字段和图像对应关系，不代表原始采集的物理时序已通过独立仿真回放。

## P0 attention：修复兼容性，数值阈值尚未全部通过

在本机 RTX 4070 Ti SUPER、PyTorch 2.10.0+cu128，以现有 20k 检查点、两个任务各一条真实观测（episode 0/100，frame 20）固定 RNG 比较 eager 与 SDPA。

初始检查发现：关闭 autocast 时，真实模型可产生 Q=bf16、K/V=float32，原 shim 会因 dtype 不同报错。已修改 shim 统一混合 dtype；同时补齐 fully-masked query 与 eager 的均匀 value 平均语义。4 个定向测试通过，涵盖输出和梯度。

| 模式 | 最大 loss 相对差 | 最大梯度相对 L2 差 |
|---|---:|---:|
| 关闭 autocast（模型参数仍含混合 dtype） | 1.054% | 3.506% |
| bf16 autocast | 4.097% | 6.060% |

25 步 flow 推理的反归一化动作块最大绝对差约 0.000738（混合关节弧度与夹爪开度单位）。预先设定诊断阈值为 loss 2%、梯度 5%、动作绝对差 0.01；bf16 的两项检查未全部满足，因此报告为 `failed`。这些阈值是诊断条件，超阈值不证明其导致了闭环退化。

上一轮保存配置同时含 `policy.use_amp=true` 与 `accelerator.mixed_precision=no`；本次无法仅据该配置证明历史进程的 autocast 状态。此前直接称上一轮采用 AMP 的表述需要修正。

训练使用当前 SDPA 实现和 GPU，`model_launch.json` 与实时日志记录具体命令。开始训练不表示 attention 数值问题已得到因果解释。没有因本次检查改动 LeRobot 工作区或启动其他实验。

证据：`data_report.json`、`attention_report.json`、`left_training_plan.json`、`provenance.json`。两个检查脚本和回归测试与本报告同目录。
