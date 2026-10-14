# 入口索引与实验管理

正式实现位于 `src/a3_dual_arm_sim/`，`examples/` 仅转发对应模块。源码用户运行脚本，wheel 用户使用 `python -m a3_dual_arm_sim.<实现模块>`。所有入口支持 `--help`。

## 主流程

| 入口 | 实现模块（包名之后） | 用途 |
| --- | --- | --- |
| `a3-sim inspect/smoke/run/teleop/replay` | `cli` | 通用仿真与策略、遥操作及回放 |
| `a3-sim download-smolvla/train-smolvla` | `cli` | Hub 下载及公开 LeRobot 训练 |
| `a3-sim train-act` | `learning.act_training` | ACT 配置适配及公开 LeRobot 训练 |
| `run_cookie_batch.py` | `workflows.run_cookie_batch_cli` | 同列/跨列专家 |
| `run_cookie_same_column.py` | `workflows.run_cookie_same_column_cli` | 同列专家便利入口 |
| `run_cookie_transfer.py` | `workflows.run_cookie_transfer_cli` | 单块/协作搬运专家 |
| `evaluate_cookie_transfer.py` | `workflows.evaluate_cookie_transfer_cli` | 单块任务评估 |
| `benchmark_cookie_batch.py` | `workflows.benchmark_cookie_batch_cli` | 统一 2×5/2×10，支持 --task、--policy、数量评分及可视化 |
| 模块运行 | `workflows.evaluate_smolvla_grid_cli` | 两任务×三档 SmolVLA 网格，默认 500 步，支持 --resume |
| `collect_cookie_2x10.py` | `data.collect_cookie_2x10_cli` | 二十块专家采集、首夹计划与续采 |
| `run_cookie_2x10.py` | `workflows.run_cookie_2x10_cli` | 兼容的 2×10 专家演示入口 |
| `preview_cameras.py` | `workflows.preview_cameras_cli` | 三相机预览 |
| `collect_cookie_benchmark.py` | `data.collect_cookie_benchmark_cli` | 成功专家演示采集 |
| `prepare_left_arm_dataset.py` | `data.prepare_left_arm_dataset_cli` | 16D→8D 数据转换 |
| `merge_cookie_datasets.py` | `data.merge_cookie_datasets_cli` | 2×5+2×10 合并为单模型多任务数据集（`target_layout: multi`） |
| `audit_smolvla_data.py` | `data.audit_smolvla_data_cli` | 数据及检查点统计审计 |
| `smolvla_offline_error.py` | `workflows.smolvla_offline_error_cli` | 训练数据首动作误差，`--policy-type act` 支持 ACT |
| `summarize_smolvla_trace.py` | `workflows.summarize_smolvla_trace_cli` | 汇总诊断动作轨迹 |

主流程指维护入口，不代表所有任务已获得成功率或完整训练验收。单块顺序专家在密集 80 块布局上的两个测试为既有预期失败；批次任务优先使用同列/跨列专家，不能据此宣称单块专家已通过该布局验收。

## 高级实验入口

| 脚本 | 实现模块（包名之后） | 限制 |
| --- | --- | --- |
| `diagnose_smolvla.py` | `workflows.diagnose_smolvla_cli` | 多阶段诊断；部分阶段假设普通/EMA 两类已有检查点 |
| `smolvla_deployment_sweep.py` | `workflows.smolvla_deployment_sweep_cli` | 部署配置对照 |
| `collect_smolvla_recovery.py` | `data.collect_smolvla_recovery_cli` | 恢复示范采集 |
| `collect_smolvla_dagger.py` | `data.collect_smolvla_dagger_cli` | 专家接管与 DAgger 实验 |
| `prepare_smolvla_retraining.py` | `learning.prepare_smolvla_retraining_cli` | 再训练数据与命令准备 |
| `probe_smolvla_retrain.py` | `learning.probe_smolvla_retrain_cli` | 再训练候选短验证 |
| `train_after_collection.py` | `learning.train_after_collection_cli` | Linux PID/锁编排，要求特定 100 回合采集契约 |
| `run_smolvla_improvement.py` | `workflows.run_smolvla_improvement_cli` | 带门控的实验流水线 |
| `improve_smolvla_50.py` | `workflows.improve_smolvla_50_cli` | 扩展实验编排 |

这些入口保留已有研究功能，其整条流水线尚无完整验收记录。使用前检查 `--help`、数据与实际检查点；不存在的 EMA 权重不得视为已产生，单阶段准备成功也不等于整条流水线成功。部分编排默认依赖仓库内 `examples/`，适用于源码安装；wheel 主流程使用模块入口。

## 结果追踪

每个实验使用独立 `outputs/<run>`，保留命令、配置、环境版本、Git 版本/未提交差异、数据版本、实际检查点及 SHA256、场景与推理随机种子。普通/EMA 权重和部署后处理分开标记。开发随机种子与最终验收随机种子隔离。

比较时保持场景、随机化、控制频率、最大步数、数据统计及随机种子一致。模型评估显式传 `--dataset-root`；默认目录仅是便利约定。原始数据、权重、缓存及日志不纳入 Git。新增功能按架构职责归属正式包，不在示例转发脚本中写业务。

## 2026-10 控制审计与基线产物

| 产物 | 内容 |
| --- | --- |
| `src/a3_dual_arm_sim/audits/control_20261013/` | 机械臂控制审计：观测混合时刻、限速分派差异、专家动作回放逐位一致；`probe.py` 可重放 |
| `scripts/probe_scene_determinism.py` + `outputs/determinism_*.json` | 场景确定性探针：同 seed 场景哈希跨 revision 对比（带档位随机化的现时代 2184b60→HEAD 192/192 逐位一致） |
| `outputs/smolvla_grid_head40k_e8_s13000/` | 40k 普通权重（SHA `fbe59160…`）8 回合/格六格新基线：2×5 basic/medium/advanced = 4.25/3.38/2.88，2×10 全 0（文档化迁移测试） |
| `outputs/expert2x*_head*.json` | 专家基线：2×5 col1 三档 500 步 100%（约 410 步）；随机列 75%（5 个失败全为 500 步预算截断）；2×10 500 步 ≈10/20（2/4 批次，受预算截断），满预算 18.95–20.0（成功率 85–100%），真实失败族为运输滑落/阶段超时/释放后不直立 |
| `outputs/f1pair_* f1model_* f2pair_* f3_prompt_* speed_* p2e_*` | 单因素配对归因（观测同步 / 限速分派 / 指令风格 / 运输提速 / 预检容差），逐 seed 记录；采纳结论见对应分支提交信息与 [动作与数据契约](contracts.md#观测时序与限速语义) |

同 seed 场景自 2184b60 起可复现，跨代数值（含 2026-10-12 六格表）只作历史参考；比较一律以新基线与逐 seed 配对为准。
