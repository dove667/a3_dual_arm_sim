# A3 机械臂控制审计（2026-10-13）

结论：确认观测同步缺陷和跨入口动作处理不一致；尚未证明它们导致模型成功率下降。当前 benchmark 的专家动作记录→关节目标回放链路，在本次 3 × 160 步验证中精确一致。此次只新增审计材料，未修改控制器、模型或训练数据。

## 环境与范围

- 本地仓库 `/home/eter/桌面/workspace/a3_dual_arm_sim`，revision `29fe90e423259f388ebc061926c3ec860b9edc74`。
- Python 3.13.12、MuJoCo 3.6.0、EGL；`configs/cookie_batch.yaml`；物理频率 1000 Hz，控制频率 20 Hz。
- 回放 seeds 0/1/2，默认 cookie reset 随机化；无箱体随机化；每个 seed 160 控制步，关闭相机。专家 A3CookieBatchExpert，全部到达 DESCEND_TO_PLACE，未完成整集。
- 图像同步探针：seed 0，开启三路相机；左肩目标增加 0.4 rad，运行 30 步；使用独立 MjData 副本 forward，不改变被测轨迹。
- 未加载神经网络 checkpoint，也未进行模型修复前后成功率评测。

## 已确认问题

### 1. 步进后的观测混合了两个时刻

`sim/env.py:257–277` 在最后一次 `mj_step` 后直接取观测，没有刷新派生状态。qpos/qvel 是步进后的值，而 site_xpos、几何和相机位姿仍落后一个物理子步。本配置下为约 1 ms，并非一个 50 ms 控制步。

独立数据副本 forward 对照结果：末端位置最大偏差 **0.4262 mm**；前相机单像素通道最大差值 **97/255**，每帧平均绝对差值最大 **0.1246/255**。后者表明整体图像差异较小，不能由最大像素差值推断模型性能影响很大。

此问题可能影响细粒度接触判定、位姿诊断及视觉/本体状态一致性。当前 SmolVLA 输入是图像和关节状态，不直接读取 eef_pose；训练数据也可能包含同样的时序偏差。需要固定 checkpoint 的配对评测才能确认净影响。

建议修复方向：在对外生成观测与任务判定前同步派生状态；对 reset、step、图像和接触传感器建立一致的采样时刻。修改会改变既有观测，必须保留原基线。

### 2. 限速依据构造时模式，而非本步识别出的动作模式

`sim/env.py:236` 会在 joint_position 环境中接受 14 维笛卡尔动作，但 `_limit_joint_action` 在 `:295` 按 `self.action_mode` 分支：cartesian_delta 环境对全部关节增量同比缩放，joint_position 环境逐轴裁剪。因此相同初始状态和相同笛卡尔请求，在两个入口得到不同轨迹。

单步复现最大关节目标差值 **0.03338 rad（约 1.91°）**。`workflows/run_cookie_batch_cli.py` 显式创建 cartesian_delta 环境，benchmark 的 `create_env` 默认 joint_position，故跨入口专家行为不可直接视为相同。反方向在 cartesian_delta 环境回放 16 维动作会抛维度错误；这是接口非对称性，不是当前模型默认入口的故障。

建议统一动作分派和限速语义，并以实际应用的 16 维目标回放验证。当前 benchmark 采集与模型部署均使用 joint_position 环境，因此该跨入口差异尚未证明污染了此路径。

## 已验证的正常链路

- 3 个种子各 160 步，共 **480 对控制步**：专家运行所得 applied_action 在独立、同 seed 环境逐步回放，最大 applied_action 差值与完整 qpos 差值均为 **0**；无安全停机。这覆盖了到达放置下降阶段的前缀，不能替代完整成功率实验。
- 当前记录器使用执行前观测与本步 applied_action 配对（`workflows/episode_execution.py:241–253`），未发现将下一时刻观测误配为当前动作输入的问题。
- 16 维关节顺序是左 7 关节、左夹爪、右 7 关节、右夹爪；夹爪 1 为开、0 为闭。8 维模型输入取左臂并补右臂保持目标，已有测试覆盖。
- `a3_front_close_left_100/meta/info.json`：100 集、42681 帧、8 维 state/action、20 fps；当前环境也是 20 Hz。元数据一致不等于逐帧历史数据重放通过。
- 单肩运动探针最大瞬时关节跟踪误差 0.04327 rad，30 步后降至 6.09e-9 rad，仿真时间 1.5 s。该探针未发现持续跟踪偏差；不代表抓取负载和全部工作空间均正常。
- 定向测试 `test_env.py test_contracts.py test_smolvla_controls.py`：**20 passed**（3.11 s）。

## 性能归因的限制与优先级

训练集 collection_summary 记录版本 `f5a416005701d813a4994f9f8b973dd5629db3b1`。对照旧版 `src/a3_dual_arm_sim/env.py` 与现版 `sim/env.py`：仅 import 路径及异常注释变化，以上控制逻辑已在旧版存在；cookie_batch.yaml 也未改变。因此不能将这两个发现描述成近期新增的控制回归。此对照不覆盖外部 MuJoCo 版本、全部资产和训练时依赖。

另有历史文件 `outputs/smolvla_fixed_layout.json`：仅 seed 1000 一集，600 步超时、5/10 个饼干入盒；实际 run 启用了箱体/饼干随机化，箱体 noise 为 0.01 m，n_action_steps=8。文件名不能作为固定布局证据；该单集也不是本次重跑结果。

要回答“是否造成性能下降”，下一步应明确发生下降的 checkpoint/基线，固定配置、MuJoCo/资产、随机种子、推理种子、执行窗口、最大步数和随机化协议，分别切换观测同步修复与动作处理修复做配对评测，记录 raw/requested/applied/actual 动作、限速比例、接触、阶段及全部尝试。分别归因，不能一次修改多个因素后仅凭分数变化下结论。

## 复现

在仓库根目录运行：

```bash
MUJOCO_GL=egl python src/a3_dual_arm_sim/audits/control_20261013/probe.py
MUJOCO_GL=egl python -m pytest tests/test_env.py tests/test_contracts.py tests/test_smolvla_controls.py -q -p no:cacheprovider
```

数值见同目录 `results.json`。脚本只写审计结果，不修改仿真实现。
