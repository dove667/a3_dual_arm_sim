# 项目记忆（A3-Mujoco）

本文件承接 Codex 本地项目「A3-Mujoco-本地」的对话内容，供后续在 Hermes 中推进本项目时使用。它记录项目环境事实、已确认的约定、以及 Dove 的学习进度与承接点。权威状态以仓库、AGENTS.md 和服务器的实际输出为准，本文件提供上下文与线索。

## 来源与可回溯位置

- Codex 本地项目「A3-Mujoco-本地」，项目 id `4a885bd6-00f7-480e-9a41-7b8a5f48a34e`（登记于 `~/.codex/.codex-global-state.json` 的 `local-projects`）。
- 会话 1：`01a15884-9bee-7fa2-a355-b05e0f76d497`（2026-10-20）——生成 AGENTS.md、建立个人 fork、修复并装齐 L40 环境依赖。
- 会话 2：`01a158a1-0da8-7671-9e51-535c7c2bde66`（2026-10-20）——具身智能入门讲解、L40 的 GitHub SSH 认证、.gitignore 与提交流程、诊断视频 H.264、midterm 提交、学习路线与 RL 讨论。
- 原始记录：`~/.codex/sessions/2026/10/20/rollout-<thread-id>.jsonl`。
- 旧记录里的路径 `/Users/dove/Desktop/智能体系统/Project/a3_dual_arm_sim` 已不存在：仓库上移为 `/Users/dove/Desktop/智能体系统/a3_dual_arm_sim`，遇到旧路径按新路径处理。
- 后续在本项目的工作完全在 Hermes 中进行，不再使用 Codex。

## 环境与已核实事实（2026-10-21 复核）

| 项目 | 状态 |
|---|---|
| Mac 副本 | `/Users/dove/Desktop/智能体系统/a3_dual_arm_sim`，`main = origin/main = 0e15e2d`，工作区干净 |
| Mac 远端 | `origin` = `https://github.com/dove667/a3_dual_arm_sim.git`（HTTPS，用本机 gh 凭据）；`upstream` = `https://github.com/Eter0109/a3_dual_arm_sim.git` |
| L40 副本 | `/data1/zsh/a3_dual_arm_sim`，`main = 0e15e2d`，工作区干净 |
| L40 远端 | `origin` = `git@github-dove:dove667/a3_dual_arm_sim.git`（SSH 主机别名，私钥 `~/.ssh/id_ed25519_github_dove`） |
| L40 环境 | `.venv` Python 3.12.12；`lerobot 0.5.1`（项目锁定）；`torch 2.10.0+cu128`，`torch.cuda.is_available() == True` |
| L40 视频组件 | `.runtime/ffmpeg`（conda 环境），用于满足 LeRobot 视频读取的 FFmpeg 版本要求 |
| 无窗口渲染 | `MUJOCO_GL=egl`；`a3-sim smoke` 关闭相机渲染，只有带诊断视频的任务评测才验证渲染链路 |

选 L40 而非 4090 的原因：4090 那台服务器当时没有空闲卡。L40 的 GPU 稀缺，不要空占显存。

建立这套环境时用的安装命令（记录在 `~/.codex/memories/MEMORY.md`，当前环境已核实一致）：

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -e ".[dev,act,dataset,train]"
```

当时的验证覆盖：`pip check`、MuJoCo smoke、CUDA 前向/反向、三路 256×256 相机、22 项聚焦测试。FFmpeg 相关依赖单独放在 `.runtime/ffmpeg`，原因是服务器的 FFmpeg 版本高于 LeRobot 视频读取组件支持的版本（TorchCodec 与 FFmpeg 7 的兼容处理）。模型权重和数据集当时尚未下载。

## 已确立的约定

- **单端修改、Git 同步**：在一端修改并提交，`git push`；另一端 `git pull --ff-only`。数据集、权重和生成产物各自本地管理，不入库。
- 训练与模型推理放在 L40；Mac 用于编辑、连接服务器和查看结果。本地 Python 用 `AI` 环境。
- 提交信息用英文 conventional commits（如 `fix(dataset): ...`）。
- 诊断视频默认自动转 H.264（提交 `f1306d1`），文件名保持 `front.mp4`、`left_wrist.mp4`、`right_wrist.mp4`，可直接在 VS Code 预览；转换失败时保留原视频。
- 数据接口兼容修复（`1b2339c`）：`audit_smolvla_data_cli` 去掉了 LeRobot 0.5.1 不支持的参数。
- `midterm.pdf`（课程中期要求）位于仓库根目录，已提交（`0e15e2d`）。

## 项目主线与 Dove 的进度

项目内容与代码细节见 `AGENTS.md`、`docs/workflow_guide.md`、`docs/contracts.md`、`docs/act.md`、`docs/smolvla.md`。

**课程中期要求**：跑通项目、调研至少三种机器人策略方法、设计 Agent 驱动的实验闭环；本阶段不要求提升模型性能。两个任务：2×5（10 块饼干）与 2×10（20 块）。

**Dove 已经理解的内容**（承接时可直接沿用，不必重讲）：

- `reset / policy.act / env.step` 闭环；观测、动作、episode、rollout 的含义；20 Hz 控制频率。
- 16D 关节动作（双臂各 7 关节 + 夹爪开度）；依赖数据用左臂 8D，部署时补右臂保持目标。
- 规则专家是人工编写的状态机 + 几何计算 + 反馈控制，读取仿真特权状态（物体精确位姿、接触力），既是基线也是示范标签来源。
- 模仿学习与 RL 的区别、闭环分布偏移、DAgger、IK 与正运动学、`a3-sim inspect` 输出含义（`nq/nv/nu/nsensor` 的来源）。
- ACT 与 SmolVLA 的输入、输出、训练目标差异（动作块、ResNet18 骨干 + 条件 VAE；SmolVLA 多语言输入、Flow Matching、action expert）；`a3-sim` 各子命令的作用。

**Dove 的偏好**：讲解像开发者带人一样一次讲一小段，不要一次给太长规划；不要让他通读代码或长 JSON，替他提取关键字段；他对 ML 与智能体系统熟悉，具身与仿真是新领域。

**待推进**：2×5 专家基线复核 → 数据与训练链路理解 → 三种方法调研对比（ACT / Diffusion Policy / VLA）→ Agent 驱动实验闭环方案设计。RL 是可选扩展方向（含模仿初始化 + RL、Residual RL、DAgger、HIL-SERL），环境已具备 Gymnasium 接口与「入盒比例」奖励，但仓库尚无 RL 训练入口。
