# 基于机器学习的亚马逊棋博弈对抗研究

本项目为毕业设计提供一套可运行、可扩展、可写论文的机器博弈对抗系统，聚焦亚马逊棋（Amazons）场景。

## 项目模块

1. 博弈环境模块
- 文件：src/envs/amazons_env.py
- 功能：实现迷你亚马逊棋环境（移动+射箭+阻塞）

2. 智能体模块
- 文件：src/agents/q_learning_agent.py
- 功能：Tabular Q-learning（epsilon-greedy）
- 文件：src/agents/random_agent.py
- 功能：随机策略基线

3. 对抗训练与评估模块
- 文件：src/train/train_selfplay.py
- 功能：双智能体自博弈训练
- 文件：src/evaluation/evaluate.py
- 功能：训练策略对随机策略胜率评估

4. 文稿与答辩模块
- docs/01_开题报告_亚马逊棋.md
- docs/02_立项答辩材料_亚马逊棋.md
- docs/03_参考文献清单_中英文.md
- docs/04_章节引用映射.md
- docs/references.bib

## 快速运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### GPU / CUDA（必读）

**现象**：在终端里 `python -c "import torch; print(torch.cuda.is_available())"` 为 `False`，且 `torch.__version__` 显示 `+cpu`。

**原因**：当前默认的 Python 往往是 **Anaconda base**，里面常见为 **CPU 版 PyTorch**（例如 `2.7.1+cpu`）。这与显卡驱动无关，换不成 GPU。

**正确做法**：使用你已配置好的 **CUDA 版** 环境（例如 `myPytorch`）里的解释器再跑本项目：

```powershell
# 任选其一：显式路径（与你机器上 conda 路径一致）
E:\Anaconda\envs\myPytorch\python.exe scripts\check_cuda.py
E:\Anaconda\envs\myPytorch\python.exe -m src.main train-dqn --episodes 2000 --device cuda
```

或先激活环境再训练：

```powershell
conda activate myPytorch
python scripts\check_cuda.py
python -m src.main train-dqn --episodes 2000 --device cuda
```

一键脚本（默认使用 `E:\Anaconda\envs\myPytorch\python.exe`，可通过环境变量 `PYTORCH_GPU_PYTHON` 覆盖）：

- `scripts\train_dqn_gpu.bat` — DQN（PER + n-step）GPU 训练  
- `scripts\train_dqn_pool_gpu.bat` — 对手池 DQN GPU 训练  

`nvidia-smi` 若提示找不到命令，可运行：`"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"`（驱动正常时 GPU 仍可用）。

### 训练

```bash
python -m src.main train --episodes 5000
```

### 评估

```bash
python -m src.main eval --model results/models/agent0_q.json --episodes 300
```

### 多智能体对抗（简单策略）

```bash
python -m src.main eval-agents --agent0 heuristic --agent1 random --episodes 300
python -m src.main eval-agents --agent0 minimax --agent1 heuristic --episodes 100 --minimax-depth 1
```

### 深度学习自博弈训练（DQN / A2C）

```bash
python -m src.main train-dqn --episodes 2000 --size 6 --max-turns 200
python -m src.main train-a2c --episodes 2000 --size 6 --max-turns 200
python -m src.main train-ppo --episodes 2000 --size 6 --max-turns 200
```

前沿 DQN（优先经验回放 PER + n 步回报，Huber TD 损失）：

```bash
python -m src.main train-dqn --episodes 4000 --use-per --n-step 3 --log-csv results/train_dqn_rainbow.csv
```

对手池 PFSP-lite（单智能体对随机/启发式/浅层 Minimax/历史快照/贪心自身，默认开启 PER+n-step）：

```bash
python -m src.main train-dqn-pool --episodes 5000 --device cuda
```

### 训练后模型对抗评估

```bash
python -m src.main eval-trained-vs-random --agent-type dqn --model results/models/agent0_dqn.pt --episodes 300
python -m src.main eval-trained-agents --agent0-type dqn --agent0-model results/models/agent0_dqn.pt --agent1-type a2c --agent1-model results/models/agent1_a2c.pt --episodes 300
python -m src.main eval-trained-agents --agent0-type ppo --agent0-model results/models/agent0_ppo.pt --agent1-type dqn --agent1-model results/models/agent0_dqn.pt --episodes 300
```

### 批量实验矩阵（论文表格推荐）

```bash
python scripts/run_experiment_matrix.py ^
  --agent rnd:random ^
  --agent heu:heuristic ^
  --agent mm:minimax:1 ^
  --agent q0:q:results/models/agent0_q.json ^
  --agent dqn0:dqn:results/models/agent0_dqn.pt ^
  --agent a2c0:a2c:results/models/agent0_a2c.pt ^
  --agent ppo0:ppo:results/models/agent0_ppo.pt ^
  --episodes 100 --size 6 --max-turns 200 --device cuda ^
  --out-csv results/eval_matrix.csv
```

可视化胜率矩阵热力图：

```bash
python scripts/plot_eval_matrix.py --csv results/eval_matrix.csv --out results/figures/eval_matrix_heatmap.png
```

长程训练与统计图（可选）：

```bash
python scripts/plot_deep_training.py --dqn-csv results/train_dqn_long_gpu.csv --a2c-csv results/train_a2c_long_gpu.csv --ppo-csv results/train_ppo_long_gpu.csv --window 100 --out results/figures/deep_training_compare_long.png
python scripts/plot_vs_random_bars.py --csv results/eval_summary_long.csv --out results/figures/vs_random_winrate_bars.png
python scripts/plot_matrix_row_ranking.py --csv results/eval_matrix_long_models_n30.csv --out results/figures/eval_matrix_long_n30_row_ranking.png
```

## 说明

该实现先提供“可复现、可答辩”的基础版本：
- 使用 Mini Amazons（6x6，双方各1子）便于快速训练和验证方法闭环。
- 文献中已覆盖 Q-learning、DQN、PPO、多智能体博弈、自博弈、MCTS 与 Amazons 主题，支持后续扩展章节撰写。
