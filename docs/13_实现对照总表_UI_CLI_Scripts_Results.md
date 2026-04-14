# 实现对照总表（UI / CLI / Scripts / Results）

> 本文档是当前项目“怎么用、产物在哪里”的单一事实来源（Single Source of Truth）。  
> 当其他文档与本文档冲突时，以本文档为准。

## 1. UI 能力对照（`app/demo_streamlit.py`）

1. **对弈**
   - 智能体选择与模型加载
   - 单步、自动、一步到终局
   - 棋盘与胜负展示
2. **日志与回放**
   - 回放滑条
   - 最近动作日志
   - 案例导出：`results/cases/*.json`
3. **训练监控**
   - 自动扫描 `results/*log*.csv`
   - 曲线：reward / win rate / epsilon / loss / entropy / turns
   - 图像浏览：`results/figures`（含关键词过滤）
4. **一键实验**
   - 一键创新训练（DQN）
   - 一键 Arena 对战
   - 一键 Arena 出图
   - 单按钮全流程：训练 -> Arena -> 出图
   - 路径口径：
     - 一键训练默认模型：`results/models/dqn_ui_run/agent0_dqn.pt`
     - 一键 Arena 默认模型：`results/models/dqn_innovation_run/agent0_dqn.pt`（若存在）
     - 全流程会先训练 `dqn_ui_run` 再用该模型跑 Arena

## 2. CLI 子命令对照（`src/main.py`）

### 2.1 训练
- `train`（Tabular Q）
- `train-dqn`
- `train-dqn-pool`
- `train-a2c`
- `train-ppo`
- `train-az`
- `train-bc`

### 2.2 评估
- `eval`
- `eval-agents`
- `eval-trained-agents`
- `eval-trained-vs-random`
- `eval-generalization`
- `run-arena`

## 3. 脚本对照（`scripts/`）

1. `run_full_pipeline.py`
   - 作用：一键跑训练 + Arena + 出图
2. `plot_arena_results.py`
   - 作用：基于 Arena 输出生成三类图
3. `run_multiseed_training.py`
   - 作用：多随机种子批量训练并汇总
4. `run_experiment_matrix.py`
   - 作用：多智能体两两矩阵评估
5. `plot_deep_training.py` / `plot_eval_matrix.py` / `plot_vs_random_bars.py`
   - 作用：训练与评估可视化

## 4. 结果目录对照（`results/`）

1. **训练日志**
   - `results/*train*_log*.csv` 或 `results/*_log.csv`
2. **模型**
   - `results/models/**`
3. **Arena 明细与汇总**
   - `results/arena_games*.csv`
   - `results/arena_summary*.json`
4. **图表**
   - `results/figures/*.png`
5. **运行元数据**
   - `results/runs/*.json`
6. **案例追溯**
   - `results/cases/*.json`

## 5. 命名建议（维护一致性）

1. 交互实验：后缀 `_ui`
2. 快速验证：后缀 `_smoke`
3. 创新配置：后缀 `_innovation`
4. 长程训练：后缀 `_long`

建议每次实验在文档中同步记录：命令、输入模型、输出文件、结论。
