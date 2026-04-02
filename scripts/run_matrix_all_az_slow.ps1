# 9 智能体全矩阵（含 AZ Stage1/2/3），适合长时间挂机跑。
# 用法（在仓库根目录）:
#   .\scripts\run_matrix_all_az_slow.ps1 -Episodes 40
param(
    [int]$Episodes = 40
)

$py = "E:\Anaconda\envs\myPytorch\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Python not found: $py — 请改成你的 myPytorch 路径"
    exit 1
}

$out = "results/eval_matrix_all_az_n$Episodes.csv"
& $py scripts/run_experiment_matrix.py `
    --agent heu:heuristic `
    --agent mm:minimax:1 `
    --agent dqnL:dqn:results/models/agent0_dqn.pt `
    --agent a2cL:a2c:results/models/a2c_long_20260331/agent0_a2c.pt `
    --agent ppoL:ppo:results/models/ppo_long_20260331/agent0_ppo.pt `
    --agent bcL:bc:results/models/bc_long_20260331/bc_policy_from_mcts.pt `
    --agent az1:az:results/models/az_stage1_20260331/alphazero_pvnet.pt `
    --agent az2:az:results/models/az_stage2_20260331/alphazero_pvnet.pt `
    --agent az3:az:results/models/az_stage3_20260331/alphazero_pvnet.pt `
    --episodes $Episodes --size 6 --max-turns 80 --device cuda `
    --out-csv $out

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py scripts/plot_eval_matrix.py --csv $out --out "results/figures/eval_matrix_all_az_n${Episodes}_heatmap.png"
& $py scripts/plot_matrix_row_ranking.py --csv $out --out "results/figures/eval_matrix_all_az_n${Episodes}_row_ranking.png"
Write-Host "Done. CSV=$out"
