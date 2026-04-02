$ErrorActionPreference = "Stop"

Write-Host "[1/3] Install dependencies"
pip install -r requirements.txt

Write-Host "[2/3] Train self-play model"
python -m src.main train --episodes 5000

Write-Host "[3/3] Evaluate model"
python -m src.main eval --model results/models/agent0_q.json --episodes 300

Write-Host "Done."
