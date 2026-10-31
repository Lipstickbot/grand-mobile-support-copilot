# Windows helper for the v3 pipeline (no GNU make needed).
#   .\scripts\run-v3.ps1 -Action predict -InputPath tickets.jsonl -OutputPath pred.jsonl
#   .\scripts\run-v3.ps1 -Action serve            # http://127.0.0.1:8000/  (POST /triage, admin UI at /)
#   .\scripts\run-v3.ps1 -Action train | eval | test | templates
# Optional LLM polishing: $env:COPILOT_V3_LLM_URL = "http://127.0.0.1:8081"
param(
  [ValidateSet("predict", "serve", "train", "eval", "test", "templates")][string]$Action = "serve",
  [string]$InputPath, [string]$OutputPath, [int]$Port = 8000,
  [string]$Python = ".\.venv-soup\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
switch ($Action) {
  "predict"   { & $Python -X utf8 -m support_copilot.v3 predict --input $InputPath --output $OutputPath }
  "serve"     { & $Python -X utf8 -m support_copilot.v3 serve --host 127.0.0.1 --port $Port }
  "train"     { & $Python -X utf8 -m support_copilot.v3 train }
  "eval"      { & $Python -X utf8 scripts\eval_v3.py }
  "templates" { & $Python -X utf8 scripts\mine_templates_v3.py }
  "test"      { & $Python -X utf8 -m pytest -q tests\test_v3.py }
}
exit $LASTEXITCODE
