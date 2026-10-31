[CmdletBinding()]
param([ValidateSet('preflight', 'train', 'reload')][string]$Stage = 'preflight')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-soup/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv-soup and install experiments/soup/requirements-lock.txt first.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $python -X utf8 (Join-Path $PSScriptRoot 'soup_experiment.py') $Stage
    $taskExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExit
