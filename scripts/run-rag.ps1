[CmdletBinding()]
param(
    [ValidateSet('serve','predict','check')][string]$Action='serve',
    [string]$InputPath,
    [string]$OutputPath,
    [switch]$TemplateOnly,
    [switch]$LexicalOnly,
    [switch]$UseAdapter,
    [switch]$KeywordClassifier,
    [string]$EvaluationReport,
    [ValidateSet('smoke','rag-v2')][string]$AdapterVersion,
    [ValidateSet('stream','resident','llamacpp')][string]$Backend,
    [int]$Port=8000
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$pythonPath=Join-Path $projectRoot '.venv-soup/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv-soup is missing. See RAG.md.' }
$savedLocation=Get-Location
$names=@('HF_HUB_OFFLINE','HF_HUB_DISABLE_TELEMETRY','COPILOT_GENERATION','COPILOT_DENSE','COPILOT_USE_ADAPTER','COPILOT_BACKEND','COPILOT_ADAPTER_PATH','COPILOT_GGUF_ADAPTER_PATH','COPILOT_CLASSIFIER','COPILOT_EVALUATION_REPORT_PATH')
$savedEnv=@{}
foreach($name in $names) { $savedEnv[$name]=[Environment]::GetEnvironmentVariable($name,'Process') }
try {
    Set-Location -LiteralPath $projectRoot
    $env:HF_HUB_OFFLINE='1'
    $env:HF_HUB_DISABLE_TELEMETRY='1'
    $env:COPILOT_GENERATION=if($TemplateOnly){'0'}else{'1'}
    $env:COPILOT_DENSE=if($LexicalOnly){'0'}else{'1'}
    $env:COPILOT_CLASSIFIER=if($KeywordClassifier){'0'}else{'1'}
    if($EvaluationReport) {
        $env:COPILOT_EVALUATION_REPORT_PATH=if([IO.Path]::IsPathRooted($EvaluationReport)){$EvaluationReport}else{Join-Path $projectRoot $EvaluationReport}
    }
    $env:COPILOT_USE_ADAPTER=if($UseAdapter -or $AdapterVersion){'1'}else{'0'}
    if($AdapterVersion) {
        $adapterRun=if($AdapterVersion -eq 'rag-v2'){'qwen3-8b-rag-v2'}else{'qwen3-8b-smoke'}
        $env:COPILOT_ADAPTER_PATH=Join-Path $projectRoot ".private/soup-runs/$adapterRun"
        $env:COPILOT_GGUF_ADAPTER_PATH=Join-Path $projectRoot ".private/models/$adapterRun-lora.gguf"
    }
    if($Backend) { $env:COPILOT_BACKEND=$Backend }
    $runArgs=@('-X','utf8','-m','support_copilot.cli',$Action)
    if($Action -eq 'predict') {
        if(-not $InputPath -or -not $OutputPath) { throw 'predict requires -InputPath and -OutputPath' }
        $runArgs+=@('--input',$InputPath,'--output',$OutputPath)
    }
    if($Action -eq 'serve') { $runArgs+=@('--host','127.0.0.1','--port',"$Port") }
    & $pythonPath @runArgs
    $ragExitCode=$LASTEXITCODE
} finally {
    Set-Location -LiteralPath $savedLocation
    foreach($name in $names) { [Environment]::SetEnvironmentVariable($name,$savedEnv[$name],'Process') }
}
exit $ragExitCode
