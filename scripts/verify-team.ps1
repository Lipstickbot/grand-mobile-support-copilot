[CmdletBinding()]
param([switch]$Live)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'python-runtime.ps1')
$teamArgs = @((Join-Path $PSScriptRoot 'team.py'), 'verify')
if ($Live) { $teamArgs += '--live' }
& (Get-TeamPython) @teamArgs
exit $LASTEXITCODE
