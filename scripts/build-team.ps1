[CmdletBinding(DefaultParameterSetName = 'Text')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Text')]
    [ValidateNotNullOrEmpty()][string]$Description,
    [Parameter(Mandatory, ParameterSetName = 'File')]
    [string]$DescriptionFile,
    [switch]$NonInteractive,
    [switch]$PrintPrompt
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'python-runtime.ps1')
$teamArgs = @((Join-Path $PSScriptRoot 'team.py'), 'build')
if ($PSCmdlet.ParameterSetName -eq 'File') {
    $teamArgs += @('--description-file', (Resolve-Path -LiteralPath $DescriptionFile).Path)
} else {
    $teamArgs += @('--description', $Description)
}
if ($NonInteractive) { $teamArgs += '--non-interactive' }
if ($PrintPrompt) { $teamArgs += '--print-prompt' }
& (Get-TeamPython) @teamArgs
exit $LASTEXITCODE
