param(
    [Parameter(Position = 0)]
    [ValidateSet('validate-preregistration', 'list', 'plan', 'hg-smg-development')]
    [string]$Command = 'validate-preregistration',
    [string]$Experiment,
    [ValidateSet('preflight', 'full-split', 'uatp', 'pcms')]
    [string]$Stage,
    [string]$ConfirmDevelopmentOnly
)

$ErrorActionPreference = 'Stop'
$PublicationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent (Split-Path -Parent $PublicationRoot)
$Python = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    $Python = 'python'
}
$Cli = Join-Path $PublicationRoot 'code\reproduction\cli.py'
$Arguments = @($Cli, $Command)
if ($Command -eq 'plan') {
    if (-not $Experiment) {
        throw 'The plan command requires -Experiment.'
    }
    $Arguments += @('--experiment', $Experiment)
}
if ($Command -eq 'hg-smg-development') {
    if (-not $Stage) {
        throw 'The hg-smg-development command requires -Stage.'
    }
    $Arguments += @('--stage', $Stage, '--confirm-development-only', $ConfirmDevelopmentOnly)
}
& $Python @Arguments
exit $LASTEXITCODE
