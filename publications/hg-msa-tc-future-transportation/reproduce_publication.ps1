param(
    [Parameter(Position = 0)]
    [ValidateSet('validate-preregistration', 'list', 'plan')]
    [string]$Command = 'validate-preregistration',
    [string]$Experiment
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
& $Python @Arguments
exit $LASTEXITCODE
