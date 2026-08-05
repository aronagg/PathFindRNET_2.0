param(
    [ValidateSet('protocol_pilot', 'annotator_A', 'annotator_B', 'adjudicator', 'protocol_designer')]
    [string]$Role = 'annotator_A',
    [ValidateSet('single', 'grid')]
    [string]$Mode = 'single',
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$PublicationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PublicationRoot)
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$App = Join-Path $PublicationRoot 'code\annotation_app\app.py'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repository virtual environment not found: $Python"
}

& $Python -m streamlit run $App --server.port $Port -- --role $Role --mode $Mode
