param(
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

$env:PYTHONPATH = "publications/hg-msa-tc-future-transportation/code"

Write-Host "Regenerating final synthesis tables and figures from persisted outputs only..."
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\final_synthesis.py

if ($RunTests) {
    Write-Host "Running publication tests..."
    .\.venv\Scripts\python.exe -m pytest publications\hg-msa-tc-future-transportation\tests -q
}

Write-Host "Done. No clustering, target estimation, homography calibration, baseline generation, or reference-label generation was run."
