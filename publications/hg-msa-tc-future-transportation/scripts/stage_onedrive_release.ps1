param(
  [string]$SourceRoot = ".",
  [Parameter(Mandatory=$true)][string]$StagingRoot,
  [string]$PriorityLevel = "P0",
  [string[]]$IncludePriorities,
  [switch]$DryRun,
  [switch]$VerifyChecksums
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$ManifestPath = Join-Path $SourceRoot "publications/hg-msa-tc-future-transportation/docs/onedrive_upload_manifest_prioritized.csv"
if (-not (Test-Path -LiteralPath $ManifestPath)) {
  throw "Missing prioritized manifest: $ManifestPath"
}

if (-not $IncludePriorities -or $IncludePriorities.Count -eq 0) {
  if ($PriorityLevel -match ",") {
    $IncludePriorities = $PriorityLevel.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
  } else {
    $IncludePriorities = @($PriorityLevel)
  }
}
$IncludeSet = @{}
foreach ($p in $IncludePriorities) { $IncludeSet[$p] = $true }

if (-not $DryRun -and -not (Test-Path -LiteralPath $StagingRoot)) {
  New-Item -ItemType Directory -Path $StagingRoot | Out-Null
}
$ManifestDir = Join-Path $StagingRoot "00_RELEASE_MANIFESTS"
if (-not $DryRun -and -not (Test-Path -LiteralPath $ManifestDir)) {
  New-Item -ItemType Directory -Path $ManifestDir -Force | Out-Null
}

function Get-FileSha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$rows = Import-Csv -LiteralPath $ManifestPath
$selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -ne "P3" }
$p3Selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -eq "P3" }
$copied = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[object]
$staged = New-Object System.Collections.Generic.List[object]

foreach ($row in $selected) {
  $src = Join-Path $SourceRoot $row.repository_relative_path
  $dst = Join-Path $StagingRoot $row.onedrive_target_relative_path
  if (-not (Test-Path -LiteralPath $src)) {
    $missing.Add([pscustomobject]@{ repository_relative_path=$row.repository_relative_path; upload_priority=$row.upload_priority; reason="missing_source_file" })
    continue
  }
  $dstDir = Split-Path -Parent $dst
  $action = "copy"
  $copiedNow = $false
  $verified = ""
  if (Test-Path -LiteralPath $dst) {
    if ($DryRun) {
      $action = "skip_existing_dry_run_no_hash"
    } else {
      $srcHash = Get-FileSha256 $src
      $dstHash = Get-FileSha256 $dst
      if ($srcHash -eq $dstHash) {
        $action = "skip_existing_same_hash"
      } else {
        $answer = Read-Host "Overwrite changed file? $dst [y/N]"
        if ($answer -match "^[Yy]") {
          $action = "overwrite_hash_differs"
        } else {
          $action = "skip_existing_hash_differs"
        }
      }
    }
  }
  if (($action -eq "copy" -or $action -eq "overwrite_hash_differs") -and -not $DryRun) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $copiedNow = $true
  }
  if ($VerifyChecksums -and -not $DryRun -and (Test-Path -LiteralPath $dst)) {
    $verified = Get-FileSha256 $dst
  }
  $record = [pscustomobject]@{
    repository_relative_path=$row.repository_relative_path
    onedrive_target_relative_path=$row.onedrive_target_relative_path
    upload_priority=$row.upload_priority
    size_bytes=$row.size_bytes
    source_sha256=$row.sha256
    staged_sha256=$verified
    action=$action
    copied=$copiedNow
  }
  $staged.Add($record)
  if ($copiedNow) { $copied.Add($record) }
}

$totalBytes = 0
foreach ($item in $selected) { $totalBytes += [int64]$item.size_bytes }
$copiedBytes = 0
foreach ($item in $copied) { $copiedBytes += [int64]$item.size_bytes }

if (-not $DryRun) {
  $staged | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "staged_release_manifest.csv")
  $missing | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "missing_source_files.csv")
  $copied | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "copied_files_log.csv")
  $checksumPath = Join-Path $ManifestDir "staged_release_checksums.sha256"
  $lines = foreach ($item in $staged) {
    if ($item.staged_sha256) { "$($item.staged_sha256)  $($item.onedrive_target_relative_path)" }
  }
  $lines | Set-Content -Encoding UTF8 -LiteralPath $checksumPath
  @"
# Staging Summary

Generated: $(Get-Date -Format o)

Source root: $SourceRoot

Staging root: $StagingRoot

Included priorities: $($IncludePriorities -join ",")

Selected files: $($selected.Count)

Selected size GB: $([math]::Round($totalBytes / 1GB, 3))

Missing sources: $($missing.Count)

Copied files: $($copied.Count)

Copied size GB: $([math]::Round($copiedBytes / 1GB, 3))

P3 selected: $($p3Selected.Count)

Cloud sync status: manual OneDrive verification still required.
"@ | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ManifestDir "staging_summary.md")
  @"
# PathFindRNET 2.0 Public Release Folder

This local OneDrive-synced folder stages public release artifacts for PathFindRNET 2.0 and the Future Transportation HG-SMG-TC manuscript package.

GitHub repository:
https://github.com/aronagg/PathFindRNET_2.0

GitHub Pages publication page:
https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/

Priority folders:

- P0: manuscript-submission release materials, compact reproducibility artifacts, final result tables, final figures, reference-label/protocol artifacts and minimal processed data needed for reported-result reproduction.
- P1: full technical reproducibility materials such as raw videos, tracking outputs and intermediate data needed to rebuild trajectories.
- P2: supplementary diagnostics.

Not redistributed unless separately reviewed:

- Google-derived or third-party map imagery;
- local caches and virtual environments;
- prior ZIP review packages;
- private annotation databases;
- unavailable raw YOLOv11 detector exports.

Citation/contact placeholders should be updated after final publication.
"@ | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "README_PUBLIC_RELEASE.md")
}

Write-Host "Selected files: $($selected.Count)"
Write-Host "Selected size GB: $([math]::Round($totalBytes / 1GB, 3))"
Write-Host "Missing sources: $($missing.Count)"
Write-Host "P3 selected: $($p3Selected.Count)"
Write-Host "Copied files: $($copied.Count)"
Write-Host "Copied size GB: $([math]::Round($copiedBytes / 1GB, 3))"
Write-Host "DryRun: $DryRun"
