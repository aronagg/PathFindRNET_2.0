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

function Get-FileSha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$rows = Import-Csv -LiteralPath $ManifestPath
$selected = $rows | Where-Object { $IncludeSet.ContainsKey($_.upload_priority) -and $_.upload_priority -ne "P3" }
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
    $srcHash = Get-FileSha256 $src
    $dstHash = Get-FileSha256 $dst
    if ($srcHash -eq $dstHash) {
      $action = "skip_existing_same_hash"
    } else {
      $answer = if ($DryRun) { "N" } else { Read-Host "Overwrite changed file? $dst [y/N]" }
      if ($answer -match "^[Yy]") {
        $action = "overwrite_hash_differs"
      } else {
        $action = "skip_existing_hash_differs"
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

if (-not $DryRun) {
  $staged | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "staged_release_manifest.csv")
  $missing | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "missing_source_files.csv")
  $copied | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $StagingRoot "copied_files_log.csv")
  $checksumPath = Join-Path $StagingRoot "staged_release_checksums.sha256"
  $lines = foreach ($item in $staged) {
    if ($item.staged_sha256) { "$($item.staged_sha256)  $($item.onedrive_target_relative_path)" }
  }
  $lines | Set-Content -Encoding UTF8 -LiteralPath $checksumPath
}

Write-Host "Selected files: $($selected.Count)"
Write-Host "Missing sources: $($missing.Count)"
Write-Host "Copied files: $($copied.Count)"
Write-Host "DryRun: $DryRun"
