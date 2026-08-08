# OneDrive Upload Manual Instructions

Generated: `2026-08-08T23:23:19Z`

Public OneDrive folder:

`https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C`

## Prepare A Local Sync Folder

1. Open the public OneDrive folder in a browser.
2. Add or sync it to the local machine using the OneDrive client.
3. Choose the local synced folder as `StagingRoot`.
4. Do not stage directly into a folder that contains unrelated private files.

## Stage P0 First

Dry run:

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\Path\To\OneDrive\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0 -DryRun
```

Copy and verify checksums:

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\Path\To\OneDrive\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0 -VerifyChecksums
```

## Stage P1 Later

```powershell
.\publications\hg-msa-tc-future-transportation\scripts\stage_onedrive_release.ps1 -SourceRoot . -StagingRoot "C:\Path\To\OneDrive\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release" -IncludePriorities P0,P1 -VerifyChecksums
```

## Synced-Folder Comparison After Upload

After OneDrive sync is complete:

```powershell
$env:ONEDRIVE_RELEASE_ROOT = "C:\Path\To\OneDrive\PathFindRNET_2.0_FutureTransportation_HG_SMG_TC_release"
.\.venv\Scripts\python.exe .\publications\hg-msa-tc-future-transportation\scripts\generate_task16_onedrive_staging_docs.py
```

This creates `onedrive_existing_inventory.csv`, `onedrive_missing_artifacts.csv`, `onedrive_mismatched_hashes.csv`, and `onedrive_extra_artifacts_review.csv` when the environment variable points to a valid folder.

## Do Not Upload

- local caches and virtual environments;
- prior ZIP review packages;
- Google-derived or third-party map imagery unless redistribution rights are documented;
- private annotation databases unless intentionally redacted and documented;
- unrelated dirty files outside the publication release scope.
