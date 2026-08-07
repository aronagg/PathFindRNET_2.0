# Post-Review Holdout Audit

Audit date: 2026-08-08. This audit examined filenames and canonical manifest
membership only. It did not inspect labels, cluster assignments, reference metrics, or
any HG-SMG-TC output.

## Scope and Method

The five fixed scene directories under `data/raw/` were recursively enumerated for
video files. Their repository-relative paths were compared exactly with unique
`source_recording_file` values in the 67,029-row canonical trajectory manifest.
Processed scene directories were also enumerated. They contain aggregate products and
no additional timestamp-named recording shards outside the raw inventory.

Canonical manifest SHA-256:
`0b26f59168f0c3996290cc1832ddd423321a5988e27fca93e78d2a02ecde7a03`.

The SHA-256 of the UTF-8, newline-separated, sorted set of 115 canonical recording
paths is:
`c6a9c28cc7650aa5b0d9d92a4c42226a2044df149adaf2ed0e784564a6d00b73`.

| Scene | Canonical trajectories | Manifest recordings | Raw recordings | Raw recordings absent from manifest |
| --- | ---: | ---: | ---: | ---: |
| bellevue_116th_ne12th | 2,321 | 21 | 21 | 0 |
| bellevue_150th_newport | 9,448 | 24 | 24 | 0 |
| bellevue_150th_eastgate | 26,266 | 24 | 24 | 0 |
| bellevue_150th_se38th | 9,060 | 23 | 23 | 0 |
| bellevue_ne8th | 19,934 | 23 | 23 | 0 |
| **Total** | **67,029** | **115** | **115** | **0** |

## Finding

No pristine unused recording from the same five scenes was found. No recording can be
frozen as a `post_review_extension_holdout`, and no holdout is fabricated. The later
evaluation must therefore be called a **locked post-review extension evaluation**, not
a pristine prospective test. The defensible prospective safeguard is the present
freeze before the first HG-SMG-TC test execution.

## Reproduction Command

```powershell
$m = Import-Csv publications/hg-msa-tc-future-transportation/data/manifests/trajectory_manifest.csv
$scenes = 'bellevue_116th_ne12th','bellevue_150th_newport','bellevue_150th_eastgate','bellevue_150th_se38th','bellevue_ne8th'
foreach ($scene in $scenes) {
  $manifest = @($m | Where-Object scene_id -eq $scene | Select-Object -ExpandProperty source_recording_file -Unique)
  $raw = @(Get-ChildItem "data/raw/$scene" -File -Recurse | Where-Object Extension -match '^\.(mp4|avi|mov|mkv|m4v)$')
  [pscustomobject]@{scene=$scene; manifest=$manifest.Count; raw=$raw.Count}
}
```
