# Release Tagging Plan

Generated/updated for Task 15 on `2026-08-08T22:47:51Z`.

## Recommended GitHub Tag

`futuretransp-hg-smg-tc-v1.0`

## Release Contents

The GitHub release should point to:

- the repository source tree;
- `publications/hg-msa-tc-future-transportation/README.md`;
- `publications/hg-msa-tc-future-transportation/DATA_AVAILABILITY.md`;
- the GitHub Pages publication page;
- the public OneDrive folder for large redistributable artifacts.

Public OneDrive folder:

`https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84MGZhYWQ2ZmVhMDViMDhjL0lnQlBxblBYemRWd1I1NERrbHVfUzB0bkFUMWJFUlBzSk1XTHM3TVJ3ZDNPZFRJP2U9b2NpOXBM&id=80FAAD6FEA05B08C%21sd773aa4fd5cd47709e03925bbf4b4b67&cid=80FAAD6FEA05B08C`

## Tagging Steps

```powershell
git status --short
git tag -a futuretransp-hg-smg-tc-v1.0 -m "Future Transportation HG-SMG-TC reproducibility release"
git push origin futuretransp-hg-smg-tc-v1.0
```

Do not create or advertise a release tag until the manuscript-support package, GitHub Pages page and OneDrive upload manifest have been reviewed.
