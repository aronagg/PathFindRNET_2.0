# GitHub Pages Integration Plan

Generated/updated for Task 15 on `2026-08-08T22:47:51Z`.

## Source Detection

The active GitHub Pages source could not be safely determined from this checkout. No root Pages source was modified.

Observed state:

- no root `_config.yml` was found;
- no root `mkdocs.yml` was found;
- no `.github/workflows/` Pages workflow was found;
- the local `gh-pages` branch was not present.

## Proposed Page

The publication page is staged at:

`publications/hg-msa-tc-future-transportation/site/publications/hg-smg-tc/index.md`

Web preview images are staged at:

`publications/hg-msa-tc-future-transportation/site/assets/images/`

Expected public URL after integration:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## Manual Integration Steps

1. Confirm the repository Pages source in GitHub settings or Actions.
2. Copy or route `site/publications/hg-smg-tc/index.md` into the active Pages source.
3. Copy `site/assets/images/` into the corresponding public assets folder.
4. Verify relative image links after deployment.
5. Add the OneDrive folder link to the deployed page.
