# GitHub Pages Publication Setup

Repository: `aronagg/PathFindRNET_2.0`

Existing public site:

`https://aronagg.github.io/PathFindRNET_2.0/`

## Detected Local Site Source

This checkout contains a root `docs/` folder, but no `docs/index.*`, no
repository-level `_config.yml`, no `mkdocs.yml`, no `.github/workflows/` Pages
workflow, and no local `gh-pages` branch. Therefore the active GitHub Pages source
cannot be safely determined from this checkout alone.

## Safe Action Taken

The existing dataset website files were not modified directly. A proposed
publication page was created at:

`publications/hg-msa-tc-future-transportation/site/publications/hg-smg-tc/index.md`

Expected public URL after integration:

`https://aronagg.github.io/PathFindRNET_2.0/publications/hg-smg-tc/`

## Manual GitHub Pages Steps

1. Open repository Settings -> Pages in GitHub.
2. Confirm whether Pages deploys from `docs/`, `gh-pages`, GitHub Actions, or
   another source.
3. If the source is `docs/`, copy the proposed page to
   `docs/publications/hg-smg-tc/index.md` and add a homepage/publications link.
4. If the source is `gh-pages`, copy or transform the page into the corresponding
   branch path.
5. Confirm relative figure links after deployment.
6. Replace DOI and release-tag placeholders after deposit/release.
