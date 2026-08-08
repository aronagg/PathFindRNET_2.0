# GitHub Pages Integration Plan

## Detection Result

Active Pages source: **not safely determined from local checkout**.

Evidence:

- root `docs/` exists but no `docs/index.*` was found;
- no `_config.yml` was found;
- no `mkdocs.yml` was found;
- no local `gh-pages` branch was found;
- no local `.github/workflows/` Pages workflow was found.

## Integration Strategy

Because the active website source cannot be confirmed, the existing dataset site
was left untouched. The publication landing page was prepared as a proposed page:

`publications/hg-msa-tc-future-transportation/site/publications/hg-smg-tc/index.md`

## Preferred Final Path

If the GitHub Pages source is root `docs/`, use:

`docs/publications/hg-smg-tc/index.md`

and add a homepage card or link:

```markdown
### HG-SMG-TC Future Transportation Revision

From geometric endpoint micro-modes to semantic maneuver graphs: homography-guided
vehicle trajectory clustering at complex urban intersections.

[Publication page](publications/hg-smg-tc/)
```

## Existing Homepage/Navigation Modification

No existing homepage or navigation file was modified in this task.

## Remaining Manual Checks

- Confirm GitHub Pages source in repository settings.
- Confirm public URL after deployment.
- Confirm relative links to figures and publication package.
- Add DOI and GitHub release tag placeholders after release.
