# Public Data Archive Plan

## Archive Target

Recommended archive: Zenodo, Figshare, OSF, or an equivalent DOI-minting
repository.

DOI placeholder: `[DOI to be added after archive deposit]`

Archive URL placeholder: `[Archive URL to be added]`

## Suitable for GitHub

- source code and tests;
- compact CSV metric summaries;
- final synthesis outputs;
- manuscript-support Markdown files;
- small generated figures;
- environment files and reproduction scripts;
- frozen configuration files and hash manifests.

## Suitable for DOI Archive

- release snapshot of `publications/hg-msa-tc-future-transportation/`;
- final figures and tables;
- split manifests and checksums if size permits;
- polygon-rule reference protocol and generated labels if permitted;
- homography correspondence tables, matrices and residual summaries;
- baseline and HG-SMG independent-test metric summaries;
- public-release documentation and citation metadata.

## Too Large or Unsuitable for GitHub

- raw videos;
- full trajectory-level intermediate tables when compact summaries are enough;
- previous review ZIP packages;
- local caches and virtual environments;
- large model weights.

## Must Not Be Redistributed Without Review

- Google Maps, Google Earth or map screenshot rasters;
- third-party satellite imagery;
- raw videos if dataset license does not permit redistribution;
- manual SQLite annotation databases containing local audit state.

## Release Sequence

1. Finalize manuscript and response files.
2. Confirm license status for dataset-derived and map-derived artifacts.
3. Create GitHub release tag `futuretransp-hg-smg-tc-v1.0`.
4. Deposit permitted release snapshot and compact data package to DOI archive.
5. Update `CITATION.cff`, manuscript availability statement and GitHub Pages page
   with DOI and release tag.
