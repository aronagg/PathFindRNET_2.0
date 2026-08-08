# Release Manifest

Recommended release tag: `futuretransp-hg-smg-tc-v1.0`

## Include in GitHub Release

- `README.md`
- `REPRODUCIBILITY.md`
- `DATA_AVAILABILITY.md`
- `LICENSES.md`
- `CITATION.cff`
- `environment/reproducibility_environment.yml`
- `scripts/reproduce_core_results.ps1`
- `scripts/reproduce_core_results.sh`
- `configs/`
- `code/`
- `tests/`
- `docs/`
- compact CSV files under `results/final_synthesis/`
- compact independent-test metric summaries;
- final synthesis figures under `figures/final_synthesis/`

## Include in DOI Archive Where Permitted

- GitHub release source snapshot;
- final synthesis tables and figures;
- reference-label protocols and generated labels if allowed;
- split manifests/checksums if size and policy permit;
- homography correspondences, matrices and quality metrics;
- compact baseline and HG-SMG metric tables.

## Exclude

- raw videos;
- `.venv`, caches and build artifacts;
- previous task ZIP files and SHA sidecars;
- Google-derived top-view rasters unless redistribution is confirmed;
- local manual annotation SQLite databases;
- large trajectory-level outputs when compact summaries are sufficient.

## Release Status

Current status: release candidate documentation prepared. DOI and GitHub release
tag are placeholders until the author creates the public release.
