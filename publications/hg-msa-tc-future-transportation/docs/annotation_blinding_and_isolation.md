# Annotation Blinding and Isolation

The annotation application renders only source trajectory geometry and source
imagery. It does not access homography-derived targets, automatic OD assignments,
clustering outputs, pseudo-reference labels or model-selection metrics. The
scientific model-selection protocol was frozen before manual labels were collected.

First-pass code reads only the canonical manifest, evaluation split, recording-level
tracker shards, source videos, blind queue, frozen annotation protocol, and frozen
manual scene guides. Its queue schema is an allowlist. Target, endpoint-region,
cluster, method, score, pseudo-label, class-frequency, suggested-label, other-
annotator, and consensus fields are rejected.

Annotators A and B have separate deterministic orders and separate SQLite identities.
A database identity mismatch is a hard error. The adjudication UI is a separate role
and starts only after both immutable first-pass exports exist. Raw labels are copied
into the adjudication record and are never edited.

The application package imports no target-estimation, clustering, EMAS_HG, or
split-aware runner module. It reads the small scientific freeze manifest only to
confirm that the scientific protocol exists and `independent_test` remains locked.
Real annotation access is logged as `human_annotation_preflight` or human-annotation
rendering, never as scientific test execution. Nothing is written under
`results/independent_test`.
