# HG-SMG Development Freeze Report

- Development freeze SHA-256: `c86830ea99b331fa8322915c33425398ac3edee76d242e0915f05c7c5ff0a14a`
- Code commit: `55bab42853ce341c75b31404ed39eb34e9841c6b`
- Protocol SHA-256: `2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6`
- Independent-test access: **none**
- Future test eligible without protocol amendment: **no**, because A8 lacks a frozen JSD threshold.

The primary A5 implementation and all completed development artifacts are immutable under this freeze. The primary pipeline is computationally valid, but protocol v1 cannot supply a reproducible A8 execution. No later task may invent the missing threshold while retaining the v1 hash.

Two complete clean executions used 500 UATP replicates and 500 within-region SAC bootstrap replicates. SHA-256 matched for SAC assignments, SMG edges, the complete UATP sequence and summary, PCMS selections, and PCMS provenance. The duplicate second-run directory is an ephemeral verification workspace; the hash comparison and exact reproduction command are retained.

No independent-test feature, assignment, reference label, manual scene guide, polygon definition, or semantic agreement metric was accessed.
