# Response to Reviewer: HG Target Estimation

**Reviewer comment:** The target-estimation method requires reproducible definitions of
the scene center, polar grouping, K selection, entry/exit grouping, support threshold,
scene-specific thresholds, and failure modes.

**Response:** We agree and have added a complete mathematical and implementation-level
definition [Section X, page X, lines X-X]. The executed code uses separate coordinate-wise
median centers for entry and exit endpoints; it does not use one manually selected scene
center. Polar direction is encoded by cosine and sine, with an explicitly defined clipped,
weakly weighted radius term. Endpoint KMeans evaluates K=3,...,8 with ten initializations,
frozen seeds, Lloyd updates, and a deterministic silhouette/DB/lower-K ordering.

We now define OD support as `q_ab=n_ab/N` and target count as
`K_HG(theta)=sum_ab 1[q_ab>=theta]`, including the inclusive comparison and denominator.
The global candidate threshold grid is 0.1%, 0.25%, 0.5%, 1%, and 2%. Different final
scene thresholds are outputs of one frozen, data-dependent heuristic; they were not
manually assigned per scene. We have stated that this heuristic is not theoretically optimal.

We verified the canonical implementation against all frozen development artifacts.
Targets 10, 12, 9, 18, and 9 and all candidate/support tables were reproduced exactly
from the target-estimation split without reference labels [Table X]. No target or selected
configuration was changed.

We also added a failure analysis [Section X, Figure X]. At SE38th, internal metrics selected
seven entry and three exit regions. Four automatic entry regions map predominantly to one
manual physical approach, while exit regions merge several manual exits. Consequently,
the target of 18 counts geometric endpoint submodes rather than nine independent-test
semantic movements. We retain this unfavorable frozen result and narrow the manuscript
claim: the estimator provides an automatically estimated observed geometric-maneuver target,
not a guaranteed legal or semantic maneuver count.

Threshold and endpoint-region sensitivity are now reported without retuning [Figure X].
SE38th remains above the semantic count over the threshold grid, while target counts vary
substantially with endpoint-region granularity. We identify prospective road-branch
consolidation as future work and do not apply it to the current results.
