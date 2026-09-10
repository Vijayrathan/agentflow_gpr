# Paper-to-implementation decisions

Reference: [Advancing Precision Agriculture using GPR and ML](https://soilx.wpi.edu/pdfs/advancing_precision_agriculture_gpr_ml.pdf),
especially sections II-C/III, Figure 7, and Tables IV/VI. This is our implementation
of a paper-guided study, not a reproduction of unavailable private source code.

| Topic | Implemented decision and basis |
|---|---|
| Study target | User selected first-layer scalar Peplinski moisture. No Topp relabeling, extra outputs, or thickness input. |
| Physical scene | Preserve the already audited 750 MHz 2D datasets. The paper's scene construction is not substituted. |
| AGC | Omitted by explicit user instruction. |
| Zero-time correction | Native synthetic source clock; no per-signal alignment or normal moveout transformation. |
| Normalization | Training-fitted global raw min/max or extracted-column min/max; no per-trace amplitude normalization. |
| Peak definition | Absolute-trace local maxima, retaining signed original amplitude. Lobes are not assigned to interfaces. |
| Unspecified extraction | The 0.5% prominence, half-period separation, six slots, 42-feature schema, and area integration are our approved defaults. |
| Compact features | Select five RF-ranked descriptors; use two PCA components and 100 F-test time samples. |
| ANOVA ambiguity | Continuous-target `f_regression`; no artificial moisture classes or multioutput score aggregation. |
| RF inconsistency | Choose the later 1,000-tree/depth-3 description as fixed primary; earlier 100-tree description remains a documented inconsistency. |
| NN | Later 64–64 dense description, one linear output; explicit weight-matrix L2 convention. |
| Search details | Approved bounded grids, shared settings, five paired folds, deterministic tie breaking. Not claimed to match undocumented author search. |
| Data split | User selected existing-population 70/15/15 over the earlier new-evaluation-seed proposal. Pair IDs stay together. |
| Selection independence | Each arm/fold fits its own supervised feature selection and unsupervised transforms. |
| Main comparison | User prespecified fixed RF/peak result; tuned secondary results cannot replace it. |
| Real data | Calibration, SFCW conversion, weighted mixed-data loss and lab evaluation deferred by user. |
| Accuracy claims | New measured results only. No comparison to paper numbers with different targets/scenes/units as an equal benchmark. |

The raw benchmark, four feature representations, model comparisons, validation,
test evaluation and controls are executable. Dataset generation and forward solves
are upstream dependencies. The pipeline blocks unverified/partial output ingestion
instead of treating earlier input or uniform-moisture fixture audits as production
signal qualification.
