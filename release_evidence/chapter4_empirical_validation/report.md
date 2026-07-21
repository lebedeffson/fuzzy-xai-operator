# Chapter 4 empirical validation evidence

## Measured result

- Dataset: Breast Cancer Wisconsin (Diagnostic) (569 objects, 30 features).
- Checkpoints: 30 unique measured model states.
- Selected case: `case_real_001`; forgetting epochs: [9].
- Rare subgroup: `rare_unsupervised_cluster_001`, fixed before training and case selection.
- Native rule: `tree_leaf_11`; target prediction changes from 1 to 0 after measured suppression.
- Cross-model contracts: 5.

## Controlled boundary

`object_85_controlled_story_fixture` remains a contract/visualization fixture and is not a real training result.
The empirical case is `case_real_001`; it is not renamed to object 85.

## Release boundary

The computational empirical gate passes. The release tag remains blocked because the independent pilot is
`planned_not_run` and the regulated-domain dictionary is awaiting external review. No human-comprehension or
clinical-validity claim is permitted.
