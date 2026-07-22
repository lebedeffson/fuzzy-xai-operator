# H10 v19 repository methodology review

## Verdict

`NEEDS_REVISION_SCIENTIFIC_RELEASE_BLOCKED`

The supplied archive is internally checksummed and the reported numerical
effects can be recomputed from its frozen `raw_results.csv`. Import-level
oracle and baseline separation is present. The primary localization and repair
claims are nevertheless not independently adjudicated.

## Blocking finding

`experiments/h10/mutations.py` obtains `source_nodes` from the oracle's static
`MutationSpec` catalog. The oracle's complete leaf-to-source mapping equals the
mapping in `fuzzyxai.audit_h10.taxonomy.FAULT_SPECS`. The mutation changes flat
route metadata fields; it does not mutate a concrete graph node or edge and
then record that physical mutation as source truth. `repair_sets` are copied
from the same catalog-assigned source nodes.

Therefore the source-localization and repair-set endpoints partly measure
agreement between duplicated target dictionaries. Import independence alone
does not remove this semantic coupling.

## Claim boundary

- H10-L: `invalid_methodology`.
- H10-R: `invalid_methodology`.
- H10-C: secondary descriptive result from the separately implemented cut
  oracle; it is not promoted to a primary claim.
- H10-U: descriptive only.
- H10-T: supported only as deterministic byte-identical trace generation.
- Safety rates remain point estimates from the controlled cases and do not
  establish production safety.

## Preserved evidence

The original handoff reports, positive claim registry, and checksums remain
under `artifacts/h10_v19/imported_handoff/`. Frozen scoring outputs are not
modified or regenerated. Repository reproduction recalculates only derived
statistics, replay summaries, tables, figures, evidence mapping, and closure
reports.
