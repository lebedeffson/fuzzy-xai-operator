# P19.2 wheel import-graph audit

The wheel boundary was checked from the canonical package entry points
(`fuzzyxai.__init__`, `runtime`, `cli`, operators, evidence, diagnostics and
visualization) and by searching imports from outside each candidate namespace.

Excluded research-only namespaces:

- `fuzzyxai.ai_pre_review*`
- `fuzzyxai.ai_pre_review_final*`
- `fuzzyxai.final_closure*`
- `fuzzyxai.q1_final*`
- `fuzzyxai.q1_validation*`
- `fuzzyxai.strong_confirmatory*`

No canonical runtime or CLI entry point imports the excluded namespaces.
`fuzzyxai.experiments` was deliberately retained: wheel-only manifest
validation proved that seven defended operators resolve callables there
(`h10_c5b.repository_grounded_route_audit`, `h10_c5.natural_incident_route_audit`,
`h9.end_to_end_latency`, `h10_c4.operational_utility`, `ch2.calibration`,
`ch2.equal_raw_structure`, and `ch3.controlled_critical_ruptures`). Excluding
that namespace would break the public operator manifest. References among
`q1_validation` and `q1_final` remain internal to the excluded research group.
Source files remain in the repository; only wheel discovery excludes the
disconnected packages.

`nicegui` has no import in the canonical runtime package. It is consequently
removed from base dependencies and exposed as the optional `ui` extra. The
fresh-venv gate verifies both installed metadata and the absence of NiceGUI
from the base environment.
