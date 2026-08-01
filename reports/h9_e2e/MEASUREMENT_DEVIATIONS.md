# H9 E2E Measurement Deviation Log

The first open execution from implementation `91eae88` produced 1,080 timing
rows and the frozen status `H9_E2E_TARGET_NOT_MET` (median overhead
0.1956712491765761; p95 3.1168128210462305). The runner did not emit the
required grouped summary or RAM/VRAM telemetry and used non-protocol output
directory names.

The follow-up change adds telemetry and canonical output paths only. It does
not change models, explainers, batch sizes, repetitions, targets, status rules,
or scientific claims. The first negative result is retained here and is not
discarded.
