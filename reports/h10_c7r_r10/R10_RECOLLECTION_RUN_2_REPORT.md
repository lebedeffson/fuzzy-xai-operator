# R10 causal recollection: technical barrier run 2

Run `30571013615` evaluated the unchanged ten-incident selection at commit
`67a6d9180384de87eb04439809c4b3726d9d2aa6`. The input audit passed all
checks, including selection and registry hashes, pinned image digests, and
zero observable Gold fields.

Eight incidents passed `R10_RUNTIME_READY`. No development metric was scored,
no held-out was created or opened, and the scientific result remains
`NOT_EVALUATED`.

Two incidents failed with collector timeouts:

- `Textualize__textual-5829`: one registered parametrized node ID could not be
  resolved. The generic fallback broadened execution to the complete test
  tree, reached 99%, and timed out.
- `python__mypy-19290`: the test reached the registered failure, but the
  prefix aggregate key included volatile value details. This created 587,904
  output rows and made the final flush exceed the container timeout.

Both failures are general collector defects and do not depend on Gold or
retrieval metrics. Collector v6 scopes pytest fallback to the registered test
files and aggregates the old event prefix by structural execution path while
retaining detailed values in the bounded 20,000-event tail. The selection,
incident order, image lock, timeout, event schema, retrieval, and gates remain
unchanged.

Artifact SHA256:
`a50001cd2c4ae964b4940a5e58f27f9734be35003b40b97e3e90654c92213bc6`.
