# H10-C5b runtime collection readiness

The repository-grounded scientific method is frozen at
`7aa72a19a70bdb5eedea520742f269bc6c26aeea`. The method lock protects
11 implementation and protocol-facing files with aggregate SHA256
`5e14cbcd583aeb447b70be022147633588f6ee9ae95dcfb637be2e399e47c9ec`.
The verified scientific implementation diff is zero.

The runtime-availability amendment is limited to deterministic replacement of
incidents that do not reproduce with a traceback. Replacement uses the next
unused candidate from the same split and repository under the preregistered
SHA256 ordering. Gold fields and method predictions cannot enter replacement
selection.

Runtime collection is not executable on the current local machine: Docker
client 29.6.2 is installed, but no daemon socket exists at
`/var/run/docker.sock`. No synthetic traceback was substituted. The current
scientific status therefore remains `H10_C5B_BLOCKED_REPOSITORY_DATA`.

The operational workflow performs ordinary source checks on GitHub-hosted
runners. Runtime collection is manual and requires a dedicated self-hosted
runner labelled `h10-c5b-runtime`. It collects development evidence first.
Runtime images must be prebuilt and pinned by immutable SHA256 digest;
containers run without network or added Linux capabilities.
Held-out collection requires a frozen development runtime lock, and the
workflow contains no official held-out scoring command.

H9-E2E-v2 remains `H9_E2E_V2_TARGET_MET` within
`registered_local_microbenchmark_pipelines`. The original negative H9-E2E and
H10-C5 evidence remains unchanged.
