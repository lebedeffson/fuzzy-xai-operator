# Roadmap

Actionable open threads, derived from `RESEARCH.md`'s Open Questions and the
"Next step" lines scattered through `PROJECT_MEMORY.md`. This is not a
committed plan — it's what the ledger itself says is unfinished. Update this
file when a thread below closes or a new one opens; don't let it silently
go stale the way a chat history would.

## Blocked on external/human action (not code)

- **Independent comprehension pilot** — `planned_not_run` as of the last
  practical-closure entries. Zero participants recruited in the ledger.
- **Regulated-domain dictionary/language review** — `pending_external_review`.
- **Expert-action review** — same status, tied to the same closure.

None of these can be advanced by writing code. If asked to "finish" the
practical/action-controller program (H3 in `RESEARCH.md`), the honest answer
is that H3 itself already failed on its own merits (not supported) — these
gates were about the *other* claims (H5-A/H7-A) reaching full external
release, not about resurrecting H3.

## Open scientific/engineering threads

- **H10-C7 natural-code causal localization** — ~10 development variants
  (R0-R10M) all failed their held-out or development gates. No held-out set
  is currently open. Before starting an R11, check whether the underlying
  premise (retrieval-based localization transferring to natural incidents at
  all) is still worth pursuing, or whether this thread should be formally
  closed as `not_supported` the way H10-C5/C5b/C5c already are.
- **H10-C2 (minimum-cut-membership confirmatory)** — blocked on power, not
  on a failed result. An expanded design (~222 pipelines / ~124K cases) was
  estimated but never approved or built. This is a design/resourcing
  question, not an open code task.
- **Framework alpha (`1.4.0aX`) vs. dissertation stable-release gates** —
  see `PROJECT.md`'s Known Problems. Needs a direct answer from the user
  about whether these tracks are intentionally decoupled.
- **Adapter implementation status** (Torch/Keras/ONNX) — `docs/research_limitations.md`
  and `CHANGELOG.md` disagree. Needs a code-level check
  (`framework/fuzzyxai/fuzzyxai/adapters/`) before either doc is trusted or
  updated.

## Housekeeping

- `PROJECT_MEMORY.md` has had no new milestone entries since 2026-08-01
  (commits since then, through 2026-08-11, are CI/dependency cleanup and a
  README rewrite, not new protocol openings). If new research work starts,
  it gets a new `PROJECT_MEMORY.md` entry *and* a corresponding update to
  `RESEARCH.md`'s relevant program section — don't let the two drift apart
  again.
- This `context/` layer itself is new (built to consolidate `CLAUDE.md`,
  `AGENTS.md`, `.codex/notes/`, and `PROJECT_MEMORY.md` into one
  cross-tool-readable index). Revisit it after the next few sessions to see
  whether the compression level is actually useful in practice, and prune
  or expand accordingly.
