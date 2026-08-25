# User Context

How to work with this user on this repository. Kept separate from
`PROJECT.md`/`RESEARCH.md` because this is about collaboration style, not
the project itself — update it when the user corrects your approach, not
when the project changes.

## Role

Sole researcher/developer on FuzzyXAI — both the dissertation author and the
framework's implementer. There is no separate team to defer to on scope
questions; architectural and research-protocol decisions in `DECISIONS.md`
were made by this user (or accepted by them after AI-assisted implementation),
not inherited from elsewhere.

## Working style

- Demands the same evidence-first discipline from AI assistance that the
  framework itself enforces (see `DECISIONS.md#D002`): don't state something
  as current/true without checking the code or docs, don't paper over a
  disagreement between sources, flag it instead (see `PROJECT.md`'s "Known
  Problems" section for the pattern this user explicitly asked for).
- Actively works across multiple AI tools on the same repository — Claude
  Code and GPT/Codex, at minimum. Wants one canonical, tool-agnostic memory
  layer (`context/*.md`) rather than a Claude-specific store, so switching
  tools doesn't mean re-explaining project history. `CLAUDE.md` and
  `AGENTS.md` are the tool-specific "how to build/test/run" entry points;
  `context/` is the shared "what do we know and why" layer both should read.
- Explicitly wants history compressed into durable knowledge, not preserved
  as chronological chat/commit logs. When adding to `context/`, prefer
  updating the relevant synthesis section over appending a dated entry —
  the append-only ledger already exists at `PROJECT_MEMORY.md` for that
  purpose and must not be duplicated.
- Cares about token/context efficiency in tooling generally — uses `rtk`
  (Rust Token Killer, a token-optimized CLI proxy) for routine git/dev
  commands via a Claude Code hook. This is a personal tool preference, not
  a project dependency; don't assume other collaborators or CI have it.

## Practical notes

- Git identity: `lebedeffson`.
- Repository has no separate "team" conventions beyond what's in
  `CONTRIBUTING.md` — that file's 6 rules are the actual bar for any change,
  including AI-assisted ones.
