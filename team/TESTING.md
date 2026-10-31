# Integration and verification policy

## Team setup

`scripts/verify-team.ps1` checks TOML syntax, the complete role roster, required
fields, compatible model efforts from the local model catalog when available,
permissions and coordination files. It also asks the installed Codex to load the
project with `--strict-config`. This does not itself prove a model executed a role.

`scripts/verify-team.ps1 -Live` additionally runs an ephemeral read-only Codex session
that invokes all twelve native agent types in batches of at most three. Each child
returns its role and role-specific instruction marker without using tools. The
parent must observe every result and close completed threads when supported. The verifier checks
native spawn events, completion results, expected markers and overlapping lifetimes.
The installed CLI JSON adapter omits hosted spawn records, so the verifier captures
the installed app-server stdio protocol with raw events enabled. It checks actual
`collaboration.spawn_agent` calls and independent child messages. The parent's final
response alone is insufficient evidence.
It stores evidence in `.team-verification/`, which is ignored by Git.
This is an invocation smoke test, not a benchmark of engineering quality.

## Future product work

The root records the actual package manager, runtime versions, and test commands in
the implementation plan. Never assume a particular application stack in this template.

- Before parallel code: agree on inputs, outputs, validation, error cases and data
  ownership. Use shared contracts so clients and servers can develop independently.
- Each writer: run relevant unit/type/lint checks supported by the selected stack.
  Add tests for meaningful behavior and failures, not implementation details alone.
- Integration: verify client -> API -> database -> AI with test data; include schema
  migration, authentication and malformed-input cases when applicable.
- QA: reproduce the main journey, loading/empty/error states, keyboard access,
  responsive layout and regression cases. Use available browser tools honestly.
- AI: test representative examples, structured-output validation, timeouts, provider
  errors, prompt injection boundaries and a deterministic test double. Label mocks;
  verify a real provider call only with authorized credentials and spend.
- Security: review auth/authz, validation, secret handling, dependency risks, unsafe
  execution and data exposure. Findings require an affected path and evidence.
- DevOps: verify installation and startup from documented commands, CI checks,
  environment placeholders, health checks and deployment rollback instructions.

Record each command, exit code and observed outcome. Distinguish PASS, FAIL,
BLOCKED and NOT RUN. Fix blocking defects and rerun affected checks. Include exact
reproduction steps for unresolved issues; do not manufacture test evidence.
