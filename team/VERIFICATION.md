# Verified installation and results

Checked on 2026-10-29 in this project on Windows.

| Check | Observed result |
| --- | --- |
| Desktop app | 26.1023.51421, build 22530, prod; updater reports up to date |
| CLI | `codex-cli 0.165.0-alpha.1.2` |
| Native support | `multi_agent` stable and enabled |
| Format | Standalone project `.codex/agents/*.toml` |
| Strict config | `codex app-server --strict-config --print-startup-config` passed |
| Role discovery | All twelve custom types in native spawn tool guidance |
| Models | `gpt-6.1-sol`; configured medium/high efforts recognized |
| Invocation | Twelve actual native spawn calls and twelve independent child completions |
| Instructions | Every child returned its own configured role marker |
| Parallelism | Three overlapping child lifetimes observed |
| Permissions | Read-only live test; workspace-write/on-request for future builds |
| Global config | SHA-256 unchanged before/after setup and verification |
| Product | No hackathon solution started |

Global `config.toml` SHA-256 before and after:

```text
18ffeb342d7f3a539018ecbdaf972c2e42d816b11056cf5fd3c885bfcfc03d0a
```

Successful raw trace SHA-256:

```text
986776e81903119a6db5e607b9977ac21146e8c920a0d3d86bb91af96b4e9935
```

Local evidence is in `.team-verification/`, excluded from Git. `verification.json`
records the latest check; `live-verification.json` preserves the latest successful
live evidence validation, including child results, spawn/completion counts,
parallelism and trace hash.

The initial `codex exec --json` test returned all markers but omitted native spawn
records. The verifier rejected that attempt as insufficient evidence. It was
corrected to capture `rawResponseItem/completed` using the installed app-server's
`experimentalRawEvents` option, then check native spawn arguments, successful
call outputs, child FINAL_ANSWER messages and overlapping lifetimes.

The complete app-server test succeeded. The final verifier checked its saved raw
trace using `verify --replay-trace`, avoiding unnecessary repeated model calls.
`verify-team.ps1 -Live` uses the same capture and validation path for a fresh run.
Saved-trace replay validates past evidence; after changing role definitions, run
a fresh live test to verify the new configuration.

This smoke test demonstrates native loading, instructions, invocation, completion
and parallelism. It does not benchmark engineering quality. Application integration
and deployment tests await a real project description.
