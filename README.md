# Grand Mobile Support Copilot · Decentrathon 6.0

Локальный исследовательский помощник поддержки: HTTP API, RAG по правилам,
классификация обращений и черновики RU/KK/UZ/EN. В репозитории также сохранена
переиспользуемая команда из12нативных агентов Codex.

**Это прототип для сотрудника, не автономная поддержка.** Все решения требуют
человека. [Текущее состояние и ограничения](STATUS.md),
[запуск RAG/API](RAG.md), [измеренные результаты](experiments/quality-v5/RESULTS.md),
[инструкция для человеческой проверки](experiments/quality-v5/HUMAN_REVIEW.md).

Исходные обращения игроков, DOCX/ZIP задания, веса моделей, обучающие выборки,
секреты и локальные журналы не включаются. Для запуска нужны собственные
разрешённые данные и локальные артефакты, описанные в RAG.md. Сеть/платные API
не включаются автоматически. GitHub-публикация не означает публичный деплой.

## Команда Codex

Local training experiment added 2026-10-30: see
[Soup Qwen3-8B instructions](experiments/soup/README.md). Private measured results
are in `.private/SOUP-8B-RESULT.md`; checkpoints and source-derived data stay out of Git.

Twelve project-local custom agents, specialized developer instructions, bounded
parallel implementation, handoffs, integration checks and a reusable build launcher.
The repository now includes the Grand Mobile prototype described above. Agent
definitions remain project-local and separate from product runtime dependencies.

## Start a project

Open **this folder** in a new Codex chat after configuration is loaded, then write:

```text
Build this hackathon project using my agent team:
[YOUR PROJECT DESCRIPTION]
```

Or run from PowerShell:

```powershell
.\scripts\build-team.ps1 -Description 'Your project description'
```

For a long brief, save it as UTF-8 text and use:

```powershell
.\scripts\build-team.ps1 -DescriptionFile .\project-brief.txt
```

`-PrintPrompt` previews the complete prompt without starting Codex.
`-NonInteractive` uses `codex exec`; actions needing a fresh approval may fail in
that mode and must be resolved in an interactive session. The default launcher
opens an interactive CLI session and preserves normal approval handling.
The launcher does not run all agents as separate shell processes; Codex performs
native delegation according to the workflow.

The primary session is the Orchestrator. It analyzes the request, creates a plan,
assigns independent work, integrates components, runs checks, coordinates fixes,
and prepares a working local MVP. For a new project, product details and technology
choices follow the supplied description; the current Grand Mobile implementation
is documented separately in RAG.md.

## The twelve roles

All native definitions are `.codex/agents/<name>.toml`. Every file contains
`name`, `description`, `developer_instructions`, `model`, and
`model_reasoning_effort`, plus supported capability restrictions.

| Role / native name | Responsibility and expected handoff | Reasoning |
| --- | --- | --- |
| Orchestrator / `orchestrator` | Plan, dependencies, ownership, integration, acceptance | high |
| Product Strategist / `product_strategist` | Users, core journey, MVP scope, measurable requirements | high |
| System Architect / `system_architect` | Stack, boundaries, API/data/AI contracts, implementation plan | high |
| UI/UX Designer / `ui_ux_designer` | Flows, layouts, components, responsive and accessible states | medium |
| Frontend Developer / `frontend_developer` | Client UI, state, accessibility, API integration | high |
| Backend Developer / `backend_developer` | APIs, validation, business logic, auth boundaries | high |
| AI Engineer / `ai_engineer` | AI adapters, prompts, output validation, evaluation, fallbacks | high |
| Database Engineer / `database_engineer` | Schema, constraints, migrations, seeds, data contracts | high |
| Research Agent / `research_agent` | Primary-source feasibility, compatible APIs, limits, alternatives | medium |
| QA Engineer / `qa_engineer` | Integration and regression tests, reproducible defects | high |
| Security Engineer / `security_engineer` | Threat boundaries and concrete findings with evidence | high |
| DevOps Engineer / `devops_engineer` | Reproducible startup, CI, health checks, release preparation | high |

Each role uses `gpt-6.1-sol`, the model already selected in this installation and
present in its catalog. No automatic model substitution is configured. Account
availability and quota still apply. To use another available model, change both
`model` and a compatible `model_reasoning_effort` in the relevant TOML file, then
rerun verification. Role-file settings take precedence over inherited spawn defaults.

For individual requests, explicitly ask Codex to use the native type, for example:

```text
Use the native system_architect agent to review this design.
Return an architecture handoff; do not edit files.
```

Substitute any name in the table and provide the relevant files, constraints,
ownership and acceptance criteria. Codex selects that `agent_type` through its
native spawn tool. There is no invented `/agent` command or `codex --agent` flag.
The current chat may need a new session to discover newly added custom types.

## Workflow and capacity

The requested dependency flow is implemented in [team/WORKFLOW.md](team/WORKFLOW.md):

```text
User -> Orchestrator -> Product + Research -> Architect -> UX + Database
     -> Frontend + Backend + AI -> Integration -> QA + Security -> DevOps -> MVP
```

`agents.max_concurrent_threads_per_session = 3` enables up to three spawned agents
alongside the primary. The runtime can impose a lower limit. Twelve roles are
available; they run in dependency-ordered batches. A single root coordinates all
stages. `max_depth = 1` constrains V1 nesting; V2 ignores this key, so the role
instructions also prohibit recursive delegation. Finished work is accepted only
after the root checks the handoff and evidence.

Shared files, manifests, lockfiles and contracts belong to the root. Workers receive
disjoint owned paths. Shared filesystem access is not per-agent worktree isolation.
This template does not initialize Git or create worktrees in an empty folder.

## Permissions

Project `.codex/config.toml` sets parent `workspace-write`, `on-request` approvals,
and restricted shell network access. Managed requirements and runtime limits take
precedence. The setup does not change Windows sandbox implementation or any global
configuration. This existing project was already trusted during setup.

Supported feature flags disable apps, plugins, memories and the separate
`request_permissions_tool` capability for this team. Product and research roles
also disable `shell_tool`. Other roles retain shell access for inspection, builds
and tests under the parent sandbox. These restrictions deliberately keep this
starter team focused on built-in coding and search capabilities.

Custom agents inherit parent sandbox, approvals, provider and MCP server configuration.
Role files cannot replace these settings. Existing inherited MCP servers are not
removed by disabling apps/plugins; there is no universal per-role MCP allowlist in
this template. "Do not edit" is a prompt instruction, not an enforced read-only
filesystem. Start a parent session with `--sandbox read-only` when that guarantee
is required. Do not put permission escalation settings into role files and assume
they are effective.

Web search defaults to cached mode; explicit live browsing is available through
the parent CLI's supported `--search` option when needed. Shell networking remains
subject to its separate sandbox and approvals. Public deployment, paid resource
creation, destructive data changes and external messages require user authority.

## Grand Mobile RAG baseline

Local support triage implementation and exact Windows startup instructions:
[RAG.md](RAG.md). This baseline uses private supplied rules, CPU E5/BM25 retrieval
and the existing local Soup/Qwen3-8B experiment. It is separate from the reusable
Codex team configuration and does not claim production readiness.

## Verify or reuse

```powershell
.\scripts\verify-team.ps1
.\scripts\verify-team.ps1 -Live
```

The normal verifier uses Python 3.11+ standard-library TOML parsing, the installed
Codex strict parser, effective feature checks, model effort compatibility and native
spawn-tool discovery of all twelve roles. It uses the bundled desktop Python when
available, then falls back to Python on PATH. No package installation is required.

The live test uses the signed-in Codex account and quota. It runs read-only,
invokes all twelve custom types and checks instruction markers and native events.
Evidence and the machine-readable report are in `.team-verification/`. See
[team/VERIFICATION.md](team/VERIFICATION.md) for the verified installation and results.

To reuse in another repository, review and copy `.codex/agents/`, merge the project
settings in `.codex/config.toml`, merge `AGENTS.md` with the repository's existing
instructions, and copy `team/` and `scripts/`. Preserve existing configurations.
Trust the destination project through Codex's normal trust flow, start a new
session and rerun verification. Do not copy global auth, model caches or private
run logs. For another Codex version, run verification first; do not assume that an
older build supports standalone custom agents or these setting names.

Troubleshooting:

- **Roles absent:** check the working directory, project trust and a fresh session.
  Inspect `.team-verification/prompt-input.json` for the native spawn definition.
- **Unknown setting:** check the installed version and official schema. The verifier
  fails instead of silently claiming support; global configuration can also cause
  strict parsing failures.
- **Model or quota error:** choose an account-accessible model and compatible effort,
  or wait for quota. Do not treat a static configuration pass as a live invocation pass.
- **Access denied under `.codex`:** this is a protected configuration directory.
  Use the normal approval mechanism to authorize an intentional configuration edit.
- **Approval unavailable in batch mode:** continue interactively; retain sandboxing.
- **PowerShell shell-snapshot warning:** this build reports that PowerShell shell
  snapshots are unsupported; that warning alone is not an agent invocation failure.

## Files and source policy

- `AGENTS.md`: project behavior and activation.
- `.codex/config.toml`: supported parent settings and native multi-agent capacity.
- `.codex/agents/*.toml`: twelve native role definitions.
- `team/WORKFLOW.md`: dependency stages, ownership and completion gates.
- `team/HANDOFF.md`: task envelopes and accepted-result protocol.
- `team/TESTING.md`: setup smoke test and future product integration checks.
- `team/BUILD-PROMPT.md`: reusable prompt.
- `scripts/`: launcher and verifier.

The role instructions were authored for this request using official Codex mechanisms.
They are not OpenAI-published agent personas. No Claude Code configuration was
copied or installed. The third-party collection mentioned in the request was not
needed. Official Skills are optional extensions, not autonomous agent definitions.

Official references checked on 2026-10-29:

- [Codex subagents and custom-agent TOML](https://learn.chatgpt.com/docs/agent-configuration/subagents)
- [Project config, trust and precedence](https://learn.chatgpt.com/docs/config-file/config-basic)
- [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Official JSON configuration schema](https://learn.chatgpt.com/docs/config-schema.json)

The installed CLI's actual parser, native tool schema and invocation results are
used to confirm compatibility; current web documentation alone is insufficient.
