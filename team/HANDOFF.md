# Task and handoff protocol

The orchestrator assigns one bounded task to a native agent type. Supply context
explicitly; do not assume an agent saw another agent's conversation.

```text
TASK ID: unique stage-role-number
ROLE: exact native name
GOAL: one concrete result
INPUTS: user constraints, relevant paths, agreed interface versions
DEPENDENCIES: completed task IDs and accepted contracts
OWNED FILES: explicit files or disjoint directories; [] for analysis
DO NOT EDIT: shared manifests, lockfiles, contracts, and other owners' paths
ACCEPTANCE: observable behavior or required analysis
VALIDATION: commands or evidence expected
BUDGET: bounded scope and retry/time limit appropriate to the task
RETURN: use the report below; no nested delegation
```

The child reports:

```text
TASK ID / ROLE:
STATUS: complete | blocked | needs_integration
SUMMARY:
FILES CHANGED: paths, or none
CONTRACTS: provided/consumed interfaces and compatibility concerns
VALIDATION: each command, exit code, relevant result; label anything not run
FINDINGS: severity, evidence and reproduction, if applicable
RISKS / BLOCKERS:
NEXT OWNER / NEXT ACTION:
```

The parent checks the files and evidence, accepts or rejects the handoff, and records
the result in `team/runs/<run-id>/plan.md`. A completed child is not automatically an
accepted task. Resume the same child for a focused fix when the runtime supports it;
otherwise close it and spawn a replacement with the same task context.

Failure policy: inspect the error before retrying. Allow at most two focused repair
attempts for the same failure without revising the plan. If the root cause remains,
record a blocker and choose a simpler implementation or request the missing input.
Never retry a destructive, billable or publishing action merely because its outcome
is uncertain; first verify the external state.
