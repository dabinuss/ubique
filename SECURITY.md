# Security model

Ubique processes untrusted GitHub issue content.

## Rules

- LLM output is never executed.
- Shell commands are static workflow code only.
- Repository secrets are never added to prompts.
- Issues can request text tasks, not arbitrary code execution.
- Autonomous writes are limited to `state/`, `memory/`, issue comments and label removal.
- Self-modification is restricted to allowlisted Python source/test files and always uses a draft branch/PR workflow with tests and benchmark gates.
- The evolution gate, workflow files, credential plumbing, state, memory and dependency metadata are protected from autonomous modification.
- Self-generated code is never auto-merged.

## Public repositories

Do not add secrets to source files. Use GitHub Actions secrets.

If a secret is accidentally committed, revoke it immediately and rotate it at the provider.
