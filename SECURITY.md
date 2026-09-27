# Security model

Ubique processes untrusted GitHub issue content.

## Rules

- LLM text is never executed directly as shell. Autonomous evolution may generate Python source; candidate source is compiled and exercised by the test/benchmark gate before publication.
- Candidate test/benchmark subprocesses run with credential-shaped environment variables removed and checkout-persisted Git HTTP credentials temporarily suspended.
- Repository secrets are never added to prompts, candidate test environments or benchmark environments.
- Issues can request text tasks, not arbitrary code execution.
- Normal autonomous persistence is limited to `state/`, `memory/`, issue comments and label removal. Bounded evolution may additionally publish validated source/test changes to a dedicated draft-PR branch.
- Self-modification is restricted to allowlisted Python source/test files and always uses a draft branch/PR workflow with tests and benchmark gates.
- The evolution gate, workflow files, credential plumbing, state, memory and dependency metadata are protected from autonomous modification.
- Self-generated code is never auto-merged.

## Public repositories

Do not add secrets to source files. Use GitHub Actions secrets.

If a secret is accidentally committed, revoke it immediately and rotate it at the provider.
