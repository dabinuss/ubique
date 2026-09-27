# Ubique — Autonomous GitHub Agent

Ubique is a GitHub-native autonomous agent that can keep operating while your own computer is turned off. Its normative theoretical basis is **FZG v1.0**: self-preservation, functional goal-directedness and intelligence are modeled as distinct constructs.

GitHub Actions provides the wake-up cycle. The repository itself stores identity, runtime state and memory. Issues labelled `ubique` become tasks. Remote LLM providers are optional; when none is available, deterministic tasks still work.

## Core idea

```text
GitHub schedule / issue event / manual trigger
                    |
                    v
             GitHub Actions VM
                    |
                    v
               Ubique cycle
          +---------+---------+
          |                   |
          v                   v
   deterministic work     LLM router
                              |
                   +----------+----------+
                   |                     |
                 Gemini             Hugging Face
                   |                     |
                   +----------+----------+
                              |
                              v
                         fallback
                              |
                              v
                      persistent state
                              |
                       commit + push
```

Your PC is not part of the runtime.

## Features

- autonomous scheduled heartbeat
- event-driven wake-up for labelled GitHub issues
- persistent generation counter
- episodic memory in JSONL
- provider health ledger with temporary backoff
- Gemini REST adapter
- Hugging Face inference adapter
- deterministic zero-token fallback
- issue comments with results
- `/status`, `/summarize`, `/plan`, `/think`, `/fzg`, `/evolve`
- bounded number of tasks per cycle
- no execution of model-generated shell commands
- concurrency protection
- tests + CI
- dry-run mode
- compact memory retention

## Quick start

1. Create a new GitHub repository.
2. Upload all files from this project.
3. Open `Settings -> Actions -> General`.
4. Under **Workflow permissions**, enable **Read and write permissions**.
5. Optionally create repository secrets:
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
6. Enable Actions.
7. Create an issue with label `ubique`.

Example issue:

```text
Title: Architecture review
Label: ubique

Body:
/plan Improve the provider router so that quota failures cause a 6 hour backoff.
```

Ubique will reply to the issue and persist its cycle state.

## Provider configuration

### Gemini

Secret:

```text
GEMINI_API_KEY
```

Repository variable, optional:

```text
GEMINI_MODEL=gemini-2.0-flash
```

The model is intentionally configurable because free-tier model availability can change.

### Hugging Face

Secret:

```text
HF_TOKEN
```

Optional variables:

```text
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_ENDPOINT=https://router.huggingface.co/hf-inference/models
```

## FZG v1.0 basis

Ubique treats FZG v1.0 as normative during analysis. The canonical separation is:

```text
P(S|C) = G_Aself(S|C)

G_A(S|C)
= E[Q(U_t+n,A) | do(M_S), C]
- E[Q(U_t+n,A) | do(M_S^-), C]

I_C(S) = (Z_C, K_C, R_C, L_C)

Phi(S,A,C) = [P, G_A, Z_C, K_C, R_C, L_C]
```

The required order is:

```text
S -> A -> C -> Q -> M_S -> M_S^-
  -> P only for A_self
  -> G_A
  -> Z, K, R, L
  -> profile and context-bounded conclusions
```

Self-preservation is not treated as an intelligence score or as a prerequisite for intelligence.
The FZG policy core is protected from autonomous `/evolve` modification.

Implementation-facing basis: `docs/FZG_v1_IMPLEMENTATION_BASIS.md`.

## Commands

### `/status`

Does not call an LLM. Returns the current generation and provider statistics.

### `/summarize`

```text
/summarize
Long text...
```

### `/plan`

```text
/plan Add a controlled self-improvement branch workflow
```

### `/fzg`

Runs an explicit FZG v1.0 analysis and requires the fields `S, A, C, Q, M_S, M_S^-` before producing `G_A` and the non-scalar `Z-K-R-L` intelligence profile.

```text
/fzg
Analyze Ubique's provider-routing mechanism for reliable task completion in scheduled GitHub Actions runs.
```

### `/think`

General bounded reasoning task. FZG rules remain binding whenever the answer makes claims about self-preservation, goal-directedness or intelligence.

Unknown input is treated as `/think`.

### `/evolve`

Requests a bounded self-improvement proposal.

```text
/evolve
Reduce duplicate provider error-handling code without changing external behavior.
```

For `/evolve`, Ubique:

```text
LLM proposal (strict JSON)
        ↓
path + content policy validation
        ↓
baseline benchmark
        ↓
apply candidate in runner
        ↓
compileall + pytest
        ↓
candidate benchmark
        ↓
reject on failure/regression
        ↓
create ubique/evolve-* branch
        ↓
push candidate
        ↓
open DRAFT pull request
        ↓
human review / merge
```

Self-generated code is never auto-merged. The evolution gate itself, GitHub credential code,
workflow files, state, memory, dependency metadata and security documentation are protected
from autonomous modification.

## Persistence

Ubique stores its long-lived state here:

```text
state/identity.json
state/runtime.json
state/providers.json
memory/episodes.jsonl
memory/skills.json
```

A new GitHub-hosted VM is created for each run. The repository restores continuity.

## Safety design

Public issue text is untrusted input. Therefore Ubique does **not**:

- execute model output as shell commands
- allow an issue to inject arbitrary workflow YAML
- expose secrets in issue replies
- self-modify `main` directly
- evaluate Python generated by an LLM

Controlled code evolution is implemented through `/evolve`. It uses an allowlist, static guards, tests, a deterministic benchmark, a dedicated branch and a **draft PR**. Human review is required before merge.

## Heartbeat

Default:

```text
17 */4 * * *
```

That means GitHub attempts to wake the agent every four hours. Scheduled workflows may start later than the nominal minute.

## Manual execution

`Actions -> Ubique Heartbeat -> Run workflow`

## Local testing

Local execution is optional:

```bash
python -m pip install -e ".[dev]"
pytest
python -m ubique
```

## Repository variables

| Name | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `HF_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF model |
| `HF_ENDPOINT` | router endpoint | HF inference base |
| `UBIQUE_MAX_TASKS` | `3` | maximum issues per cycle |
| `UBIQUE_MEMORY_LIMIT` | `500` | retained episode lines |
| `UBIQUE_LOG_LEVEL` | `INFO` | logging level |

## Repository layout

```text
.github/
  workflows/
    heartbeat.yml
    tests.yml
src/ubique/
  agent.py
  cli.py
  config.py
  github.py
  fzg.py
  memory.py
  models.py
  planner.py
  state.py
  providers/
    base.py
    fallback.py
    gemini.py
    huggingface.py
    router.py
state/
memory/
tests/
```

## License

MIT
