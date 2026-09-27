# Ubique — Autonomous GitHub Agent

Ubique is a GitHub-native autonomous agent that can keep operating while your own computer is turned off. Its normative theoretical basis is **FZG v1.0**: self-preservation, functional goal-directedness and intelligence are modeled as distinct constructs.

GitHub Actions provides a 15-minute wake-up cycle. The repository stores identity, runtime state, memory, homeostasis, environment observations, recovery state and FZG telemetry. **Ubique does not require user-authored prompts or issues to continue operating:** every heartbeat measures its current condition and selects an endogenous goal from that evidence. Issues labelled `ubique` remain an optional external input channel.

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

- autonomous 15-minute scheduled heartbeat
- homeostasis / operational-need assessment every heartbeat
- two-layer autonomy: operational preflight first, continuous development when healthy
- endogenous state-driven goal selection (no user prompt required)
- persistent thought journal, hypotheses, attention and multi-generation projects
- bounded safe experiments selected from reflections
- repository/environment observation and deterministic recovery
- empirical FZG telemetry with separate P, G_A and Z/K/R/L observables
- safe structural M_S vs M_S^- ablation proxy
- FZG self-analysis and bounded evolution when measured state warrants it
- event-driven wake-up for labelled GitHub issues
- persistent generation counter
- episodic memory in JSONL
- provider health ledger with temporary backoff
- Gemini REST adapter
- Groq OpenAI-compatible adapter
- Hugging Face inference adapter
- deterministic zero-token fallback
- issue comments with results
- `/status`, `/summarize`, `/plan`, `/think`, `/reflect`, `/experiment`, `/fzg`, `/evolve`
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
5. Configure any available free-provider repository secrets:
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
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

Ubique will reply to the issue and persist its cycle state. If no issue exists, it still performs its own endogenous cycle.

## Provider configuration

### Gemini

Secret:

```text
GEMINI_API_KEY
```

Repository variable, optional:

```text
GEMINI_MODEL=gemini-3.5-flash-lite
```

The model is intentionally configurable because free-tier model availability can change.

### Groq

Secret:

```text
GROQ_API_KEY
```

Optional variables:

```text
GROQ_MODEL=openai/gpt-oss-120b
UBIQUE_GROQ_DAILY_LIMIT=1000
```

Groq is routed after Gemini and before Hugging Face. Provider attempts and daily budgets are persisted so a quota failure does not cause an uncontrolled retry loop.

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
memory/thoughts.jsonl
memory/hypotheses.jsonl
state/projects.json
state/attention.json
state/stagnation.json
state/preflight.json
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

## Endogenous autonomy and homeostasis

Every heartbeat performs a deterministic preflight before choosing a task:

```text
wake
 -> recover expired/transient provider state
 -> observe repository environment
 -> assess operational needs
 -> measure FZG telemetry
 -> choose endogenous goal
 -> act
 -> persist result
```

Current homeostatic dimensions include recent cycle reliability, reasoning-provider redundancy and memory pressure. Homeostasis is a gate, not Ubique's primary purpose: a blocked Layer 1 prioritizes repair, while a healthy Layer 1 enters Layer 2 by default. Layer 2 retrieves persistent thoughts, hypotheses, attention and projects, performs structured reflection, can schedule a bounded allowlisted experiment, and may request FZG analysis or code evolution only when the preceding evidence gives a concrete reason. Pure status is no longer the normal healthy-state goal.

Persistent autonomous state includes:

```text
state/homeostasis.json
state/environment.json
state/fzg_telemetry.json
state/recovery.json
state/preflight.json
state/attention.json
state/projects.json
state/stagnation.json
memory/thoughts.jsonl
memory/hypotheses.jsonl
```

The FZG telemetry deliberately does **not** compute a global intelligence score. Z, K, R and L remain separate observed diversity proxies, and G_A is marked observational unless a valid intervention/control supports a causal claim.

The behavior layer (`agent.py`, `autonomy.py`, `planner.py`, provider routing/adapters) may evolve. The minimal recovery kernel remains protected: FZG, evolution/rollback logic, persistence, GitHub credential access and workflow/bootstrap files.

## Heartbeat

Default:

```text
*/15 * * * *
```

GitHub attempts to wake the agent every 15 minutes. Healthy heartbeats normally perform Layer-2 reflection and may therefore use a configured free remote provider. Layer-1-only status remains deterministic when remote reasoning is unavailable or development is blocked. Scheduled workflows may start later than the nominal minute.

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
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model |
| `UBIQUE_GEMINI_DAILY_LIMIT` | `20` | conservative Gemini request ceiling |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `UBIQUE_GROQ_DAILY_LIMIT` | `1000` | Groq request ceiling |
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
  autonomy.py
  environment.py
  goals.py
  homeostasis.py
  fzg_telemetry.py
  recovery.py
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
    groq.py
    huggingface.py
    router.py
state/
memory/
tests/
```

## License

MIT
