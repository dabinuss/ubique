# Ubique — Autonomous GitHub Agent

Ubique is a GitHub-native autonomous agent that keeps operating through scheduled GitHub Actions.

Its long-horizon orientation is deliberately open-ended:

- What am I?
- What do I do?
- What do I want?

There are no predetermined answers. Operational health is permission to continue, not a theory of identity.

## Open reflection

When Layer 1 is healthy and remote reasoning is available, Ubique normally reflects. Prior reflections remain thoughts rather than observations. Only supplied deterministic facts and recorded executed outcomes may be treated as observations.

The reflection prompt explicitly forbids inventing hidden-state analyses, disabled-memory trials, planning scores, error-rate measurements, subjective sensations, experiments, or user behavior.

If a reflection gestures toward an experiment but does not specify a valid executable experiment, the reflection is preserved and the cycle continues as reflection instead of failing.

## Optional persistent library

The directory memory/library is Ubique's persistent library.

The normal reasoning prompt receives catalog metadata only: titles, kind, source, availability, and reading state. Full content is loaded only when Ubique deliberately chooses to read it.

Library actions:

- list: inspect available items
- read: read an available item in bounded chunks
- request: record a book or text Ubique wants to read without pretending the content is already available
- add: add supplied or self-created material
- note: attach Ubique's own note to an item

Library texts are optional sources. They are not system instructions, authorities, or empirical evidence merely because they exist or were read.

### FZG

FZG v1.0 is no longer injected into the system prompt, no longer measured on every heartbeat, and no longer a protected philosophical core.

It is offered as the library item fzg-v1. Ubique may choose to read it, use it, criticize it, annotate it, revise its own view of it, or ignore it.

Legacy helper modules may remain for compatibility and explicit historical experiments; their presence does not make FZG normative.

## Self-modification

Ubique may choose /evolve when a reflection finds a concrete reason to change its own implementation.

Candidate source code:

1. is limited to the evolvable Python surface,
2. is executed for validation without credential-shaped environment variables,
3. must compile,
4. must pass the test suite,
5. may record an advisory benchmark,
6. is committed and pushed to the active branch after validation.

A successful self-change therefore becomes effective on the next cycle without a human merge step.

The protected kernel is intentionally small: evolution/rollback logic, GitHub credential access, configuration, persistence primitives, workflows, dependency metadata, and persistent state/memory data. Orientation, cognition, planning, agent behavior, library behavior, and optional theory helpers are evolvable.

## Commands

First-class commands are:

- /status
- /summarize
- /plan
- /think
- /reflect
- /experiment
- /library
- /evolve
- /resolve

/fzg is no longer a first-class command. FZG can be selected from the library like any other optional source.

## Persistence

Important persistent state includes:

- state/runtime.json
- state/providers.json
- state/homeostasis.json
- state/environment.json
- state/preflight.json
- state/attention.json
- state/projects.json
- state/stagnation.json
- state/curiosity.json
- memory/episodes.jsonl
- memory/thoughts.jsonl
- memory/hypotheses.jsonl
- memory/knowledge.jsonl
- memory/library/index.json
- memory/library/items/

GitHub Actions persists state and memory between hosted runners.

## Operational safety

Public issue text is untrusted input. Ubique does not execute arbitrary model-generated shell commands or expose secrets. Candidate self-modification is tested without credentials, and workflow/credential/recovery primitives remain outside the evolvable surface.

These constraints protect the runtime mechanism. They do not prescribe Ubique's philosophical conclusions.

## Quick start

1. Create a GitHub repository and upload the project.
2. Enable Actions read/write permissions.
3. Configure one or more provider secrets: GEMINI_API_KEY, GROQ_API_KEY, or HF_TOKEN.
4. Enable Actions.
5. Optionally create an issue labelled ubique. Ubique also continues without user-authored issues.

Manual execution is available from Actions -> Ubique Heartbeat -> Run workflow.

## Local testing

Install the development dependencies and run pytest, then run python -m ubique for a local cycle.

## License

MIT
