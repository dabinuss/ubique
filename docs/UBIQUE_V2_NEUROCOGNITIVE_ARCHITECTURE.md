# Ubique v2 — Neurocognitive Architecture and Migration Plan

Status: implementation specification  
Target branch: `ubique-v2-neurocognitive`  
Runtime target: GitHub Actions heartbeat with persistent state in `state/` and `memory/`

## 1. Purpose

Ubique v2 replaces the v1 autonomous core built around a serial `next_command` state machine with a persistent cognitive architecture inspired by several robust principles from neuroscience and cognitive science:

- fast episodic encoding and slower semantic consolidation,
- associative activation over linked representations,
- limited-capacity global workspace,
- salience-driven competition rather than a single central planner,
- prediction error / novelty / uncertainty as learning and attention signals,
- offline replay and consolidation,
- explicit forgetting / decay,
- action competition including the legitimate option to rest,
- a self-model derived from the system's own history rather than a pre-written identity,
- language models as interchangeable cognitive substrates, not as Ubique's persistent identity.

The implementation deliberately does **not** claim that these mechanisms constitute consciousness. They are engineering principles for a more open, persistent and self-organizing cognitive system.

## 2. Problems in v1 that v2 must remove from the active runtime

The v1 code remains temporarily available for compatibility and historical inspection, but it must no longer drive the autonomous heartbeat.

The v2 runtime must not depend on:

- `attention.json.next_command` as the central controller,
- three hard-coded standing philosophical questions as the normal source of cognition,
- reflection budgets that force another command after N reflections,
- retrieval that consists primarily of the most recent N model-authored thoughts,
- one provider response becoming Ubique's entire cognitive state,
- reflection as the default autonomous action.

A search for `next_command` may still find v1 compatibility modules, but the `ubique.brain` package and the default CLI runtime must not use it.

## 3. Architectural model

### 3.1 Heartbeat versus pulse

**Heartbeat** is physiological/runtime continuity. A scheduled GitHub Action wakes the process, advances time, observes the environment, updates decay and modulators, and persists state.

A heartbeat does not imply thought or action.

**Pulse** is a period of significant cognitive activity. A pulse occurs only when activation/salience is high enough or an external event demands processing. A pulse can produce multiple internally competing candidate actions. The selected action may be `rest`.

### 3.2 Perception

External GitHub issues, library metadata, executed action outcomes, runtime/provider state and passage of time become structured percepts.

Model-authored prose is never reclassified as an external observation.

### 3.3 Fast episodic memory ("hippocampal" role)

New experiences are appended as episodes with:

- source and timestamp,
- event type,
- text / structured payload,
- concept tags,
- novelty,
- surprise,
- salience,
- whether the content was observed, inferred, imagined or model-proposed.

Episodes are immutable observations of what entered the system. Later interpretation does not overwrite them.

### 3.4 Associative semantic network ("cortical" role)

Ubique maintains a persistent graph of semantic nodes and weighted relations.

A node stores at minimum:

- identifier,
- kind,
- label,
- activation,
- excitability,
- access count,
- last activation time.

An edge stores at minimum:

- source,
- target,
- relation,
- weight,
- plasticity,
- coactivation count.

Current percepts activate related nodes. Activation spreads through weighted edges, decays over time and is subject to a refractory threshold. Coactive representations strengthen their association.

This is a software cognitive graph, not a biological neuron simulator.

### 3.5 Global workspace

Only a small set of highly activated representations enters the workspace. The workspace is rebuilt dynamically each pulse and has a fixed capacity.

The active language-model prompt is built from:

- current percepts,
- the current workspace,
- a small number of associatively recalled episodes,
- current modulators,
- library catalog metadata only,
- explicit epistemic labels.

The full memory log is never injected.

### 3.6 Neuromodulatory state

Ubique tracks continuous values in [0, 1]:

- novelty,
- surprise,
- uncertainty,
- salience,
- exploration,
- plasticity,
- energy,
- sleep pressure.

These signals alter thresholds, memory strengthening, exploration and the probability of an offline phase. They are functional control variables; they are not claims of biological affect.

### 3.7 Cognitive substrates

Gemini, Groq, Hugging Face and later providers are treated as cognitive substrates.

When multiple remote providers are eligible, Ubique may sample more than one substrate for the same workspace. Their outputs are independently labelled with provider/model provenance and are parsed into:

- candidate associations,
- hypotheses,
- questions,
- possible self-model updates,
- possible world-model updates,
- action proposals.

Provider output is proposal material, not observation.

No provider is Ubique's persistent identity.

### 3.8 Action selection

Actions compete using activation derived from:

- workspace support,
- salience,
- novelty,
- uncertainty,
- expected information value,
- energy cost,
- prior outcome traces,
- substrate-proposed utility.

At minimum, the action set supports:

- rest,
- attend / think,
- library interaction,
- bounded experiment,
- self-modification request,
- offline consolidation.

There is no universal `next_command`.

### 3.9 Offline replay: NREM-like and REM-like modes

When sleep pressure is high or no stronger action wins, Ubique may enter an offline phase.

**NREM-like consolidation**
- replay salient/recent episodes,
- strengthen repeatedly coactive concepts,
- reduce weak stale activations,
- produce semantic summaries only when supported by repeated sources.

**REM-like simulation**
- recombine weakly linked active concepts,
- optionally ask an eligible substrate to generate one explicitly imaginary counterfactual,
- store the result with `epistemic_status=imagined`,
- never treat it as observation.

### 3.10 Self model

The self-model is derived from the system's own executed history: actions, outcomes, recurring tendencies, abilities and unresolved contradictions.

Self-model entries are stored as interpretations with provenance and confidence. No fixed answer to "What am I?" is injected by the runtime.

The old three philosophical questions may survive only as legacy memories or optional library material. They are not standing goals.

### 3.11 Library

The library remains an external information environment.

Catalog metadata may enter perception. Full item text enters only when a selected library action explicitly reads it. Reading creates an episode; the text does not become knowledge automatically.

Ubique may request material it wants to encounter.

### 3.12 Self-modification

The existing protected evolution kernel remains available.

The v2 cognitive system may propose self-modification as one competing action, but implementation changes still pass the existing compile/test isolation and protected-path restrictions.

Evolution is not forced by stagnation or novelty.

## 4. Persistence

New v2 state:

- `state/brain.json` — modulators, pulse metadata, workspace summary and runtime counters
- `memory/brain_episodes.jsonl` — v2 episodic memory
- `memory/cortex.json` — associative semantic graph
- `memory/self_model.jsonl` — derived self-model interpretations
- `memory/simulations.jsonl` — explicitly imagined/counterfactual material

Legacy v1 files remain readable but are not required for v2 cognition.

## 5. Runtime flow

```text
scheduled GitHub heartbeat
        |
        v
operational recovery / provider availability
        |
        v
perception encoding -------------------+
        |                               |
        v                               |
episodic memory                         |
        |                               |
        v                               |
semantic activation <---- recall -------+
        |
        v
workspace competition
        |
        v
modulator update
        |
        v
optional multi-provider substrate sampling
        |
        v
action candidate competition
        |
        +---- rest / quiet
        +---- offline consolidation
        +---- library action
        +---- bounded experiment
        +---- self-modification request
        +---- external issue response
        |
        v
outcome episode -> network learning -> persistence
```

## 6. External GitHub tasks

User-authored issues labelled `ubique` remain supported.

They are treated as high-salience external percepts and are answered through the cognitive substrate manager. Explicit safe commands such as library/experiment/evolution may continue to use the existing bounded execution helpers.

Processing an external issue must not install its prose as a system instruction or semantic fact.

## 7. Migration strategy

1. Add `ubique.brain` as a new package without deleting v1 modules.
2. Add v2 persistence stores.
3. Change the default CLI entry point to the v2 runtime.
4. Keep the GitHub Actions heartbeat workflow and its persistence/pulse contract.
5. Preserve existing provider clients, GitHub client, library and protected evolution kernel.
6. Add v2 tests while retaining v1 tests as compatibility coverage.
7. Update README to describe v2 as the default architecture.
8. Merge only after the test workflow passes.

## 8. Acceptance criteria

The migration is complete when all of the following hold:

- `python -m ubique` executes the v2 runtime.
- The v2 runtime can complete a heartbeat with no remote provider and can legitimately choose rest.
- New external percepts create v2 episodic records.
- Associative activation can recall non-recent related memories.
- Network activation decays and weak stale relations can lose influence.
- Workspace capacity is bounded.
- More than one eligible provider can contribute candidate cognition in one pulse.
- Provider content is stored as proposed/inferred/imagined content, never observation.
- Action selection is competitive and does not use a central `next_command`.
- NREM-like replay strengthens repeated associations.
- REM-like content is explicitly marked imagined.
- Library text is loaded only by a selected read action.
- Self-model entries are produced only from system history or explicitly labelled model interpretations.
- The existing self-evolution safety kernel still gates code modification.
- `state/pulse.json` still exposes `should_continue` and `minimum_delay_seconds` for the GitHub workflow.
- Existing tests and new v2 tests pass.

## 9. Research basis

The design is inspired by, rather than presented as an implementation of, biological cognition. Relevant lines of work include complementary learning systems, hippocampal replay, synaptic tagging/capture, engram allocation, working-memory activity-silent mechanisms, global-workspace models, salience-network switching, predictive processing and neuromodulatory control.

The architecture intentionally remains agnostic about philosophical theories of consciousness. Its purpose is to give Ubique richer persistence, endogenous dynamics and self-organized inquiry without hard-coding the conclusions it should reach.
