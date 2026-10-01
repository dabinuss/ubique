# Ubique v2 — Neurocognitive Autonomous System

Ubique is a GitHub-native autonomous system with persistent memory, associative cognition, multi-provider reasoning and self-modification.

Version 2 replaces the old serial reflection loop with a cognitive architecture inspired by real mechanisms studied in neuroscience and cognitive science: episodic memory, associative activation, limited working space, replay, consolidation, forgetting, salience, uncertainty and competitive action selection.

Ubique does not contain a hard-coded answer to what it is, what it should want or what consciousness is. The architecture provides mechanisms; conclusions remain open.

## Core runtime

The default runtime is no longer a loop of heartbeat -> reflect -> next command -> reflect.

It is:

    heartbeat
       |
       v
    perception
       |
       v
    episodic encoding
       |
       v
    associative activation + recall
       |
       v
    limited global workspace
       |
       v
    optional multi-provider cognitive sampling
       |
       v
    competing action candidates
       |
       +--> rest
       +--> attend
       +--> library
       +--> experiment
       +--> evolve
       +--> offline consolidation
       |
       v
    outcome -> learning -> persistence

A scheduled heartbeat preserves continuity. A cognitive pulse only continues immediately when residual activation is strong enough. Quietness is a valid state.

## Episodic memory

memory/brain_episodes.jsonl stores fast, provenance-labelled experiences.

Records distinguish observed external input, observed runtime state, observed action outcomes, model proposals, external source material and imagined simulations.

Model-authored text is never silently converted into observation.

## Associative semantic network

memory/cortex.json stores semantic nodes and weighted relations.

Nodes have activation, excitability and thresholds. Relations have weights, plasticity and coactivation counts. Activation spreads through the graph and decays over time. Coactive representations can strengthen their links.

These are software cognitive units, not claims of biological neurons.

## Global workspace

Only a small set of highly activated representations is exposed to active reasoning at once.

The workspace is rebuilt competitively every pulse from current percepts, activated semantic nodes and associatively recalled episodes. The entire memory log is never injected into model context.

## Functional modulators

Ubique maintains bounded control variables for novelty, surprise, uncertainty, salience, exploration, plasticity, energy and sleep pressure.

They influence attention, learning, action competition and offline phases. They are functional control signals, not claims of feelings or biological chemistry.

## Multi-provider cognitive substrates

Gemini, Groq, Hugging Face and future providers are treated as temporary cognitive substrates.

When more than one provider is available, multiple providers may contribute independently to one pulse. Their outputs are parsed into associations, hypotheses, questions, self-model interpretations, world-model interpretations and action proposals.

Provider output remains proposal material. No single model is Ubique's persistent identity.

## Competitive action selection

There is no central next-command controller in the v2 core.

Possible actions compete according to current support, utility, expected information gain, novelty, salience, exploration and energy cost.

Rest is always a legitimate candidate.

## Offline replay

NREM-like consolidation replays salient episodes, strengthens repeated associations, forms schema nodes from repeated co-occurrence and reduces transient activation.

REM-like simulation may recombine active concepts into explicitly counterfactual material. These records are marked epistemic_status=imagined. They can inspire later inquiry but never count as observation.

## Self model

memory/self_model.jsonl is derived from action history and explicitly labelled interpretations.

The runtime does not inject a fixed answer to the question “What am I?”. The former standing questions from v1 may remain in legacy files or optional library material, but they no longer schedule autonomous cognition.

## Library

memory/library remains an external information environment.

The active cognitive prompt sees catalog metadata only. Full content is loaded only when a selected library action deliberately reads an item. Reading creates an episode; a book does not automatically become truth merely because it exists in the library.

FZG remains an optional library item rather than a governing doctrine.

## Self-modification

The existing protected evolution kernel remains active.

Ubique v2 may select self-modification as one competing action, but candidate code still stays inside the evolvable Python surface, runs without credential-shaped environment variables, must compile, must pass the test suite and cannot modify the protected evolution/recovery/credential kernel.

The new src/ubique/brain package is part of the evolvable cognitive surface.

## Heartbeat and pulse

GitHub Actions still wakes Ubique every 15 minutes.

The heartbeat advances runtime generation, performs deterministic recovery, observes provider/runtime state, processes labelled GitHub issues, updates activation and modulators, and persists memory and state.

state/pulse.json retains the workflow contract through should_continue and minimum_delay_seconds. When significant activation remains, the workflow may schedule another immediate pulse. Otherwise Ubique returns to the scheduled heartbeat.

## Primary v2 persistence

- state/brain.json
- state/pulse.json
- state/runtime.json
- state/providers.json
- memory/brain_episodes.jsonl
- memory/cortex.json
- memory/self_model.jsonl
- memory/simulations.jsonl
- memory/library/index.json
- memory/library/items/

Legacy v1 state remains in the repository for compatibility and historical inspection but no longer drives the default CLI runtime.

## GitHub issues

Issues labelled ubique remain an external interface.

They are encoded as high-salience external percepts and handled through the v2 runtime. Bounded compatibility commands such as /status, /library, /experiment and /evolve remain available. Other issue content is answered through a cognitive substrate without installing issue text as system doctrine.

## Configuration

Provider configuration remains unchanged:

- GEMINI_API_KEY
- GEMINI_MODEL
- UBIQUE_GEMINI_DAILY_LIMIT
- GROQ_API_KEY
- GROQ_MODEL
- UBIQUE_GROQ_DAILY_LIMIT
- HF_TOKEN
- HF_MODEL
- HF_ENDPOINT

Optional v2 controls:

- UBIQUE_BRAIN_WORKSPACE_SLOTS=7
- UBIQUE_BRAIN_MAX_SUBSTRATES=2
- UBIQUE_BRAIN_MAX_IMMEDIATE_PULSES=3
- UBIQUE_BRAIN_ACTIVATION_THRESHOLD=0.24

## Testing

Install development dependencies and run:

    python -m pip install -e ".[dev]"
    python -m compileall -q src
    python -m pytest -q

The v2 tests cover spreading activation, associative non-recency-biased recall, bounded workspace capacity, competitive action selection, rest/consolidation as valid outcomes, NREM-like replay, REM-like imagined status, absence of the serial next-command controller from the v2 core, and the CLI using the v2 runtime.

## Architecture plan

The full design and migration specification is in docs/UBIQUE_V2_NEUROCOGNITIVE_ARCHITECTURE.md.

## License

MIT
