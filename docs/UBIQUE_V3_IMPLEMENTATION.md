# Ubique 3.0 — Implementation and operations

**Date:** 2026-10-13. **Basis:** the 21-chapter Ubique 3 technical architecture concept.

## Components

| Module | Responsibility |
|---|---|
| v3/storage.py | Hash-linked append-only event journals, snapshots, SHA-256 migration references |
| v3/migration.py | Non-destructive v2 to v3 importer and continuity checks |
| v3/representation.py | Sparse ensembles, recurrent spreading, adaptive synapses and reversible archiving |
| v3/hippocampus.py | Episodic encoding, legacy retrieval, diverse replay without trimming |
| v3/world_model.py | Source claims, hypotheses, predictions and evidence-bound stances |
| v3/curiosity.py | Interests, novelty, saturation and related research requests |
| v3/workspace.py | Limited activation-based global workspace |
| v3/planner.py | Procedural action preference learned from real successes and failures |
| v3/consolidation.py | NREM/REM-inspired replay and explicitly simulated analogies |
| v3/actions.py | Typed executable local actions, library integration, experiments |
| v3/library.py | Read availability and semantic reading-request deduplication |
| v3/research.py | Attributed general-public-knowledge retrieval |
| v3/providers.py | Optional Gemini, Groq and HF LLM contributions |
| v3/self_model.py | V2 stances, procedural skills and autobiographical outcomes |
| v3/evolution.py | Provenance and audit events for autonomous code evolution |
| v3/runtime.py | Complete heartbeat integration and delta persistence |

## Cognitive operating loop

Per pulse: load v3 snapshot and events; recall episodes; infer associations from sparse patterns; activate and compete within a limited workspace; optionally ask a language provider; rank read/explore/connect/consolidate/research/evolution choices; execute one actionable choice; record its consequences; modulate eligible synapses; persist incremental graph deltas and outcome events.

LLMs do not own persistent identity or memory. With no remote providers, local association, source reading, memory consolidation, interest formation and synaptic learning continue.

## Synaptic dynamics

Neuron ensembles are deterministic sparse unit patterns. Weighted directed synapses support plasticity, stability, eligibility traces and late feedback. No global 1200-edge deletion cap exists in v3: weak unused synapses are recoverably archived, not erased. Typed graph relationships, hypothesized analogies and evidence-supported world-model claims remain distinct.

## Storage, migration, and recovery

V3 stores incremental events in memory/v3/events/YYYY-MM.jsonl, a hash-bearing manifest and periodic cortex_snapshot.json. The manifest references the actual remaining v2 files by SHA-256. V2 memories, books and self-model files are never removed. Current v2 graph nodes and synapses are imported on first v3 boot. The append-only event journal supports hash-chain integrity verification and replay from snapshots.

Historical memories already truncated by v2 cannot be invented during migration. Older source snapshots in Git history may be recovered separately.

## Runtime control

Production heartbeat reads UBIQUE_RUNTIME_VERSION from the repository variable, defaulting to 3. Set the variable to 2 to switch back to the existing v2 runtime without deleting v3 data. Regular GitHub Actions heartbeats are discrete compute windows, not continuous GPU or CPU execution. Existing GitHub issue actions temporarily delegate to the proven v2 issue handler.

## Reading and researching

Reading must have real source text. An unfulfilled book request is not a reading event. Original excerpts are recorded as source passages and are not automatically treated as established truth. Explicit citations, sources and license metadata accompany external public texts. Curious topic exploration also supports sourced encyclopedia summaries at a bounded rate.

## Verification

The v3 tests verify event integrity/tamper detection, more than 1200 edges, sparse ensemble stability, later feedback, non-trimmed memory, curiosity saturation, semantic wish matching, source vs opinion distinction, provider-free consolidation, migration, journal replay and a full offline heartbeat cycle.

## Research extensions

Further research options include adaptive embedding representations, more precise local language understanding, spiking neural sub-networks, graph-native discrepancy detection, richer autonomously discovered sources, scalable indexes, and native v3 issue handling. Those extensions do not require replacing the autobiographical store.

**Scientific inspirations:** Complementary Learning Systems (Kumaran et al. 2016); synaptic plasticity (Feldman 2009); eligibility traces (Izhikevich 2007); memory consolidation (Klinzing et al. 2019); curiosity as learning progress (Oudeyer et al. 2016). Engineering mechanisms are approximations, not exact biological neuron simulation.
