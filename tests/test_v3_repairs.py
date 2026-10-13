"""Regression tests for live v3 issues found in October 2026."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from ubique.v3.storage import Journal
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.world_model import WorldModel
from ubique.v3.representation import Cortex
from ubique.v3.semantics import SemanticLearner
from ubique.brain.substrate import CognitiveSubstrateManager


def test_large_legacy_journal_repacks_losslessly(tmp_path, monkeypatch):
    """The existing 20MB monthly journal must remain hash linked after repack."""
    import ubique.v3.storage as storage
    monkeypatch.setattr(storage, "ARCHIVE_LIMIT", 320)
    root = tmp_path / "v3"
    journal = Journal(root)
    for i in range(32):
        journal.append("episode", {"value": i, "text": "brain" * 30}, generation=i)
    before = journal.events(verify=True)
    path = root / "events" / "2026-10.jsonl"
    # Re-create the original monolithic monthly layout to mimic production.
    files = list((root / "events").glob("*.jsonl"))
    with path.open("w", encoding="utf-8") as target:
        for file in files:
            if file != path:
                target.write(file.read_text(encoding="utf-8"))
                file.unlink()
    reopened = Journal(root)
    after = reopened.events(verify=True)
    assert [e["hash"] for e in after] == [e["hash"] for e in before]
    assert not path.exists()
    assert len(list((root / "events").glob("*.jsonl.gz"))) > 1
    reopened.append("test", {"next": True}, generation=33)
    assert len(Journal(root).events(verify=True)) == 33


def test_bounded_segment_rotation_and_integrity(tmp_path, monkeypatch):
    import ubique.v3.storage as storage
    monkeypatch.setattr(storage, "SEGMENT_LIMIT", 380)
    journal = Journal(tmp_path / "v3")
    for i in range(25):
        journal.append("episode", {"text": "concept" * 30, "index": i})
    assert len(list((tmp_path/"v3"/"events").glob("*.jsonl.gz"))) >= 2
    assert len(Journal(tmp_path/"v3").events(verify=True)) == 25


def test_compressed_snapshot_replays_without_losing_edges(tmp_path):
    from ubique.v3.representation import Cortex
    cortex = Cortex()
    nodes = cortex.activate(["learning", "memory", "identity"])
    cortex.learn(nodes)
    journal = Journal(tmp_path / "v3")
    journal.snapshot(cortex.as_dict())
    assert journal.snapshot_path.suffix == ".gz"
    assert journal.snapshot_path.is_file()
    loaded = Cortex.from_dict(Journal(tmp_path/"v3").load_snapshot()["graph"])
    assert len(loaded.edges) == len(cortex.edges)


def test_legacy_mutation_is_not_mislabeled_corruption(tmp_path):
    journal = Journal(tmp_path/"v3")
    path = tmp_path / "legacy_library.json"
    path.write_text('{"books": 1}')
    journal.migration_record("library_index", path)
    path.write_text('{"books": 2}')
    assert journal.verify_migration()["library_index"]
    assert journal.migration_drift()["library_index"]


def test_episode_identity_survives_new_process(tmp_path):
    journal = Journal(tmp_path/"v3")
    hip = Hippocampus(journal, legacy_path=tmp_path/"none")
    source = hip.encode(
        "Conscious experience has prompted debate across biology and philosophy.",
        source="library:mind", epistemic_status="source_passage")
    restored = Hippocampus(Journal(tmp_path/"v3"), legacy_path=tmp_path/"none")
    new = restored.all_episodes()[0]
    assert new["id"] == source["id"]
    assert new["timestamp"] == source["timestamp"]


def test_independent_books_produce_sourced_tentative_insight(tmp_path):
    journal = Journal(tmp_path/"v3")
    hip = Hippocampus(journal, legacy_path=tmp_path/"none")
    world = WorldModel(journal)
    cortex = Cortex()
    learner = SemanticLearner(journal, hip, world, cortex)
    a = hip.encode(
        "The relation of memory and attention influences learning in human beings. "
        "We must distinguish learning from mere repetition.",
        source="library:book-one", epistemic_status="source_passage")
    assert learner.observe(a, generation=1)
    b = hip.encode(
        "Learning from memory requires attention, and our understanding of such "
        "connections should be revisable.",
        source="library:book-two", epistemic_status="source_passage")
    insights = learner.observe(b, generation=2)
    comparisons = [x for x in insights if x.get("kind")=="cross_source_comparison"]
    assert comparisons and comparisons[0]["uncertainty"] > 0.5
    assert set(comparisons[0]["evidence_ids"]) == {a["id"], b["id"]}
    assert any(x["status"] == "INFERRED" for x in world.claims.values())
    assert not world.stances  # comparisons alone are not fabricated opinions


def test_contributor_preserves_world_claim_evidence_identifiers():
    packet = CognitiveSubstrateManager.parse_packet(json.dumps({
        "world_model_updates":[{
            "statement":"Two authors offer distinct approaches to memory.",
            "confidence":0.42, "evidence_ids":["v3:abc","v3:def"]}],
        "stance_updates":[{
            "topic":"learning","position":"Tentatively favour comparing sources.",
            "confidence":0.4,"evidence_ids":["v3:abc"]}]}),
        "groq")
    assert packet["world_model_updates"][0]["evidence_ids"] == ["v3:abc","v3:def"]


def test_model_claims_do_not_require_unverified_self_stances(tmp_path):
    from ubique.v3.runtime import UbiqueV3
    from ubique.config import Config
    agent = UbiqueV3(Config.from_env(), storage_root=tmp_path/"v3",
                     enable_remote=False)
    actual = agent.hippocampus.encode(
        "This witnessed reading excerpt discusses memory and attention.",
        source="library:test", epistemic_status="source_passage")
    agent._packet_recall = [actual]
    packet={"provider":"test","associations":[],"questions":[],"hypotheses":[],
            "actions":[],"world_model_updates":[{
                "statement":"A tentative interpretation of the source.",
                "confidence":0.95,"evidence_ids":[actual["id"]]}],
            "stance_updates":[{"topic":"reading","position":"Compare independent evidence.",
                               "confidence":0.4,"relation":"new",
                               "evidence_ids":[actual["id"]]}]}
    agent._handle_model(packet, 3)
    assert len(agent.world.stances)==1
    assert next(iter(agent.world.claims.values()))["confidence"]<=0.45
    agent.world.stances.clear()
    packet["stance_updates"][0]["evidence_ids"]=["invented:episode"]
    agent._handle_model(packet, 4)
    assert not agent.world.stances


def test_old_uncompressed_snapshot_is_converted_without_offset_change(tmp_path):
    import ubique.v3.storage as storage
    from ubique.v3.representation import Cortex
    root=tmp_path/"v3"
    journal=Journal(root)
    graph=Cortex()
    graph.activate(["remember", "truth"])
    old={"schema_version":3,"generation":17,"events":3,"graph":graph.as_dict()}
    storage.write_atomic(root/"cortex_snapshot.json",old)
    journal.manifest["snapshot_events"]=3
    journal.manifest["snapshot_sha256"]=storage.digest(storage.canonical(old))
    storage.write_atomic(root/"manifest.json",journal.manifest)
    reopened=Journal(root)
    assert reopened.snapshot_path.exists()
    assert not reopened.legacy_snapshot_path.exists()
    assert reopened.manifest["snapshot_events"]==3
    assert reopened.load_snapshot()==old


def test_repeated_noop_actions_are_failures(tmp_path):
    from ubique.v3.actions import build_registry, Candidate
    cortex=Cortex()
    registry=build_registry()
    result=registry.execute(Candidate("connect","nothing",{"cues":["unknown"]}),
        {"cortex":cortex})
    assert not result.success


def test_multi_source_research_fallback(monkeypatch):
    import ubique.v3.research as research
    monkeypatch.setattr(research,"wikipedia",lambda topic:[])
    fake=[{"title":"Memory and cognition","text":"The mind learns from sources",
           "url":"https://openalex.org/W123","source":"OpenAlex",
           "attribution":"research metadata"}]
    monkeypatch.setattr(research,"openalex",lambda topic:fake)
    assert research.discover("mind")==fake
