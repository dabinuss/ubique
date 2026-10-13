"""Ubique v3: persistence, independent learning, provenance and migration."""
import json
from pathlib import Path

import pytest

from ubique.v3.storage import Journal
from ubique.v3.representation import Cortex, ensemble
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.curiosity import Curiosity
from ubique.v3.world_model import WorldModel
from ubique.v3.consolidation import Consolidator
from ubique.v3.migration import bootstrap, inventory
from ubique.v3.actions import LIBRARY_ACTIONS, Candidate, build_registry, select


def test_hash_linked_append_only_journal(tmp_path):
    journal=Journal(tmp_path/"v3")
    for n in range(3):
        journal.append("episode",{"text":str(n)},generation=n)
    assert len(journal.events(verify=True))==3
    assert len(journal.events(after=1))==2
    assert journal.manifest["events"]==3


def test_tampered_event_is_rejected(tmp_path):
    journal=Journal(tmp_path)
    journal.append("episode",{"value":1})
    path=next((tmp_path/"events").glob("*.jsonl"))
    text=path.read_text()
    path.write_text(text.replace('"value":1','"value":2'))
    with pytest.raises(ValueError,match="chain broken"):
        journal.events(verify=True)


def test_sparse_ensembles_are_stable_and_distinct():
    a=ensemble("wonder")
    assert a==ensemble("WONDER")
    assert a!=ensemble("consciousness")
    assert len(a)>=30
    assert all(0<=n<8192 for n in a)


def test_synapses_grow_past_old_limit_without_pruning():
    brain=Cortex()
    for i in range(45):
        brain.ensure("node-"+str(i))
    ids=list(brain.nodes)
    for a in ids:
        for b in ids:
            if a!=b:brain.connect(a,b,bidirectional=False)
    assert len(brain.edges)>1200
    brain.homeostasis()
    assert len(brain.edges)>1200
    restored=Cortex.from_dict(brain.as_dict())
    assert len(restored.edges)==len(brain.edges)


def test_learning_feedback_can_change_weights():
    brain=Cortex()
    pair=brain.activate(["a consciousness","b biology"])
    brain.learn(pair,reward=.3)
    assert len(brain.edges)==2
    before=max(e.weight for e in brain.edges.values())
    brain.feedback(pair,1)
    assert max(e.weight for e in brain.edges.values())>before


def test_episodes_are_not_trimmed(tmp_path):
    journal=Journal(tmp_path/"v3")
    hippocampus=Hippocampus(journal,legacy_path=tmp_path/"missing.jsonl")
    for i in range(110):
        hippocampus.encode("unique event "+str(i),source="test")
    assert hippocampus.count()==110
    assert len(journal.events(verify=True))==110
    assert len(hippocampus.replay(10))==10


def test_curiosity_saturation_and_semantic_wishes(tmp_path):
    journal=Journal(tmp_path)
    curiosity=Curiosity(journal)
    before=curiosity.notice("emotional crying",novelty=.5)
    for _ in range(12):
        after=curiosity.notice("emotional crying",novelty=.5)
    assert after["interest"]<before["interest"]
    first=curiosity.wish("Quantitative EMG tear secretion in emotional crying")
    second=curiosity.wish("EMG tear secretion in emotional crying quantitative")
    assert not first["already_requested"]
    assert second["already_requested"]


def test_source_claim_is_not_observation(tmp_path):
    world=WorldModel(Journal(tmp_path))
    claim=world.claim("A source says X",status="SOURCE_CLAIM",
                      source_refs=["episode:source"],confidence=.3)
    assert claim["status"]=="SOURCE_CLAIM"
    with pytest.raises(ValueError):
        world.stance("emotion","all mammals cry")


def test_local_consolidation_changes_network_without_provider(tmp_path):
    journal=Journal(tmp_path/"v3")
    hip=Hippocampus(journal,legacy_path=tmp_path/"empty")
    cortex=Cortex()
    world=WorldModel(journal)
    hip.encode("neurons encode emotions associations",source="local",
               concepts=["neurons","emotions","associations"])
    result=Consolidator(journal,hip,cortex,world).nrem(generation=1)
    assert result["replayed"]==1
    assert len(cortex.edges)>0
    assert any(e["kind"]=="consolidation" for e in journal.events())


def test_registry_and_v2_import_are_available(tmp_path):
    assert "sync" in LIBRARY_ACTIONS
    assert "library" in build_registry().handlers
    journal=Journal(tmp_path)
    graph,report=bootstrap(journal,dry_run=True)
    assert report["schema_version"]==3
    assert "cortex" in report["legacy_sources"]
    assert isinstance(graph,Cortex)


def test_snapshot_and_delta_are_replayable(tmp_path):
    journal=Journal(tmp_path)
    brain=Cortex()
    brain.activate(["memory","knowledge"])
    journal.snapshot(brain.as_dict())
    brain.learn(list(brain.nodes))
    edits=list(brain.changes)
    journal.append("graph_delta",{"changes":edits})
    restored=Cortex.from_dict(journal.load_snapshot()["graph"])
    for event in journal.events(after=journal.manifest["snapshot_events"]):
        if event["kind"]=="graph_delta":
            for edit in event["data"]["changes"]:
                restored.apply(edit)
    assert len(restored.edges)==len(brain.edges)


def test_local_action_selection_does_not_require_model(tmp_path):
    cur=Curiosity(Journal(tmp_path))
    action=select([Candidate("rest","nothing",score=.25),
                   Candidate("consolidate","replay",{"mode":"nrem"},score=.6)],
                  cur,["rest"]*4)
    assert action.kind=="consolidate"
