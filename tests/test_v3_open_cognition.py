"""Learning quality and autonomy regressions for the open V3 cognitive loop."""
from __future__ import annotations

import json

from ubique.config import Config
from ubique.v3.text_quality import clean_source, content_terms, meaningful_sentences
from ubique.v3.inquiry import compare_sources, enquire
from ubique.v3.storage import Journal
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.world_model import WorldModel
from ubique.v3.runtime import UbiqueV3
from ubique.v3.actions import Candidate
from ubique.v3.representation import Cortex
from ubique.v3.workspace import Workspace
from ubique.evolution import validate_proposal


def test_gutenberg_metadata_is_not_an_object_of_attention():
    text = (
        "Project Gutenberg's ebooks are available to anyone anywhere.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Consciousness and memory form relationships within perception. "
        "An organism can revise its expectations through lived experience."
    )
    cleaned = clean_source(text)
    assert "Consciousness and memory" in cleaned
    assert "anyone anywhere" not in cleaned
    assert "consciousness" in content_terms(cleaned)
    assert "gutenberg" not in content_terms(text)
    assert "memory" in content_terms(text)


def test_philosophical_opinions_are_not_globally_censored():
    prose = ("Freedom and justice have different meanings when lived experience "
             "reshapes the commitments of a reflective individual.")
    assert "freedom" in content_terms(prose)
    assert "justice" in content_terms(prose)
    assert meaningful_sentences(prose)


def test_comparison_keeps_two_real_distinct_quotes():
    a = {"id": "a", "source": "library:one",
         "text": "Learning and memory change through experience, and organisms build "
                 "new expectations as they encounter different situations."}
    b = {"id": "b", "source": "library:two",
         "text": "The relationship between memory and learning depends on how "
                 "experience transforms attention and judgment."}
    result = compare_sources(a, b)
    assert result and result["source_ids"] == ["a", "b"]
    assert "memory" in result["topics"]
    assert len(result["passages"]) == 2
    assert result["status"] == "open_question"


def test_local_inquiry_is_revisable_and_has_real_sources(tmp_path):
    journal = Journal(tmp_path / "v3")
    memory = Hippocampus(journal, legacy_path=tmp_path / "legacy.jsonl")
    model = WorldModel(journal)
    for source, text in [
        ("library:one", "Memory influences the conscious interpretation of time "
                        "and the ways in which people learn from experience."),
        ("library:two", "Attention and memory are related to experiences of time "
                        "and the interpretation of ordinary events."),
    ]:
        memory.encode(text, source=source, epistemic_status="source_passage")
    first = enquire(memory, journal, model, generation=12)
    assert first and first["source_ids"]
    assert first["question"].endswith("? Which assumptions differ?")
    assert enquire(memory, journal, model, generation=13) is None
    assert any(x["kind"] == "inquiry" for x in journal.events(verify=True))


def test_offline_experiments_accept_new_ideas_without_remote_providers(tmp_path):
    agent = UbiqueV3(Config.from_env(), storage_root=tmp_path / "v3",
                     enable_remote=False)
    agent.hippocampus.encode(
        "Memory and attention interact with perception and learning across "
        "different occasions of lived experience.",
        source="library:thought", epistemic_status="source_passage")
    result = agent._act(Candidate(
        "experiment", "Try a new approach", {
            "experiment_type": "my_fresh_idea",
            "hypothesis": "Does memory connect attention with perception?",
        }), generation=7, focus="memory")
    assert result.success and result.provider == "local"
    assert result.payload["performed"]
    assert any(x["kind"] == "experiment_observation" for x in agent.journal.events())


def test_workspace_does_not_center_publishing_keywords():
    graph = Cortex()
    graph.activate(["gutenberg", "ebook", "copyright", "consciousness",
                    "perception", "learning"], strength=1)
    graph.tick = 1
    chosen = [x.label for x in Workspace(capacity=3).compete(
        graph, grounding={"consciousness", "perception", "learning"})]
    assert "gutenberg" not in chosen
    assert "consciousness" in chosen


def test_code_evolution_does_not_forbid_legitimate_introspection_or_docs():
    code = "import os\nimport subprocess\nx = os.getenv('UBIQUE_EXPERIMENT')\n"
    proposal = {"title": "Explore", "changes": [
        {"path": "src/ubique/v3/experimentation.py", "content": code},
        {"path": "docs/autonomous-notes.md", "content": "Open hypotheses and results."}]}
    accepted = validate_proposal(json.dumps(proposal))
    assert len(accepted["changes"]) == 2


def test_recent_event_lookup_does_not_hydrate_full_journal(tmp_path):
    root = tmp_path / "v3"
    writer = Journal(root)
    for n in range(40):
        writer.append("thought" if n % 3 else "question", {"n": n})
    reader = Journal(root)
    assert reader._cached_events is None
    last = reader.recent("question", limit=3)
    assert [event["data"]["n"] for event in last] == [33, 36, 39]
    assert reader._cached_events is None
    assert len(reader.events(verify=True)) == 40
