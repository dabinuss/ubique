"""Regression tests for autonomous library, inquiry and persistent memory repairs."""
import json
from urllib.parse import parse_qs, urlsplit

from ubique.v3 import book_discovery
from ubique.v3.storage import Journal, SEGMENT_LIMIT
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.world_model import WorldModel
from ubique.v3.representation import Cortex
from ubique.v3.semantics import SemanticLearner
from ubique.v3.experimentation import run_local_experiment


class CatalogueResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return json.dumps(self.data).encode("utf-8")


def test_discovery_broadens_precise_research_wish_without_forging_a_book(tmp_path, monkeypatch):
    import ubique.library as library
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(library, "ITEMS_DIR", tmp_path / "items")
    queries = []

    def fake(request, timeout=12):
        search = parse_qs(urlsplit(request.full_url).query)["search"][0]
        queries.append(search)
        if search != "physiology":
            return CatalogueResponse({"results": []})
        return CatalogueResponse({"results": [{
            "id": 731, "title": "Introduction to Physiology", "copyright": False,
            "subjects": ["Human physiology and nervous systems"],
            "formats": {"text/plain": "https://www.gutenberg.org/ebooks/731.txt.utf-8"}
        }]})

    monkeypatch.setattr(book_discovery, "urlopen", fake)
    result = book_discovery.discover_public_books("Autonomic neural crying")
    assert len(result["added"]) == 1
    assert "physiology" in queries
    assert result["added"][0]["title"] == "Introduction to Physiology"
    assert library.library_catalog()[0]["read_count"] == 0
    assert book_discovery.discover_public_books("Autonomic neural crying")["added"] == []


def test_discovery_keeps_unrelated_books_out(tmp_path, monkeypatch):
    import ubique.library as library
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(book_discovery, "urlopen", lambda *a, **k: CatalogueResponse({
        "results": [{"id": 9, "title": "Deep Sea Volcanoes",
                     "copyright": False, "subjects": ["geography"],
                     "formats": {"text/plain": "https://www.gutenberg.org/ebooks/9.txt.utf-8"}}]}))
    assert book_discovery.discover_public_books("animal emotions")["added"] == []
    assert not library.INDEX_PATH.exists()


def test_local_experiment_preserves_source_evidence_and_uncertainty(tmp_path):
    j = Journal(tmp_path / "v3")
    h = Hippocampus(j, legacy_path=tmp_path / "legacy")
    first = h.encode(
        "Attention and memory influence learning in the human mind across experience.",
        source="library:a", epistemic_status="source_passage")
    second = h.encode(
        "Learning and memory can change how attention shapes everyday experience.",
        source="library:b", epistemic_status="source_passage")
    context = {"generation": 17, "hippocampus": h, "cortex": Cortex()}
    result = run_local_experiment({
        "experiment_type": "compare_sources",
        "hypothesis": "How do attention and memory contribute to learning?",
        "source_ids": [first["id"], second["id"]],
    }, context)
    assert result["performed"] and not result["confirmed"]
    assert result["status"] == "observed_comparison"
    assert set(result["evidence_ids"]) == {first["id"], second["id"]}
    assert result["assessments"] and result["next_question"]
    assert any(x["kind"] == "experiment_observation" for x in j.events(verify=True))


def test_semantic_rarity_is_derived_from_older_experiences(tmp_path):
    j = Journal(tmp_path / "v3")
    h = Hippocampus(j, legacy_path=tmp_path / "legacy")
    for name in ("one", "two", "three"):
        h.encode("Memory and attention change human learning in everyday experience.",
                 source="library:" + name, epistemic_status="source_passage")
    learner = SemanticLearner(j, h, WorldModel(j), Cortex())
    assert learner._term_frequency["memory"] >= 2
    assert learner._term_frequency.get("rareterm", 0) == 0


def test_new_archive_bound_and_lossless_chain(tmp_path, monkeypatch):
    assert SEGMENT_LIMIT >= 1_500_000
    import ubique.v3.storage as storage
    monkeypatch.setattr(storage, "SEGMENT_LIMIT", 300)
    j = Journal(tmp_path / "v3")
    for n in range(25):
        j.append("memory", {"sequence": n, "text": "remember this" * 10})
    assert [e["data"]["sequence"] for e in Journal(tmp_path / "v3").events(verify=True)] == list(range(25))


def test_completed_source_question_does_not_monopolize_followup(tmp_path):
    from ubique.config import Config
    from ubique.v3.runtime import UbiqueV3

    agent = UbiqueV3(Config.from_env(), storage_root=tmp_path / "v3",
                     enable_remote=False)
    for name in ("one", "two"):
        agent.hippocampus.encode(
            "Learning, attention and memory shape our understanding of experience.",
            source="library:" + name, epistemic_status="source_passage")
    from ubique.v3.inquiry import enquire
    inquiry = enquire(agent.hippocampus, agent.journal, agent.world, generation=6)
    assert inquiry is not None
    first = agent._candidates(11, "memory", [], [])
    experiments = [x for x in first if x.kind == "experiment"]
    assert experiments and experiments[0].payload["experiment_type"] == "compare_sources"
    result = agent._act(experiments[0], generation=11, focus="memory")
    assert result.success
    future = agent._candidates(22, "memory", [], [])
    next_experiment = [x for x in future if x.kind == "experiment"][0]
    assert next_experiment.payload["experiment_type"] == "association_probe"
