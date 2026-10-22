"""Regression coverage for substantive source links and autonomous reading."""
import json

from ubique.v3.semantics import meaningful_overlap, SemanticLearner
from ubique.v3 import book_discovery
from ubique.v3.storage import Journal
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.world_model import WorldModel
from ubique.v3.representation import Cortex


def test_generic_words_cannot_create_cross_book_themes():
    assert meaningful_overlap({"every", "them", "which", "about"}) == set()
    assert meaningful_overlap({"every", "memory", "attention"}) == {"memory", "attention"}


def test_persisted_comparison_includes_quotes_and_revisable_question(tmp_path):
    journal = Journal(tmp_path / "v3")
    hip = Hippocampus(journal, legacy_path=tmp_path / "none")
    learner = SemanticLearner(journal, hip, WorldModel(journal), Cortex())
    first = hip.encode(
        "Memory and attention influence learning in human minds. "
        "Human awareness is shaped by experience and memory.",
        source="library:first", epistemic_status="source_passage")
    learner.observe(first, generation=1)
    second = hip.encode(
        "Human learning changes with attention and memory across experiences. "
        "Attention shapes the interpretation of memories in the mind.",
        source="library:second", epistemic_status="source_passage")
    learner.observe(second, generation=2)
    comparisons = [e["data"] for e in journal.events(verify=True)
                   if e["kind"] == "association_hypothesis"]
    assert comparisons
    last = comparisons[-1]
    assert len(last["passages"]) == 2 and all(last["passages"])
    assert "question" in last and len(last["evidence_ids"]) == 2
    assert "every" not in last["topic"]


def test_catalogue_discovery_fulfills_exact_wish_without_fake_read(tmp_path, monkeypatch):
    import ubique.library as library
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(library, "ITEMS_DIR", tmp_path / "items")
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")
    wish = library.request_library_item("Memory and Attention")["item"]

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            return json.dumps({"results": [{
                "title": "Memory and Attention", "id": 42,
                "copyright": False,
                "formats": {"text/plain; charset=utf-8":
                            "https://www.gutenberg.org/ebooks/42.txt.utf-8"}
            }]}).encode("utf-8")

    monkeypatch.setattr(book_discovery, "urlopen", lambda *a, **k: Response())
    result = book_discovery.discover_public_books("Memory and Attention")
    assert len(result["added"]) == 1
    items = library.library_catalog()
    assert len(items) == 1
    assert items[0]["id"] == wish["id"]
    assert items[0]["content_available"] is True
    assert items[0]["read_count"] == 0
    assert not items[0]["fully_read"]
    assert book_discovery.discover_public_books("Memory and Attention")["added"] == []


def test_unrelated_catalogue_result_does_not_get_installed(tmp_path, monkeypatch):
    import ubique.library as library
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            return json.dumps({"results": [{
                "title": "Volcanoes", "id": 5, "copyright": False,
                "formats": {"text/plain": "https://untrusted.example/book.txt"}
            }]}).encode("utf-8")
    monkeypatch.setattr(book_discovery, "urlopen", lambda *a, **k: Response())
    assert book_discovery.discover_public_books("memory")["added"] == []
    assert not library.INDEX_PATH.exists()
