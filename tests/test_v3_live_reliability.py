"""Tests for live October 24 fixes: retryable sources, focused questions and compact evolution."""
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from ubique.v3 import book_discovery
from ubique.v3.inquiry import compare_sources
from ubique.v3.text_quality import content_terms
from ubique.v3.actions import Candidate


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return json.dumps(self.value).encode("utf-8")


def _empty_library(tmp_path, monkeypatch):
    import ubique.library as library
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(library, "ITEMS_DIR", tmp_path / "items")
    return library


def test_dead_book_catalogue_uses_only_one_timeout(tmp_path, monkeypatch):
    _empty_library(tmp_path, monkeypatch)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(book_discovery, "urlopen", unavailable)
    result = book_discovery.discover_public_books("Consciousness and memory")
    assert not result["added"]
    assert result["status"] == "source_unavailable"
    assert len(calls) == 1 and calls[0] <= 12
    assert len(result["attempted_queries"]) == 1


def test_catalogue_topic_fallback_discovers_genuine_source(tmp_path, monkeypatch):
    library = _empty_library(tmp_path, monkeypatch)
    requests = []

    def respond(request, timeout=8):
        params = parse_qs(urlsplit(request.full_url).query)
        requests.append(params)
        if "topic" not in params:
            return FakeResponse({"results": []})
        return FakeResponse({"results": [{
            "title": "Introduction to Human Emotion and Physiology",
            "id": 827, "copyright": False,
            "subjects": ["psychology", "emotion", "physiology"],
            "formats": {"text/plain; charset=utf-8":
                        "https://www.gutenberg.org/ebooks/827.txt.utf-8"}
        }]})

    monkeypatch.setattr(book_discovery, "urlopen", respond)
    found = book_discovery.discover_public_books("Autonomic emotional crying")
    assert len(found["added"]) == 1
    assert any("topic" in request for request in requests)
    book = library.library_catalog()[0]
    assert book["status"] == "available"
    assert book["read_count"] == 0
    assert library._load_index()["items"][0]["public_domain_catalogue_id"] == 827
    assert book["source_url"].startswith("https://www.gutenberg.org/")


def test_common_grammar_no_longer_creates_conceptual_questions():
    for word in ("have", "true", "concerning", "regarding", "however"):
        assert word not in content_terms("Memory and attention " + word + " philosophy")
    a = {"id": "a", "source": "library:a",
         "text": "Our memory of living experience influences how attention makes concepts "
                 "available to conscious judgment across different situations."}
    b = {"id": "b", "source": "library:b",
         "text": "Reflective knowledge arises through memory and perception whenever "
                 "experience informs the understanding of the mind."}
    result = compare_sources(a, b)
    assert result is not None
    assert "memory" in result["topics"]
    assert result["distinctive_terms"][0] and result["distinctive_terms"][1]
    assert result["question"].endswith("? Which assumptions differ?")
    assert "concerning" not in result["question"]


def test_compact_evolution_replacements_preserve_full_source(tmp_path, monkeypatch):
    import ubique.evolution as evolution

    target = tmp_path / "src" / "ubique" / "v3" / "inquiry.py"
    target.parent.mkdir(parents=True)
    target.write_text("def enquiry():\n    return 'unexamined'\n", encoding="utf-8")
    monkeypatch.setattr(evolution, "ROOT", tmp_path)
    proposal = {"title": "Improve inquiry", "changes": [{
        "path": "src/ubique/v3/inquiry.py",
        "replacements": [{"old": "return 'unexamined'",
                          "new": "return 'open to revision'"}]}]}
    parsed = evolution.validate_proposal(json.dumps(proposal))
    assert parsed["changes"][0]["content"] == (
        "def enquiry():\n    return 'open to revision'\n")
    proposal["changes"][0]["replacements"][0]["old"] = "unknown source"
    with pytest.raises(ValueError, match="exactly once"):
        evolution.validate_proposal(json.dumps(proposal))


def test_earlier_failed_book_search_retried_without_busy_loop(tmp_path, monkeypatch):
    import ubique.library as library
    import ubique.v3.runtime as runtime
    from ubique.config import Config

    monkeypatch.setattr(library, "library_catalog", lambda limit=250: [
        {"kind": "book", "status": "wanted", "title": "A Study of Memory"}])
    monkeypatch.setattr(runtime, "reading_candidates", lambda cur: [
        Candidate("read", "Continue reading", {"title": "Test", "item_id": "t"}, .7)])
    bot = runtime.UbiqueV3(Config.from_env(), storage_root=tmp_path / "v3",
                           enable_remote=False)
    bot.journal.append("library_discovery", {
        "query": "Memory", "success": False, "status": "source_unavailable",
        "effect": "TimeoutError: read operation timed out"}, generation=99)
    existing = [Candidate("read", "Continue reading", {"title": "Test"}, .7)]
    assert bot._discovery_candidate(100, "memory", existing) is None
    due = bot._discovery_candidate(105, "memory", existing)
    assert due is not None and due.kind == "library"
    assert due.payload["action"] == "discover"
    assert bot._discovery_candidate(105, "memory", [
        Candidate("search", "Already researching", {"topic": "memory"}, .7)
    ]) is None
