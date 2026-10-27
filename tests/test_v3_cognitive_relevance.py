"""Prevent source-title noise and false evidence in autonomous inquiry."""
from types import SimpleNamespace

from ubique.v3.inquiry import compare_sources, enquire
from ubique.v3.text_quality import concept_terms
from ubique.v3.workspace import Workspace


MEMORY = (
    "The author connects memory and attention to the formation of enduring "
    "experience and suggests that learned expectations affect later perception."
)
LEARNING = (
    "The researcher discusses attention and memory in the development of "
    "human learning, particularly how knowledge shapes future perception."
)
ARCHAEOLOGY = (
    "The archaeological report documents ancient ceramic vessels and "
    "regional excavation techniques in the remains of a desert settlement."
)


def episode(identifier, source, content):
    return {"id": identifier, "source": "library:" + source,
            "text": content, "epistemic_status": "source_passage"}


def test_source_concepts_are_not_generic_connectives_or_catalogue_labels():
    words = concept_terms(
        "Every other illustrated catalogue allows attention and memory", 10)
    assert "memory" in words and "attention" in words
    assert "every" not in words and "illustrated" not in words


def test_grounded_question_keeps_two_real_source_ids():
    result = compare_sources(
        episode("e1", "james", MEMORY), episode("e2", "baldwin", LEARNING))
    assert result is not None
    assert result["bridge_quality"] == "source_linked"
    assert "memory" in result["topics"] or "attention" in result["topics"]
    assert set(result["source_ids"]) == {"e1", "e2"}


def test_unrelated_passages_are_not_misreported_as_an_observed_connection():
    result = compare_sources(
        episode("e1", "archive", ARCHAEOLOGY), episode("e2", "james", MEMORY))
    assert result is not None
    assert result["bridge_quality"] == "speculative"
    assert result["topics"] == []
    assert "Is there any meaningful relationship" in result["question"]


def test_inquiry_prefers_genuine_topic_overlap_to_nearest_unrelated_document():
    class Journal:
        def __init__(self):
            self.saved = []
        def events(self):
            return []
        def append(self, kind, data, **kwargs):
            self.saved.append((kind, data))

    class Hippocampus:
        def all_episodes(self):
            # Deliberately puts unrelated first in the reverse recent ordering.
            return [episode("e2", "baldwin", LEARNING),
                    episode("e1", "james", MEMORY),
                    episode("e3", "archive", ARCHAEOLOGY)]

    class World:
        def claim(self, *args, **kwargs):
            return {"id": "c1"}

    journal = Journal()
    chosen = enquire(Hippocampus(), journal, World(), generation=23)
    assert chosen is not None
    assert chosen["bridge_quality"] == "source_linked"
    assert set(chosen["source_ids"]) == {"e1", "e2"}


def test_workspace_retains_publishing_noise_but_prefers_lived_concepts():
    nodes = [
        SimpleNamespace(id="concept:1", label="every", activation=1,
                        use_count=0, last_active=4),
        SimpleNamespace(id="concept:2", label="memory", activation=.65,
                        use_count=2, last_active=4),
        SimpleNamespace(id="concept:3", label="attention", activation=.65,
                        use_count=2, last_active=4),
    ]
    cortex = SimpleNamespace(nodes={node.id: node for node in nodes}, tick=4)
    labels = [item.label for item in Workspace(3).compete(
        cortex, grounding={"attention", "memory", "every"})]
    assert labels.index("memory") < labels.index("every")
    assert labels.index("attention") < labels.index("every")
