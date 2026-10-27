from ubique.v3.change_feedback import review_changes
from ubique.v3.curiosity import Curiosity
from ubique.v3.semantics import SemanticLearner


class FakeJournal:
    def __init__(self, events):
        self.history = events
    def events(self):
        return self.history
    def recent(self, kind, *, limit=30):
        return [e for e in self.history if e["kind"] == kind][-limit:]
    def append(self, kind, data, *, generation=None):
        self.history.append({"id": "review-event", "kind": kind,
                             "generation": generation, "data": data})


def event(identifier, kind, generation, data=None):
    return {"id": identifier, "kind": kind, "generation": generation,
            "data": data or {}}


def test_change_review_records_actual_before_after_activity_not_success_claim():
    history = [
        event("old", "association_hypothesis", 11),
        event("edit", "change_outcome", 25, {
            "success": True, "intent": {"target": "src/ubique/v3/semantics.py"}}),
        event("new1", "association_hypothesis", 29),
        event("new2", "association_hypothesis", 31),
    ]
    journal = FakeJournal(history)
    review = review_changes(journal, 50, observation_window=24)
    assert review["observed_event_kind"] == "association_hypothesis"
    assert review["events_before"] == 1
    assert review["events_after"] == 2
    assert review["verdict"] == "activity_changed"
    assert review_changes(journal, 52, observation_window=24) is None


def test_review_does_not_pretend_an_unexercised_code_path_worked():
    journal = FakeJournal([
        event("edit", "change_outcome", 25, {
            "success": True, "intent": {"target": "src/ubique/v3/inquiry.py"}})
    ])
    result = review_changes(journal, 50, observation_window=24)
    assert result["verdict"] == "not_observed"


def test_long_catalogue_metadata_loses_to_a_real_interest_in_attention():
    curiosity = Curiosity(FakeJournal([]))
    topic = "memory"
    catalogue = ("Illustrated catalogue of an historical collection of "
                 "objects and observations, from a report and a publication " * 2)
    assert curiosity.select([catalogue, topic], generation=24) == topic


def test_quotation_alone_does_not_count_as_new_semantic_comparison():
    learner = object.__new__(SemanticLearner)
    learner.exploration_factor = .10
    learner._no_comp_counter = 0
    learner._adjust_exploration_factor(produced_comparisons=False)
    assert learner._no_comp_counter == 1
    assert learner.exploration_factor > .10
    learner._adjust_exploration_factor(produced_comparisons=True)
    assert learner._no_comp_counter == 0
