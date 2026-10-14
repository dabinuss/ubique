"""Regression coverage: do not mistake associations for grounded thought."""
import json
from types import SimpleNamespace

from ubique.config import Config
from ubique.v3.evidence import assess
from ubique.v3.storage import Journal
from ubique.v3.hippocampus import Hippocampus
from ubique.v3.runtime import UbiqueV3
from ubique.v3.representation import Cortex
from ubique.v3.workspace import Workspace


def _source(text, key="v3:one"):
    return {"id":key,"text":text,"source":"library:book","epistemic_status":"source_passage"}


def test_evidence_screen_rejects_irrelevant_citation():
    result=assess("Grass is composed of cobalt and gold",
        [_source("Experiments on animal perception distinguish memory from attention.")])
    assert result.verdict=="unrelated"
    assert result.cited==1


def test_evidence_screen_accepts_direct_textual_correspondence():
    passage="Studies of memory and attention demonstrate the contribution of attention to learning."
    result=assess("Memory and attention contribute to learning",[_source(passage)])
    assert result.verdict in {"plausible","directly_supported"}
    assert result.matches>=2


def test_new_stances_require_relevant_text_not_just_valid_ids(tmp_path):
    bot=UbiqueV3(Config.from_env(),storage_root=tmp_path/"v3",enable_remote=False)
    episode=bot.hippocampus.encode(
        "Memory and attention contribute to learning in human psychology.",
        source="library:psychology",epistemic_status="source_passage")
    bot._packet_recall=[episode]
    proposed={"provider":"groq","actions":[],"associations":[],"questions":[],"hypotheses":[],
              "world_model_updates":[
                {"statement":"The planet is made of glass",
                 "confidence":0.9,"evidence_ids":[episode["id"]]},
                {"statement":"Memory and attention contribute to learning",
                 "confidence":0.5,"evidence_ids":[episode["id"]]}],
              "stance_updates":[
                {"topic":"memory","position":"Memory and attention contribute to learning",
                 "confidence":0.9,"evidence_ids":[episode["id"]]},
                {"topic":"unrelated","position":"The planet is made of glass",
                 "confidence":0.9,"evidence_ids":[episode["id"]]}]}
    bot._handle_model(proposed,42)
    assert bot.world.stances.get("memory")
    assert not bot.world.stances.get("unrelated")
    assert bot.world.stances["memory"]["uncertainty"]>=.6
    assert any(x["status"]=="HYPOTHESIS" and "glass" in x["statement"]
               for x in bot.world.claims.values())
    assert any(e["kind"]=="evidence_review" for e in bot.journal.events())


def test_legacy_infrastructure_nodes_do_not_dominate_working_memory():
    graph=Cortex()
    graph.activate(["runtime", "issue:21", "constraints", "plant evolution",
                    "human psychology", "animal adaptation"],strength=1)
    graph.tick=1
    workspace=Workspace(capacity=3)
    chosen=[item.label for item in workspace.compete(graph)]
    assert "runtime" not in chosen
    assert "issue:21" not in chosen
    assert "plant evolution" in chosen


def test_journal_shard_index_skips_archived_data_for_snapshot_replay(tmp_path,monkeypatch):
    import ubique.v3.storage as store
    monkeypatch.setattr(store,"SEGMENT_LIMIT",400)
    j=Journal(tmp_path/"v3")
    for n in range(35):
        j.append("experience",{"text":"biology"*30,"number":n})
    snap_count=j.manifest["events"]
    j.snapshot({"tick":0,"nodes":[],"edges":[]})
    for n in range(5):
        j.append("experience",{"number":n,"text":"brain"*40})
    reopened=Journal(tmp_path/"v3")
    assert len(reopened.manifest["segment_counts"])>=2
    assert len(reopened.events(after=snap_count))==5
    assert reopened._cached_events is None
    assert len(reopened.events(verify=True))==40


def test_v3_evolution_generates_source_grounded_reversible_proposal(tmp_path,monkeypatch):
    import ubique.v3.evolution as evolution
    from ubique.v3.evolution import Evolution
    source=tmp_path/"src"/"ubique"/"v3"/"workspace.py"
    source.parent.mkdir(parents=True)
    source.write_text("def think():\n    return 1\n")
    monkeypatch.setattr(evolution,"ROOT",tmp_path)
    prompts=[]
    class Brain:
        def __init__(self,config):self.router=self
        def generate(self,prompt):
            prompts.append(prompt)
            return SimpleNamespace(provider="groq",text=json.dumps({
                "title":"Try new associations", "summary":"Add a local learning idea",
                "changes":[{"path":"src/ubique/v3/workspace.py","content":"def think():\n    return 2\n"}]}))
    monkeypatch.setattr(evolution,"NeurocognitiveRuntime",Brain)
    monkeypatch.setattr(evolution,"EVOLVABLE",("src/ubique/v3/workspace.py",))
    from ubique.evolution import EvolutionOutcome
    monkeypatch.setattr(evolution,"run_evolution",lambda *a,**k:EvolutionOutcome(
        accepted=True,title="Try new associations",summary="test",branch="test",pr_url="url",
        commit_sha="abcd",reason=None,baseline_score=None,candidate_score=None))
    journal=Journal(tmp_path/"memory"/"v3")
    result=Evolution(journal,Config.from_env()).attempt(
        "Observed too many repetitive thought associations",generation=500)
    assert result["success"]
    assert prompts and "Ubique v3" in prompts[0]
    assert "def think()" in prompts[0]
    kinds=[x["kind"] for x in journal.events(verify=True)]
    assert "change_intent" in kinds and "change_outcome" in kinds


def test_reading_candidate_prefers_progressing_real_book(monkeypatch,tmp_path):
    import ubique.v3.actions as actions
    from ubique.v3.curiosity import Curiosity
    rows=[{"kind":"book","id":"a","title":"Learning", "read_count":0,
           "fully_read":False,"content_available":True},
          {"kind":"book","id":"b","title":"A Human Mind","read_count":8,
           "fully_read":False,"content_available":True}]
    monkeypatch.setattr(actions,"library_catalog",lambda n:rows)
    selected=actions.reading_candidates(Curiosity(Journal(tmp_path)))
    assert selected[0].payload["item_id"]=="b"


def test_negation_in_unrelated_sentence_does_not_invert_evidence():
    passage=("Memory and attention contribute to learning in human psychology. "
             "A separate debate has not reached an agreement about weather patterns.")
    result=assess("Memory and attention contribute to learning",[_source(passage)])
    assert result.verdict in {"plausible","directly_supported"}
    assert not result.explicit_negation_conflict


def test_explicitly_inverted_claim_is_not_accepted():
    passage="Memory and attention contribute to learning in human psychology."
    result=assess("Memory and attention do not contribute to learning",[_source(passage)])
    assert result.verdict=="contested"
    assert result.explicit_negation_conflict


def test_infrastructure_filter_keeps_nontechnical_phrase():
    from ubique.v3.evidence import is_system_topic
    assert is_system_topic("continuous")
    assert is_system_topic("runtime ↔ status")
    assert not is_system_topic("the continuous emergence of life")
    assert not is_system_topic("human psychological constraints")


def test_reading_sessions_continue_then_rotate(monkeypatch,tmp_path):
    from ubique.v3.actions import Candidate
    import ubique.v3.runtime as runtime
    def books(_):
        return [Candidate("read","a",{"title":"book a","item_id":"a"},.65),
                Candidate("read","b",{"title":"book b","item_id":"b"},.65)]
    monkeypatch.setattr(runtime,"reading_candidates",books)
    agent=UbiqueV3(Config.from_env(),storage_root=tmp_path/"v3",enable_remote=False)
    for n in range(1):
        agent.journal.append("action_outcome",{"kind":"read","success":True,
                                           "payload":{"title":"book a"}},generation=n)
    scored=agent._candidates(10,"uncertainty",[],[])
    values={a.payload.get("title"):a.score for a in scored if a.kind=="read"}
    assert values["book a"]>values["book b"]
    for n in range(1,4):
        agent.journal.append("action_outcome",{"kind":"read","success":True,
                                           "payload":{"title":"book a"}},generation=n)
    scored=agent._candidates(15,"uncertainty",[],[])
    values={a.payload.get("title"):a.score for a in scored if a.kind=="read"}
    assert values["book b"]>values["book a"]


def test_reading_grounded_workspace_outranks_generic_old_issue_vocabulary():
    brain=Cortex()
    labels=["uninterrupted","expanded","lightweight","successful",
            "memory","attention","psychology","evolution"]
    brain.activate(labels,strength=.85)
    brain.tick=1
    workspace=Workspace(capacity=4)
    selected=[i.label for i in workspace.compete(
        brain,grounding={"memory","attention","psychology","evolution"})]
    assert set(selected)=={"memory","attention","psychology","evolution"}


def test_technical_memories_remain_retrievable_after_grounded_competition():
    brain=Cortex()
    brain.activate(["runtime","attention","knowledge","memory"],strength=1)
    Workspace().compete(brain,grounding={"attention","knowledge","memory"})
    assert any(n.label=="runtime" for n in brain.nodes.values())


def test_real_book_source_is_primed_before_focus_selection(tmp_path):
    brain=UbiqueV3(Config.from_env(),storage_root=tmp_path/"v3",enable_remote=False)
    reading=brain.hippocampus.encode(
        "Biological evolution changes the relationships among organisms. "
        "Memory and attention influence human learning and perception.",
        source="library:brain-and-mind",
        epistemic_status="source_passage",
        payload={"title":"Human Brain and Biological Evolution"})
    labels=set(["human","brain","biological","evolution","memory","attention"])
    focus, episodes=brain._recent_focus(1000)
    activated={n.label for n in brain.cortex.nodes.values()
               if n.activation>=.45 and n.label in labels}
    assert len(activated)>=3,activated
    assert any(x.label in activated for x in brain.workspace.items)
