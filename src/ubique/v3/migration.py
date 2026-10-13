"""Reversible v2->v3 migration. Never delete or rewrite historical v2 records."""
from __future__ import annotations
import json
from ..state import ROOT
from .representation import Cortex
from .storage import Journal

LEGACY = {
    "cortex":"memory/cortex.json",
    "brain_episodes":"memory/brain_episodes.jsonl",
    "self_model":"memory/self_model.jsonl",
    "library_index":"memory/library/index.json",
    "concepts":"memory/concepts.jsonl",
    "hypotheses":"memory/hypotheses.jsonl",
    "projects":"state/projects.json",
    "skills":"memory/skills.json",
}

def inventory(journal:Journal) -> dict:
    report={"schema_version":3,"legacy_sources":{},"counts":{}}
    for name,rel in LEGACY.items():
        path=ROOT/rel
        if not path.exists():
            report["legacy_sources"][name]={"exists":False}
            continue
        text=path.read_text(encoding="utf-8")
        if rel.endswith(".jsonl"):
            count=sum(1 for line in text.splitlines() if line.strip())
        else:
            data=json.loads(text)
            count=len(data.get("items",[])) if name=="library_index" else (
                len(data.get("nodes",[])) if name=="cortex" else 1)
        report["legacy_sources"][name]={"exists":True,"records":count,
                                        "size_bytes":path.stat().st_size}
    return report

def bootstrap(journal:Journal, *, dry_run:bool=False) -> tuple[Cortex,dict]:
    report=inventory(journal)
    if dry_run:return Cortex(),report
    snapshot=journal.load_snapshot()
    if snapshot:
        graph=Cortex.from_dict(snapshot["graph"])
        for event in journal.events(after=int(snapshot.get("events",0))):
            if event.get("kind")=="graph_delta":
                for item in event["data"].get("changes",[]):graph.apply(item)
            elif event.get("kind") in {"node_upsert","synapse_upsert","cortex_tick"}:
                graph.apply(event)
        report["loaded_from"]="v3_snapshot_and_deltas"
        return graph,report
    # If no v3 snapshot exists, previously collected events alone are still replayable.
    # The first import preserves stable v2 identifiers and actual synaptic strengths.
    graph=Cortex.from_legacy()
    for name,rel in LEGACY.items():
        path=ROOT/rel
        if path.exists():
            journal.migration_record(name,path,
                records=int(report["legacy_sources"][name]["records"]))
    journal.snapshot(graph.as_dict())
    report["loaded_from"]="v2_import"
    report["migrated_synapses"]=len(graph.edges)
    report["migrated_nodes"]=len(graph.nodes)
    return graph,report

def continuity(journal:Journal) -> dict:
    return {"migration_integrity":journal.verify_migration(),
            "journal_events":journal.manifest.get("events",0),
            "snapshot_events":journal.manifest.get("snapshot_events",0),
            "legacy_files_intact":all(journal.verify_migration().values())}
