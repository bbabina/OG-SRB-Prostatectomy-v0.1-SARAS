from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "ontology" / "ogsrb_prostatectomy.yaml"


@dataclass
class Ontology:
    raw: dict
    action_by_id: dict = field(default_factory=dict)
    phase_by_action: dict = field(default_factory=dict)
    phase_order_index: dict = field(default_factory=dict)
    requires: set = field(default_factory=set)
    acts_on: set = field(default_factory=set)
    phase_order: set = field(default_factory=set)
    contradicts: set = field(default_factory=set)
    tool_ids: set = field(default_factory=set)
    tissue_ids: set = field(default_factory=set)
    phase_ids: set = field(default_factory=set)
    event_ids: set = field(default_factory=set)

    @property
    def action_ids(self) -> set:
        return set(self.action_by_id.keys())

    def valid_node_ids(self) -> set:
        """Every id that counts as a legitimate ontology node, across all categories."""
        return self.action_ids | self.tool_ids | self.tissue_ids | self.phase_ids | self.event_ids

    def tool_for(self, action_id: str) -> str | None:
        node = self.action_by_id.get(action_id)
        return node.get("tool") if node else None

    def tissue_for(self, action_id: str) -> str | None:
        node = self.action_by_id.get(action_id)
        return node.get("tissue") if node else None

    def event_for(self, action_id: str) -> str | None:
        node = self.action_by_id.get(action_id)
        return node.get("event") if node else None

    def phase_for(self, action_id: str) -> str | None:
        return self.phase_by_action.get(action_id)

    def is_legal_phase_transition(self, prev_phase: str, next_phase: str) -> bool:
        if prev_phase is None or next_phase is None:
            return True  # nothing to compare (e.g. first segment, or generic-only segment)
        return (prev_phase, next_phase) in self.phase_order


def load_ontology(path: Path | str = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    action_by_id = {a["id"]: a for a in raw.get("actions", [])}

    phase_by_action = {}
    for phase in raw.get("phases", []):
        for action_id in phase.get("actions", []):
            phase_by_action[action_id] = phase["id"]

    requires = {tuple(pair) for pair in raw.get("requires", [])}
    acts_on = {tuple(pair) for pair in raw.get("acts_on", [])}
    phase_order = {tuple(pair) for pair in raw.get("phase_order", [])}
    contradicts = {frozenset(pair) for pair in raw.get("contradicts", [])}

    tool_ids = {t["id"] for t in raw.get("tools", [])}
    tissue_ids = set(raw.get("tissues", []))
    phase_ids = {p["id"] for p in raw.get("phases", [])}
    event_ids = {e["id"] for e in raw.get("events", [])}

    return Ontology(
        raw=raw,
        action_by_id=action_by_id,
        phase_by_action=phase_by_action,
        requires=requires,
        acts_on=acts_on,
        phase_order=phase_order,
        contradicts=contradicts,
        tool_ids=tool_ids,
        tissue_ids=tissue_ids,
        phase_ids=phase_ids,
        event_ids=event_ids,
    )
