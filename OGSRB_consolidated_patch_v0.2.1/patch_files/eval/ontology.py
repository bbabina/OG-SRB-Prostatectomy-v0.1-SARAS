from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import yaml

# v0.2 is the canonical ontology for the corrected experimental protocol.
DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parent.parent
    / "ontology"
    / "ogsrb_prostatectomy_v0.2.yaml"
)


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
    phase_transition_policy: dict = field(default_factory=dict)

    @property
    def action_ids(self) -> set:
        return set(self.action_by_id.keys())

    def valid_node_ids(self) -> set:
        return (
            self.action_ids
            | self.tool_ids
            | self.tissue_ids
            | self.phase_ids
            | self.event_ids
        )

    def tool_for(self, action_id: str) -> str | None:
        node = self.action_by_id.get(action_id)
        return node.get("tool") if node else None

    def targets_for(self, action_id: str) -> list[str]:
        node = self.action_by_id.get(action_id)
        if not node:
            return []
        targets = node.get("targets")
        if targets:
            return list(targets)
        tissue = node.get("tissue")
        return [tissue] if tissue else []

    def tissue_for(self, action_id: str) -> str | None:
        """Legacy primary-target accessor retained for existing code."""
        targets = self.targets_for(action_id)
        return targets[0] if targets else None

    def event_for(self, action_id: str) -> str | None:
        node = self.action_by_id.get(action_id)
        return node.get("event") if node else None

    def phase_for(self, action_id: str) -> str | None:
        return self.phase_by_action.get(action_id)

    def ordered_phase_ids(self) -> list[str]:
        return sorted(self.phase_ids, key=lambda p: self.phase_order_index[p])

    def transition_cost(self, prev_phase: str | None, next_phase: str | None) -> float:
        """Return the v0.2 soft transition cost; inf means not permitted.

        v0.2 semantics:
          stay                -> 0.0
          adjacent forward    -> 0.1
          adjacent backward   -> 0.5
          non-adjacent        -> not permitted
        Values are read from the ontology when present.
        """
        if prev_phase is None or next_phase is None:
            return 0.0

        if (prev_phase, next_phase) not in self.phase_order:
            return math.inf

        policy = self.phase_transition_policy or {}
        costs = policy.get("recommended_costs", {})
        prev_i = self.phase_order_index[prev_phase]
        next_i = self.phase_order_index[next_phase]
        delta = next_i - prev_i

        if delta == 0:
            return float(costs.get("stay", 0.0))
        if delta == 1:
            return float(costs.get("adjacent_forward", 0.0))
        if delta == -1:
            return float(costs.get("adjacent_backward", 0.0))

        if policy.get("allow_nonadjacent_transitions", False):
            return float(costs.get("nonadjacent", 1.0))
        return math.inf

    def is_legal_phase_transition(
        self, prev_phase: str | None, next_phase: str | None
    ) -> bool:
        return math.isfinite(self.transition_cost(prev_phase, next_phase))


def load_ontology(path: Path | str = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    action_by_id = {a["id"]: a for a in raw.get("actions", [])}

    phases = raw.get("phases", [])
    phase_by_action = {}
    phase_order_index = {}
    for phase in phases:
        phase_id = phase["id"]
        phase_order_index[phase_id] = int(phase["order"])
        for action_id in phase.get("actions", []):
            phase_by_action[action_id] = phase_id

    requires = {tuple(pair) for pair in raw.get("requires", [])}
    acts_on = {tuple(pair) for pair in raw.get("acts_on", [])}
    phase_order = {tuple(pair) for pair in raw.get("phase_order", [])}
    contradicts = {frozenset(pair) for pair in raw.get("contradicts", [])}

    tool_ids = {t["id"] for t in raw.get("tools", [])}
    tissue_ids = set(raw.get("tissues", []))
    phase_ids = {p["id"] for p in phases}
    event_ids = {e["id"] for e in raw.get("events", [])}

    return Ontology(
        raw=raw,
        action_by_id=action_by_id,
        phase_by_action=phase_by_action,
        phase_order_index=phase_order_index,
        requires=requires,
        acts_on=acts_on,
        phase_order=phase_order,
        contradicts=contradicts,
        tool_ids=tool_ids,
        tissue_ids=tissue_ids,
        phase_ids=phase_ids,
        event_ids=event_ids,
        phase_transition_policy=raw.get("phase_transition_policy", {}),
    )
