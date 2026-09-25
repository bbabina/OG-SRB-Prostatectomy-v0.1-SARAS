# Recognition and ontology-grounded benchmark metrics.

from __future__ import annotations


def check_requires(segment: dict, onto) -> list[str]:
    """Legacy function name; measures canonical tool mapping conformance."""
    violations = []
    tool_set = set(segment.get("tools", []))
    for action in segment.get("actions", []):
        required_tool = onto.tool_for(action)
        if required_tool and required_tool not in tool_set:
            violations.append(
                f"canonical_tool_mapping: action {action} missing tool {required_tool}"
            )
    return violations


def check_acts_on(segment: dict, onto) -> list[str]:
    """Legacy function name; measures canonical target mapping conformance."""
    violations = []
    target_set = set(segment.get("tissues", []))
    for action in segment.get("actions", []):
        required_targets = set(onto.targets_for(action))
        missing = sorted(required_targets - target_set)
        if missing:
            violations.append(
                f"canonical_target_mapping: action {action} missing targets {missing}"
            )
    return violations


def check_contradicts(segment: dict, onto) -> list[str]:
    """Evaluate only contradiction rules actually present in the canonical ontology.

    Ontology v0.2 intentionally contains no validated global contradiction pairs, so
    this normally returns an empty list. The evaluator reports the rate as N/A when
    there are no validated rules rather than claiming a misleading 0%.
    """
    nodes = (
        set(segment.get("actions", []))
        | set(segment.get("tools", []))
        | set(segment.get("tissues", []))
        | set(segment.get("events", []))
    )
    if segment.get("phase"):
        nodes.add(segment["phase"])

    violations = []
    for pair in onto.contradicts:
        if pair.issubset(nodes):
            violations.append(f"validated_rule: {sorted(pair)} co-occur")
    return violations


def check_phase_order(segments_for_video: list[dict], onto) -> tuple[list[str], int]:
    """Proxy phase transition consistency under the v0.2 legal transition graph."""
    violations = []
    ordered = sorted(segments_for_video, key=lambda s: s["frame_start"])
    phased = [s for s in ordered if s.get("phase")]
    for prev, nxt in zip(phased, phased[1:]):
        if not onto.is_legal_phase_transition(prev["phase"], nxt["phase"]):
            violations.append(
                f"proxy_phase_transition: "
                f"{prev['segment_id']}({prev['phase']}) -> "
                f"{nxt['segment_id']}({nxt['phase']}) illegal"
            )
    return violations, (len(phased) - 1 if len(phased) > 1 else 0)


def schema_vocabulary_conformance(segment: dict, onto) -> float:
    valid_ids = onto.valid_node_ids()
    nodes = (
        list(segment.get("actions", []))
        + list(segment.get("tools", []))
        + list(segment.get("tissues", []))
        + list(segment.get("events", []))
    )
    if segment.get("phase"):
        nodes.append(segment["phase"])
    if not nodes:
        return 1.0
    n_valid = sum(1 for n in nodes if n in valid_ids)
    return n_valid / len(nodes)


# Backwards-compatible alias; do not use this old name in new reports.
ontology_factuality = schema_vocabulary_conformance


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def temporal_coherence(pred_phase_seq: list[str], true_phase_seq: list[str]) -> float:
    max_len = max(len(pred_phase_seq), len(true_phase_seq))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(pred_phase_seq, true_phase_seq)
    return 1 - dist / max_len


def macro_f1(
    pred_by_segment: dict[str, list[str]],
    gt_by_segment: dict[str, list[str]],
    action_classes: list[str],
) -> dict:
    """Multi-label macro-F1 over action classes, matched by segment_id."""
    per_class = {}
    for cls in action_classes:
        tp = fp = fn = 0
        for seg_id, gt_actions in gt_by_segment.items():
            pred_actions = set(pred_by_segment.get(seg_id, []))
            gt_has = cls in gt_actions
            pred_has = cls in pred_actions
            if gt_has and pred_has:
                tp += 1
            elif pred_has and not gt_has:
                fp += 1
            elif gt_has and not pred_has:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    classes_with_support = [
        c for c in action_classes if per_class[c]["support"] > 0
    ]
    macro = (
        sum(per_class[c]["f1"] for c in classes_with_support)
        / len(classes_with_support)
        if classes_with_support
        else 0.0
    )
    return {"macro_f1": macro, "per_class": per_class}


def top1_accuracy(
    pred_top_by_segment: dict[str, str | None],
    gt_by_segment: dict[str, list[str]],
) -> float:
    """Count explicit no-prediction/parse failures as incorrect, not as missing."""
    n = 0
    correct = 0
    for seg_id, gt_actions in gt_by_segment.items():
        if seg_id not in pred_top_by_segment:
            continue
        n += 1
        pred = pred_top_by_segment[seg_id]
        if pred is not None and pred in gt_actions:
            correct += 1
    return correct / n if n else 0.0
