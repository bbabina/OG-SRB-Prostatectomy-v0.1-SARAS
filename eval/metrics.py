"""Reasoning-validity and recognition metrics (project doc, Section 4 Step 5
and Section 5 Evaluation Design).

This module is intentionally agnostic to *where* a segment's predicted
nodes came from — the same functions score ground-truth annotations
(data_prep/validate_annotations.py, sanity-checking the ontology itself)
and model predictions (eval/evaluator.py, scoring CLIP/LLaVA/GPT-4V). A
"segment" here is any dict with: actions, tools, tissues, events, phase.

Ontology-rule metrics (per segment / per video):
  - requires    -> Requirement Satisfaction (RS)
  - acts_on     -> Acts-On Consistency (AOC)
  - contradicts -> Contradiction Rate (CR)
  - phase_order -> Step-Ordering Consistency (SOC)
  - node validity -> Ontology Factuality (OF)
  - phase sequence vs ground truth -> Temporal Coherence (TC)

Recognition metrics (predictions vs ground truth, per action class):
  - Macro-F1, Top-1 accuracy
"""
from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Ontology-rule checks (operate on a single segment's predicted/annotated
# node set, or on a video's ordered segment list)
# ---------------------------------------------------------------------------

def check_requires(segment: dict, onto) -> list[str]:
    violations = []
    tool_set = set(segment["tools"])
    for action in segment["actions"]:
        required_tool = onto.tool_for(action)
        if required_tool and required_tool not in tool_set:
            violations.append(f"requires: action {action} missing tool {required_tool}")
    return violations


def check_acts_on(segment: dict, onto) -> list[str]:
    violations = []
    tissue_set = set(segment["tissues"])
    for action in segment["actions"]:
        tissue = onto.tissue_for(action)
        if tissue and tissue not in tissue_set:
            violations.append(f"acts_on: action {action} missing tissue {tissue}")
    return violations


def check_contradicts(segment: dict, onto) -> list[str]:
    nodes = set(segment["actions"]) | set(segment["tools"]) | set(segment["tissues"]) | set(segment["events"])
    if segment.get("phase"):
        nodes.add(segment["phase"])
    violations = []
    for pair in onto.contradicts:
        if pair.issubset(nodes):
            violations.append(f"contradicts: {sorted(pair)} co-occur")
    return violations


def check_phase_order(segments_for_video: list[dict], onto) -> tuple[list[str], int]:
    violations = []
    ordered = sorted(segments_for_video, key=lambda s: s["frame_start"])
    phased = [s for s in ordered if s.get("phase")]
    for prev, nxt in zip(phased, phased[1:]):
        if not onto.is_legal_phase_transition(prev["phase"], nxt["phase"]):
            violations.append(
                f"phase_order: {prev['segment_id']}({prev['phase']}) -> "
                f"{nxt['segment_id']}({nxt['phase']}) illegal"
            )
    return violations, (len(phased) - 1 if len(phased) > 1 else 0)


def ontology_factuality(segment: dict, onto) -> float:
    """Fraction of this segment's predicted nodes that are valid ontology ids.

    On closed-vocabulary baselines (CLIP zero-shot restricted to the 21
    action prompts) this is always 1.0 by construction — it only becomes
    discriminating once a model can emit free text (LLaVA/GPT-4V), which is
    exactly the point of the metric per the project doc.
    """
    valid_ids = onto.valid_node_ids()
    nodes = list(segment["actions"]) + list(segment["tools"]) + list(segment["tissues"]) + list(segment["events"])
    if segment.get("phase"):
        nodes.append(segment["phase"])
    if not nodes:
        return 1.0
    n_valid = sum(1 for n in nodes if n in valid_ids)
    return n_valid / len(nodes)


# ---------------------------------------------------------------------------
# Temporal Coherence: edit distance between predicted and true phase
# sequences for a video (1 - normalized Levenshtein distance)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Recognition metrics: predicted actions vs ground-truth actions, per segment
# ---------------------------------------------------------------------------

def macro_f1(pred_by_segment: dict[str, list[str]], gt_by_segment: dict[str, list[str]],
             action_classes: list[str]) -> dict:
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    classes_with_support = [c for c in action_classes if per_class[c]["support"] > 0]
    macro = sum(per_class[c]["f1"] for c in classes_with_support) / len(classes_with_support) if classes_with_support else 0.0
    return {"macro_f1": macro, "per_class": per_class}


def top1_accuracy(pred_top_by_segment: dict[str, str], gt_by_segment: dict[str, list[str]]) -> float:
    """Fraction of segments where the model's single top-ranked action is
    anywhere in that segment's (possibly multi-label) ground-truth action set."""
    n = 0
    correct = 0
    for seg_id, gt_actions in gt_by_segment.items():
        if seg_id not in pred_top_by_segment:
            continue
        n += 1
        if pred_top_by_segment[seg_id] in gt_actions:
            correct += 1
    return correct / n if n else 0.0
