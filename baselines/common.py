"""Shared helpers for VLM baseline scripts.

Every baseline (CLIP zero-shot, the CLIP linear probe, and later LLaVA/
GPT-4V) ends up with the same problem: it predicts a set of *actions* for a
segment, and that needs to be translated into the full ontology-node record
(tools/tissues/events/phase) so eval/evaluator.py can score any baseline the
same way, in the same schema map_annotations.py uses for ground truth.
"""
from __future__ import annotations

from collections import Counter


def resolve_phase_from_actions(actions: list[str], onto) -> tuple[str | None, bool, list[str]]:
    """Equal-weight majority vote over predicted actions (no per-frame data
    available for predictions, unlike map_annotations.py's frame-weighted
    version for ground truth)."""
    votes = Counter(onto.phase_for(a) for a in actions if onto.phase_for(a))
    if not votes:
        return None, False, []
    phase = votes.most_common(1)[0][0]
    return phase, len(votes) > 1, sorted(votes.keys())


def derive_nodes(actions: list[str], onto) -> tuple[list[str], list[str], list[str]]:
    tools = sorted({onto.tool_for(a) for a in actions if onto.tool_for(a)})
    tissues = sorted({onto.tissue_for(a) for a in actions if onto.tissue_for(a)})
    events = sorted({onto.event_for(a) for a in actions if onto.event_for(a)})
    return tools, tissues, events


def build_prediction_record(seg: dict, model_name: str, pred_actions: list[str],
                             action_probs: dict[str, float], onto) -> dict:
    tools, tissues, events = derive_nodes(pred_actions, onto)
    phase, phase_ambiguous, phase_candidates = resolve_phase_from_actions(pred_actions, onto)
    return {
        "segment_id": seg["segment_id"],
        "split": seg["split"],
        "video": seg["video"],
        "frame_start": seg["frame_start"],
        "frame_end": seg["frame_end"],
        "model": model_name,
        "actions": pred_actions,
        "action_probs": action_probs,
        "tools": tools,
        "tissues": tissues,
        "events": events,
        "phase": phase,
        "phase_ambiguous": phase_ambiguous,
        "phase_candidates": phase_candidates,
    }
