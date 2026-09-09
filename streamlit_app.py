from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.metrics import check_requires, check_acts_on, check_contradicts  # noqa: E402

REPORTS_DIR = ROOT / "reports"
VLM_OUTPUTS_DIR = ROOT / "vlm_outputs"
ANNOTATIONS_DIR = ROOT / "annotations"
MESAD_ROOT = ROOT.parent / "mesad-real 2"

st.set_page_config(page_title="OG-SRB Visual Reasoning Explorer", layout="wide")


@st.cache_resource
def get_ontology():
    return load_ontology(DEFAULT_ONTOLOGY_PATH)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data
def build_metrics_table() -> pd.DataFrame:
    rows = []
    for path in sorted(REPORTS_DIR.glob("metrics_*.json")):
        d = json.loads(path.read_text())
        rows.append({
            "model": d.get("model", path.stem),
            "scope": d.get("evaluation_scope", "?"),
            "split": d.get("split", "?"),
            "n_segments": d.get("n_segments_evaluated"),
            "top1_accuracy": d.get("top1_accuracy"),
            "macro_f1": d.get("macro_f1"),
            "schema_vocab_conformance_pct": d.get("schema_vocabulary_conformance_pct"),
            "canonical_tool_mapping_pct": d.get("canonical_tool_mapping_conformance_pct"),
            "canonical_target_mapping_pct": d.get("canonical_target_mapping_conformance_pct"),
            "proxy_phase_consistency_pct": d.get("proxy_phase_transition_consistency_pct"),
            "temporal_coherence": d.get("temporal_coherence"),
            "validated_rule_status": d.get("validated_rule_status"),
            "source_file": path.name,
        })
    return pd.DataFrame(rows)


@st.cache_data
def build_decoder_effects_table() -> pd.DataFrame:
    rows = []
    for path in sorted(REPORTS_DIR.glob("decoder_effects_*.json")):
        d = json.loads(path.read_text())
        rows.append({
            "raw_model": d.get("raw_model"),
            "decoded_model": d.get("decoded_model"),
            "n_segments": d.get("n_segments"),
            "segments_changed": d.get("segments_changed"),
            "decisions_changed": d.get("action_decisions_changed"),
            "correction_precision": d.get("correction_precision"),
            "correction_harm": d.get("correction_harm"),
            "raw_macro_f1": d.get("raw_macro_f1"),
            "decoded_macro_f1": d.get("decoded_macro_f1"),
            "delta_macro_f1": d.get("delta_macro_f1"),
        })
    return pd.DataFrame(rows)


@st.cache_data
def list_val_segment_ids() -> list[str]:
    return sorted(p.stem for p in (ANNOTATIONS_DIR / "val").glob("*/*.json"))


@st.cache_data
def list_models() -> list[str]:
    return sorted(p.name for p in VLM_OUTPUTS_DIR.iterdir() if p.is_dir())


def video_from_segment_id(segment_id: str) -> str:
    return segment_id.split("_seg")[0]


@st.cache_data
def find_prediction(model: str, segment_id: str) -> dict | None:
    video = video_from_segment_id(segment_id)
    candidates = [
        VLM_OUTPUTS_DIR / model / "val" / video / f"{segment_id}.json",
        VLM_OUTPUTS_DIR / model / video / f"{segment_id}.json",
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    # fall back to a search, in case of an unusual layout
    matches = list((VLM_OUTPUTS_DIR / model).glob(f"**/{segment_id}.json"))
    return json.loads(matches[0].read_text()) if matches else None


def frame_image_path(video: str, frame_num: int) -> Path:
    return MESAD_ROOT / "val" / "images" / f"{video}_frame_{frame_num}.jpg"


def node_badges(nodes: list[str], gt_set: set[str] | None = None) -> str:
    """Render a list of nodes as inline badges, green/red against gt_set if given."""
    if not nodes:
        return "_none_"
    parts = []
    for n in nodes:
        if gt_set is None:
            color = "#5b8def"
        else:
            color = "#2e7d32" if n in gt_set else "#c62828"
        parts.append(
            f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;'
            f'margin:2px;display:inline-block;font-size:0.85em">{n}</span>'
        )
    return " ".join(parts)



st.title("OG-SRB Visual Reasoning Explorer")

page = st.sidebar.radio("View", ["Model Comparison", "Segment Explorer"])

if page == "Model Comparison":
    st.header("Per-model summary metrics")
    metrics_df = build_metrics_table()
    if metrics_df.empty:
        st.warning(f"No metrics_*.json files found in {REPORTS_DIR}")
    else:
        scopes = sorted(metrics_df["scope"].dropna().unique())
        scope_filter = st.multiselect("Filter by evaluation scope", scopes, default=scopes)
        shown = metrics_df[metrics_df["scope"].isin(scope_filter)].drop(columns=["source_file"])
        st.dataframe(
            shown.sort_values("macro_f1", ascending=False, na_position="last"),
            width="stretch", hide_index=True,
        )

        st.subheader("Top-1 / Macro-F1 comparison")
        chartable = shown.dropna(subset=["top1_accuracy", "macro_f1"]).set_index("model")
        if not chartable.empty:
            st.bar_chart(chartable[["top1_accuracy", "macro_f1"]])

    st.header("Decoder effects (raw vs. decoded, per model)")
    effects_df = build_decoder_effects_table()
    if effects_df.empty:
        st.info("No decoder_effects_*.json files found.")
    else:
        st.dataframe(
            effects_df.sort_values("correction_precision", ascending=False, na_position="last"),
            width="stretch", hide_index=True,
        )

    with st.expander("Per-class F1 for one model"):
        options = {f"{r.model} ({r.scope})": r.source_file for r in metrics_df.itertuples()}
        if options:
            choice = st.selectbox("Model", list(options))
            d = load_json(str(REPORTS_DIR / options[choice]))
            per_class = d.get("per_class_f1", {})
            support = d.get("per_class_support", {})
            if per_class:
                cdf = pd.DataFrame({
                    "action": list(per_class.keys()),
                    "f1": list(per_class.values()),
                    "support": [support.get(k, 0) for k in per_class],
                }).sort_values("f1", ascending=False)
                st.dataframe(cdf, width="stretch", hide_index=True)

else:  # Segment Explorer
    onto = get_ontology()
    seg_ids = list_val_segment_ids()
    models = list_models()

    st.sidebar.subheader("Segment")
    search = st.sidebar.text_input("Search segment_id (substring)", "")
    filtered = [s for s in seg_ids if search in s] if search else seg_ids
    if not filtered:
        st.sidebar.warning("No match.")
        st.stop()
    segment_id = st.sidebar.selectbox(f"Segment ({len(filtered)} matches)", filtered)

    video = video_from_segment_id(segment_id)
    gt_path = ANNOTATIONS_DIR / "val" / video / f"{segment_id}.json"
    gt = load_json(str(gt_path))

    st.sidebar.subheader("Models to compare")
    selected_models = st.sidebar.multiselect("Models", models, default=models[:2] if len(models) >= 2 else models)

    col_img, col_gt = st.columns([1, 1])
    with col_img:
        frame_num = gt["frames"][len(gt["frames"]) // 2]
        img_path = frame_image_path(video, frame_num)
        if img_path.exists():
            st.image(str(img_path), caption=f"{video}, frame {frame_num}")
        else:
            st.warning(f"Frame image not found: {img_path}")

    with col_gt:
        st.subheader("Ground truth")
        st.markdown(f"**Phase**: {gt.get('phase') or '_none_'}")
        st.markdown("**Actions**: " + node_badges(gt["actions"]), unsafe_allow_html=True)
        st.markdown("**Tools**: " + node_badges(gt["tools"]), unsafe_allow_html=True)
        st.markdown("**Tissues**: " + node_badges(gt["tissues"]), unsafe_allow_html=True)
        st.markdown("**Events**: " + node_badges(gt["events"]), unsafe_allow_html=True)

    st.divider()
    st.subheader("Model predictions")

    gt_actions = set(gt["actions"])
    for model in selected_models:
        pred = find_prediction(model, segment_id)
        with st.expander(f"**{model}**", expanded=True):
            if pred is None:
                st.info("No prediction found for this segment.")
                continue

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"Phase: **{pred.get('phase') or '_none_'}**"
                            + ("  ⚠️ ambiguous" if pred.get("phase_ambiguous") else ""))
                st.markdown("Actions: " + node_badges(pred["actions"], gt_actions), unsafe_allow_html=True)
                st.markdown("Tools: " + node_badges(pred.get("tools", [])), unsafe_allow_html=True)
                st.markdown("Tissues: " + node_badges(pred.get("tissues", [])), unsafe_allow_html=True)
                if pred.get("raw_actions") is not None:
                    st.caption(f"Before decoding: {', '.join(pred['raw_actions']) or 'none'}")

            with c2:
                req_violations = check_requires(pred, onto)
                acts_violations = check_acts_on(pred, onto)
                contra_violations = check_contradicts(pred, onto)
                st.markdown("**Live validator checks**")
                st.markdown(f"- Requires (tool): {'✅' if not req_violations else '❌ ' + '; '.join(req_violations)}")
                st.markdown(f"- Acts-on (target): {'✅' if not acts_violations else '❌ ' + '; '.join(acts_violations)}")
                st.markdown(f"- Contradicts: {'✅ (no evaluable rules)' if not contra_violations else '❌ ' + '; '.join(contra_violations)}")

            if pred.get("raw_response"):
                st.caption(f"Raw model response: _{pred['raw_response'][:300]}_")
