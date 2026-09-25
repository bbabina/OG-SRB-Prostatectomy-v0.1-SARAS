# Ontology-Grounded Surgical Reasoning Benchmark for Robotic-Assisted Radical Prostatectomy

MSc dissertation project testing whether grounding a model's surgical action predictions in an explicit ontology of tools, targets and phase order via an **Ontology-Grounded Structured Decoder (OGSD)**  can correct recognition errors without introducing new ones.

## Central finding

OGSD's effect is **conditional**, not universally beneficial. The same decoder, applied identically to five perception models, improved the trained probes but measurably **harmed** zero-shot CLIP and a GPT-4o ensemble. Two independent controlled interventions established why: it's the **calibration of a model's phase evidence**, not its raw accuracy,  that determines whether OGSD helps or harms.
<img width="1460" height="512" alt="image" src="https://github.com/user-attachments/assets/6167ccd3-c154-47da-a4f6-92abc510d370" />



Calibrating a model's phase evidence, leaving its raw predictions completely untouched, was enough on its own to flip the outcome:

## Calibration effect


| Model | Correction Precision | Correction Harm |
|---|---:|---:|
| Zero-shot CLIP (uncalibrated) | 64.8% | 35.2% |
| **Zero-shot CLIP + calibrated phase** | **99.2%** | **0.8%** |
| GPT-4o + probe ensemble (uncalibrated) | 74.5% | 25.5% |
| **GPT-4o + probe ensemble + calibrated phase** | **95.3%** | **4.7%** |
| Linear probe (trained) | 94.3% | 5.7% |
| Padded-crop probe (best raw model) | 95.4% | 4.6% |

*Correction precision = how often OGSD's changes to a model's prediction were actually right. Correction harm = how often they made it worse. Confirmed twice, independently, on two structurally different models.*

## How it works

### Pipeline
 <img width="2278" height="562" alt="image" src="https://github.com/user-attachments/assets/591e2f5c-1f24-4bbf-ae3d-d539d15111fb" />


A base model produces raw action probabilities; OGSD combines these with the ontology's rules and the model's phase evidence to decode a corrected prediction. Whether that correction is trustworthy depends entirely on the calibration of the phase evidence, not on which base model is used:

# Localisation

The padded-crop probe above ("best raw model") works by cropping to the tool-tissue interaction, with some padding kept for surrounding context, rather than encoding the whole frame:

<img width="1692" height="480" alt="image" src="https://github.com/user-attachments/assets/12823301-66f6-47a7-b152-961fc7006615" />


This uses the dataset's ground-truth bounding boxes, so it's a best-case ceiling for what localisation-aware recognition can achieve, not a number a real-time detector-based system gets for free.

## Structure

```
ontology/       domain ontology (v0.2: 21 actions, 6 tools, 11 targets, 7 phases)
data_prep/      segmentation, ground-truth mapping, QC validation
baselines/      CLIP, LLaVA-7B, GPT-4o/5, Gemini, trained probes, the OGSD decoder
eval/           metrics, evaluator, leave-one-video-out cross-validation
streamlit_app.py  interactive model comparison / segment explorer
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision open_clip_torch numpy pandas scikit-learn pillow streamlit openai google-genai ollama
```

Requires the SARAS-MESAD-Real dataset (four annotated RARP procedures), not included here.

## Run

```bash
python data_prep/build_segments.py
python data_prep/validate_annotations.py
python baselines/clip_linear_probe.py
python baselines/decoder_constrained.py
python eval/evaluator.py
streamlit run streamlit_app.py
```

## Status

Research benchmark, not a deployable clinical system. Single-site data (4 procedures); the padded-crop probe's raw result uses oracle bounding-box localisation. 

## Author

Babina Banjara, MSc, Oxford Brookes University 
