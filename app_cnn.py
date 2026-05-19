import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utilities.model import CNNMLPModel
from utilities.src import detect_peaks_and_valleys_clean

# --------------------------------------------------
# Constants
# --------------------------------------------------

# Which subject's reps to predict on. Free knob — change to "s1", "s3", "s4", "s5"
# at will. Only "s2" is a true held-out evaluation; the others were in training and
# will show inflated accuracy.
SIMULATION_SUBJECT = "s3"

CNN_MODEL_PATH = "notebooks/CNN/fold_results.pth"
# DO NOT couple this to SIMULATION_SUBJECT. fold_results.pth only contains the "s2"
# fold's weights — the LOPO loop in CV_lopo_cnn.ipynb never saved the other folds.
CNN_FOLD_KEY   = "s2"
CNN_MAX_LEN    = 237

CNN_INPUT_SEQ_COLS = [
    "acc_mag_u1", "gyr_mag_u1", "mag_mag_u1",
    "acc_mag_u2", "gyr_mag_u2", "mag_mag_u2",
    "acc_mag_u3", "gyr_mag_u3", "mag_mag_u3",
    "acc_mag_u4", "gyr_mag_u4", "mag_mag_u4",
    "acc_mag_u5", "gyr_mag_u5", "mag_mag_u5",
]
EXERCISES_ORDER = [f"e{i}" for i in range(1, 9)]
LABELS_ORDER    = ["correct", "low_amplitude", "fast"]
LIMB_MAP = {
    "e1": 0, "e2": 1, "e3": 0, "e4": 0,
    "e5": 0, "e6": 1, "e7": 1, "e8": 1,
}
IDX_TO_LABEL = {0: "correct", 1: "low_amplitude", 2: "fast"}

STYLES = {
    "correct": {
        "bg":       "#E8F5EE",
        "accent":   "#2D8A6B",
        "title":    "PERFECT FORM",
        "feedback": "Perfect! Well executed.",
    },
    "fast": {
        "bg":       "#FFF4E6",
        "accent":   "#E08E2A",
        "title":    "TOO FAST",
        "feedback": "Too fast! Slow down your movements.",
    },
    "low_amplitude": {
        "bg":       "#FCE8E6",
        "accent":   "#D14343",
        "title":    "LOW AMPLITUDE",
        "feedback": "Completely wrong you fat bastard!",
    },
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _build_global_vector(exercise):
    g = [float(LIMB_MAP[exercise])]
    g += [1.0 if exercise == e else 0.0 for e in EXERCISES_ORDER]
    return torch.tensor(g, dtype=torch.float32)


@st.cache_resource
def get_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMLPModel(input_dim_seq=15, input_dim_global=9, num_classes=3).to(device)
    ckpt = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt[CNN_FOLD_KEY]["model_state_dict"])
    model.eval()
    return model, device


@st.cache_data
def build_peak_dataset(file):
    """
    For every (exercise, label) block belonging to SIMULATION_SUBJECT in the
    uploaded CSV, detect rep boundaries via detect_peaks_and_valleys_clean and
    slice valley-to-valley. Each rep becomes a (15, CNN_MAX_LEN) tensor —
    zero-padded or truncated to match the CNN training shape.

    Returns (X_seq, X_glob, meta_df).
    """
    df = pd.read_csv(file)
    df = df[df["subject"] == SIMULATION_SUBJECT].reset_index(drop=True)

    seqs, globs, meta = [], [], []
    for exercise in EXERCISES_ORDER:
        for label in LABELS_ORDER:
            block = (
                df[(df["exercise"] == exercise) & (df["label"] == label)]
                .sort_values("time index")
                .reset_index(drop=True)
            )
            if block.empty:
                continue
            unit = block["most_active_unit"].iloc[0]
            _, valleys, _ = detect_peaks_and_valleys_clean(
                used=block, u=unit,
                expected_reps=10, max_valleys=11,
                peak_prominence=0.5, valley_prominence=0.05,
                plot=False,
            )
            for i in range(len(valleys) - 1):
                rep = block.iloc[valleys[i]:valleys[i + 1]]
                if rep.empty:
                    continue
                arr = rep[CNN_INPUT_SEQ_COLS].to_numpy().astype(np.float32)
                T = arr.shape[0]
                if T < CNN_MAX_LEN:
                    arr = np.concatenate(
                        [arr, np.zeros((CNN_MAX_LEN - T, 15), dtype=np.float32)],
                        axis=0,
                    )
                else:
                    arr = arr[:CNN_MAX_LEN]
                seqs.append(torch.from_numpy(arr).transpose(0, 1))
                globs.append(_build_global_vector(exercise))
                meta.append({"exercise": exercise, "label": label, "rep_id": i})

    if not seqs:
        return torch.empty(0, 15, CNN_MAX_LEN), torch.empty(0, 9), pd.DataFrame(columns=["exercise", "label", "rep_id"])

    X_seq   = torch.stack(seqs)
    X_glob  = torch.stack(globs)
    meta_df = pd.DataFrame(meta)
    return X_seq, X_glob, meta_df


# ==================================================
# APP
# ==================================================

st.set_page_config(page_title="PT Exercise Tracker", layout="centered")

st.markdown(
    """
    <style>
      .hero {
          padding: 28px 0 12px;
          border-bottom: 2px solid #E2EEEA;
          margin-bottom: 24px;
      }
      .hero h1 {
          font-size: 2.1rem;
          font-weight: 700;
          color: #2D8A6B;
          margin: 0;
          letter-spacing: -0.02em;
      }
      .hero .subtitle {
          color: #6B7C8C;
          font-size: 1rem;
          margin-top: 6px;
      }
      .hero .badge {
          display: inline-block;
          background: #E8F5EE;
          color: #2D8A6B;
          padding: 3px 10px;
          border-radius: 999px;
          font-size: 0.8rem;
          font-weight: 600;
          letter-spacing: 0.04em;
      }

      div[data-testid="stFileUploader"] section {
          border: 2px dashed #2D8A6B !important;
          background: #E8F5EE !important;
          border-radius: 14px;
          padding: 18px;
      }
      div[data-testid="stFileUploader"] section:hover {
          background: #DCEEE3 !important;
      }

      .stats-card {
          background: #FFFFFF;
          border-left: 6px solid #2D8A6B;
          border-radius: 12px;
          padding: 18px 22px;
          margin-bottom: 20px;
          box-shadow: 0 2px 10px rgba(45,138,107,0.10);
      }
      .stats-card .n {
          font-size: 2.2rem;
          font-weight: 700;
          color: #2D8A6B;
          line-height: 1;
      }
      .stats-card .lbl {
          color: #6B7C8C;
          font-size: 0.9rem;
          margin-top: 4px;
      }

      .section-heading {
          font-size: 1.15rem;
          font-weight: 700;
          color: #2C3E50;
          margin: 24px 0 10px;
      }

      label[data-testid="stWidgetLabel"] p {
          color: #6B7C8C;
          font-weight: 600;
          letter-spacing: 0.04em;
          font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1>Physical Therapy Exercise Tracker</h1>
      <div class="subtitle">
        Movement-quality assessment from wearable sensor data &nbsp;·&nbsp;
        <span class="badge">SUBJECT {SIMULATION_SUBJECT.upper()}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
if uploaded is None:
    st.stop()

with st.spinner("Detecting repetitions and building dataset…"):
    X_seq, X_glob, meta_df = build_peak_dataset(uploaded)

if len(meta_df) == 0:
    st.error(f"No reps detected for subject {SIMULATION_SUBJECT} in this CSV.")
    st.stop()

st.markdown(
    f"""
    <div class="stats-card">
      <div style="display: flex; gap: 28px; align-items: baseline;">
        <div>
          <div class="n">{len(meta_df)}</div>
          <div class="lbl">repetitions detected</div>
        </div>
        <div>
          <div class="n">{meta_df['exercise'].nunique()}</div>
          <div class="lbl">exercises covered</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Choose an exercise</div>', unsafe_allow_html=True)

available_exercises = [e for e in EXERCISES_ORDER if e in meta_df["exercise"].unique()]
exercise_labels = {e: f"{e.upper()} · {'Arm' if LIMB_MAP[e] else 'Leg'}" for e in available_exercises}

if hasattr(st, "pills"):
    exercise = st.pills(
        "Exercise",
        options=available_exercises,
        format_func=lambda e: exercise_labels[e],
        default=available_exercises[0],
        label_visibility="collapsed",
    )
else:
    exercise = st.selectbox(
        "Exercise",
        options=available_exercises,
        format_func=lambda e: exercise_labels[e],
        label_visibility="collapsed",
    )

if exercise is None:
    st.stop()

ex_indices = np.where(meta_df["exercise"].to_numpy() == exercise)[0]
n_pick     = min(10, len(ex_indices))
sample_idx = np.sort(np.random.choice(ex_indices, size=n_pick, replace=False))

actual_labels = meta_df.iloc[sample_idx]["label"].tolist()
n_reps        = len(sample_idx)

model, device = get_cnn_model()
with torch.no_grad():
    logits = model(X_seq[sample_idx].to(device), X_glob[sample_idx].to(device))
    probs  = torch.softmax(logits, dim=1).cpu().numpy()
pred_idx = probs.argmax(axis=1)

st.markdown(
    f'<div class="section-heading">Results — {exercise.upper()} '
    f'<span style="color:#6B7C8C; font-weight:500;">'
    f'({n_reps} random reps of {len(ex_indices)})</span></div>',
    unsafe_allow_html=True,
)

CARD_HTML = """
<div style="
    background: {bg};
    border-left: 6px solid {accent};
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(45,138,107,0.06);
">
  <div style="font-size: 0.78rem; color: #6B7C8C; font-weight: 700; letter-spacing: 0.08em;">
    REPETITION {rep_num}
  </div>
  <div style="font-size: 1.35rem; font-weight: 700; color: {accent}; margin-top: 2px;">
    {title}
  </div>
  <div style="color: #4A5568; margin-top: 2px; font-size: 0.95rem;">
    {feedback}
  </div>
  <div style="margin-top: 12px; display: flex; align-items: center; gap: 12px;">
    <div style="flex: 1; height: 8px; background: #E2E8F0; border-radius: 4px; overflow: hidden;">
      <div style="width: {pct:.1f}%; height: 100%; background: {accent}; border-radius: 4px;"></div>
    </div>
    <div style="font-weight: 700; color: #2D3748; min-width: 60px; text-align: right;">
      {pct:.1f}%
    </div>
  </div>
  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.07); display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;">
    <span style="color: #6B7C8C;">Actual label:
      <span style="font-weight: 700; color: #2C3E50;">{actual}</span>
    </span>
    <span style="font-weight: 700; color: {mark_color};">{mark}</span>
  </div>
</div>
"""

for k in range(n_reps):
    pred_label   = IDX_TO_LABEL[int(pred_idx[k])]
    actual_label = actual_labels[k]
    is_match     = (pred_label == actual_label)
    style        = STYLES[pred_label]
    st.markdown(
        CARD_HTML.format(
            bg=style["bg"],
            accent=style["accent"],
            title=style["title"],
            feedback=style["feedback"],
            rep_num=k + 1,
            pct=probs[k, pred_idx[k]] * 100,
            actual=actual_label,
            mark="MATCH" if is_match else "MISMATCH",
            mark_color="#2D8A6B" if is_match else "#D14343",
        ),
        unsafe_allow_html=True,
    )
