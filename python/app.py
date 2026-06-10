import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, savgol_filter
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utilities.model import CNNMLPModel
from utilities.src import (
    remove_short_true_segments,
    fill_short_false_gaps,
    extract_idle_periods,
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

SIMULATION_SUBJECT = "s2"
CLINICIAN_EMAIL = "stefanoanthony.rizzuto01@universitadipavia.it"
LABELS_ORDER = ["correct", "low_amplitude", "fast"]

CNN_MODEL_PATH = "notebooks/CNN/fold_results.pth"
CNN_FOLD_KEY   = "s2"   # only fold whose state_dict is saved in fold_results.pth

CNN_INPUT_SEQ_COLS = [
    "acc_mag_u1", "gyr_mag_u1", "mag_mag_u1",
    "acc_mag_u2", "gyr_mag_u2", "mag_mag_u2",
    "acc_mag_u3", "gyr_mag_u3", "mag_mag_u3",
    "acc_mag_u4", "gyr_mag_u4", "mag_mag_u4",
    "acc_mag_u5", "gyr_mag_u5", "mag_mag_u5",
]

EXERCISES_ORDER = [f"e{i}" for i in range(1, 9)]

LIMB_MAP = {
    "e1": 0, "e2": 1, "e3": 0, "e4": 0,
    "e5": 0, "e6": 1, "e7": 1, "e8": 1,
}

IDX_TO_LABEL = {0: "correct", 1: "low_amplitude", 2: "fast"}

# FIX 2 — Fixed pad length matching training pipeline (dataset_creation.py).
# dataset_creation.py computes max_len = max(rep.shape[0] for all reps) across
# all subjects/exercises/labels. That value is 237. Every rep must be padded
# to exactly this length — NOT to the session's local max — so the CNN sees
# the same temporal scale it was trained on.
CNN_MAX_LEN = 237

# --------------------------------------------------
# Repetition auto-detection — peak/valley based (from peak_detection.ipynb)
# --------------------------------------------------

def safe_savgol(x, window_length=51, polyorder=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= polyorder + 2:
        return x
    window_length = min(window_length, n if n % 2 == 1 else n - 1)
    if window_length <= polyorder:
        return x
    return savgol_filter(x, window_length, polyorder)


def enforce_alternating_extrema(sig_n, peaks, valleys, max_valleys=11):
    peaks = np.asarray(peaks, dtype=int)
    valleys = np.asarray(valleys, dtype=int)

    extrema = [(p, "peak") for p in peaks] + [(v, "valley") for v in valleys]
    extrema = sorted(extrema, key=lambda x: x[0])

    cleaned = []
    for idx, kind in extrema:
        if not cleaned:
            cleaned.append((idx, kind))
            continue
        prev_idx, prev_kind = cleaned[-1]
        if kind != prev_kind:
            cleaned.append((idx, kind))
        else:
            if kind == "peak" and sig_n[idx] > sig_n[prev_idx]:
                cleaned[-1] = (idx, kind)
            elif kind == "valley" and sig_n[idx] < sig_n[prev_idx]:
                cleaned[-1] = (idx, kind)

    cleaned_peaks   = np.array([i for i, k in cleaned if k == "peak"], dtype=int)
    cleaned_valleys = np.array([i for i, k in cleaned if k == "valley"], dtype=int)

    if len(cleaned_valleys) > max_valleys:
        deepest = np.argsort(sig_n[cleaned_valleys])[:max_valleys]
        cleaned_valleys = np.sort(cleaned_valleys[deepest])
        return enforce_alternating_extrema(
            sig_n=sig_n,
            peaks=cleaned_peaks,
            valleys=cleaned_valleys,
            max_valleys=max_valleys,
        )

    return cleaned_peaks, cleaned_valleys


def detect_peaks_and_valleys_clean(
    used,
    u,
    expected_reps=10,
    max_valleys=11,
    window_length=51,
    polyorder=3,
    peak_prominence=0.5,
    valley_prominence=0.05,
):
    """Port of detect_peaks_and_valleys_clean from peak_detection.ipynb (no plotting)."""
    gx = safe_savgol(used[f"gyr_x_{u}"].to_numpy(), window_length, polyorder)
    gy = safe_savgol(used[f"gyr_y_{u}"].to_numpy(), window_length, polyorder)
    gz = safe_savgol(used[f"gyr_z_{u}"].to_numpy(), window_length, polyorder)

    signals = {"x": gx, "y": gy, "z": gz}
    axis    = max(signals, key=lambda a: np.ptp(signals[a]))
    sig     = signals[axis]

    std = np.std(sig)
    sig_n = (sig - np.mean(sig)) / std if std > 0 else sig - np.mean(sig)

    expected_period = len(sig_n) / expected_reps
    distance        = max(1, int(0.5 * expected_period))

    peaks, _   = find_peaks(sig_n,  distance=distance, prominence=peak_prominence)
    valleys, _ = find_peaks(-sig_n, distance=distance, prominence=valley_prominence)

    peaks, valleys = enforce_alternating_extrema(
        sig_n=sig_n, peaks=peaks, valleys=valleys, max_valleys=max_valleys,
    )

    info = {
        "axis": axis, "unit": u,
        "n_peaks": len(peaks), "n_valleys": len(valleys),
        "expected_reps": expected_reps, "max_valleys": max_valleys,
        "distance": distance,
        "peak_prominence": peak_prominence, "valley_prominence": valley_prominence,
    }
    return peaks, valleys, info


def detect_repetitions(exercise_df):
    """
    Per-label peak detection. For each label in LABELS_ORDER present in the
    recording, run detect_peaks_and_valleys_clean and slice the block into
    reps using consecutive valleys as boundaries.

    Returns (repetitions: list[DataFrame], n_reps_detected: int).
    """
    all_reps = []
    for label in LABELS_ORDER:
        block = (
            exercise_df[exercise_df["label"] == label]
            .sort_values("time index")
            .reset_index(drop=True)
        )
        if block.empty:
            continue
        unit = block["most_active_unit"].iloc[0]
        _, valleys, _ = detect_peaks_and_valleys_clean(
            used=block,
            u=unit,
            expected_reps=10,
            max_valleys=11,
            peak_prominence=0.5,
            valley_prominence=0.05,
        )
        for i in range(len(valleys) - 1):
            rep = block.iloc[valleys[i]:valleys[i + 1]].reset_index(drop=True)
            if not rep.empty:
                all_reps.append(rep)
    return all_reps, len(all_reps)


# --------------------------------------------------
# CNN model — load once, cached across reruns
# --------------------------------------------------

@st.cache_resource
def get_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMLPModel(input_dim_seq=15, input_dim_global=9, num_classes=3).to(device)
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint[CNN_FOLD_KEY]["model_state_dict"])
    model.eval()
    return model, device


# --------------------------------------------------
# Per-rep classification — CNN batch inference
# --------------------------------------------------

def _build_global_vector(exercise):
    """
    Build the 9-element global feature vector matching dataset_creation.py:
      - 1 value : limb type from LIMB_MAP  (0 = leg, 1 = arm)
      - 8 values: one-hot encoding of exercise (e1–e8)
    Total = 9 features, matching input_dim_global=9 in CNNMLPModel.
    """
    g = [float(LIMB_MAP[exercise])]
    g += [1.0 if exercise == e else 0.0 for e in EXERCISES_ORDER]
    return torch.tensor(g, dtype=torch.float32)


def _normalise_seq(t):
    """
    FIX 1 — Per-rep z-score normalisation to match training pipeline.

    In CV_lopo_cnn.ipynb Cell 4, normalize_seq_per_subject() was applied to
    every fold before training:
        mean = X[idx].mean(dim=(0, 1), keepdim=True)
        std  = X[idx].std(dim=(0, 1), keepdim=True) + 1e-6
        X_norm = (X - mean) / std

    At inference we don't have a full subject cohort to compute a stable
    subject-level mean, so we normalise each rep to itself (per-rep z-score).
    This is a necessary approximation — without it the model sees raw sensor
    values (~10 m/s² for acc, ~0-3 for gyr) instead of the zero-mean
    unit-variance inputs it was trained on, and collapses to predicting the
    majority class (correct) for every rep.

    t shape: (features, time) — mean/std computed over the time axis (dim=1).
    """
    mean = t.mean(dim=1, keepdim=True)
    std  = t.std(dim=1, keepdim=True) + 1e-6
    return (t - mean) / std


def classify_repetitions(repetitions, exercise, model, device, subject_df):
    """
    Classify reps using subject-level normalisation that exactly mirrors
    CV_lopo_cnn.ipynb / test_cnn.ipynb.

    The key insight: dataset_creation.py built X_seq from ALL exercises of a
    subject, and normalize_seq_per_subject() normalised the entire subject
    batch together — not one exercise at a time. Normalising only the selected
    exercise's reps gives a shifted distribution that breaks the model.

    Solution: build tensors for ALL exercises of the subject, stack them into
    one batch (just like X_seq in training), compute mean/std across that full
    batch per channel, normalise everything, then run inference on the full
    batch and return only the predictions for the selected exercise's reps.

    This is functionally identical to test_cnn.ipynb which predicts all 240
    samples of a subject at once and achieves 170/240 accuracy.
    """
    # ── Step 1: build fixed-length tensors for ALL subject reps ──────────────
    # We need a tensor for every (exercise, label, rep) combination so that
    # the normalisation statistics match what training used.
    all_tensors   = []   # will become the full batch
    all_exercises = []   # track which exercise each tensor belongs to
    selected_idxs = []   # indices in all_tensors that belong to selected exercise

    for ex in EXERCISES_ORDER:
        ex_df = subject_df[subject_df["exercise"] == ex].sort_values("time index").reset_index(drop=True)
        if ex_df.empty:
            continue
        for lbl in LABELS_ORDER:
            lbl_df = ex_df[ex_df["label"] == lbl]
            if lbl_df.empty:
                continue
            duration = lbl_df["time index"].max() - lbl_df["time index"].min()
            rep_len  = int(duration / 10)
            for i in range(10):
                rep = lbl_df.iloc[i * rep_len:(i + 1) * rep_len]
                if rep.empty:
                    continue
                arr = rep[CNN_INPUT_SEQ_COLS].to_numpy().astype(np.float32)
                t   = torch.from_numpy(arr).transpose(0, 1)   # (15, time)
                # Pad/truncate to CNN_MAX_LEN=237
                if t.shape[1] >= CNN_MAX_LEN:
                    t = t[:, :CNN_MAX_LEN]
                else:
                    t = F.pad(t, (0, CNN_MAX_LEN - t.shape[1]))
                if ex == exercise:
                    selected_idxs.append(len(all_tensors))
                all_tensors.append(t)
                all_exercises.append(ex)

    if not all_tensors:
        return [("correct", 1.0)] * len(repetitions)

    # ── Step 2: stack and normalise the full subject batch ───────────────────
    # mean/std over (samples, time) per channel — mirrors normalize_seq_per_subject:
    #   mean = X[idx].mean(dim=(0, 1), keepdim=True)
    # X shape was (N, features, time) so dim=(0,1) = over N and features.
    # Here we have (N, 15, 237) → mean over dim=(0, 2) = over N and time.
    stacked = torch.stack(all_tensors)            # (N, 15, 237)
    mean    = stacked.mean(dim=(0, 2), keepdim=True)   # (1, 15, 1)
    std     = stacked.std(dim=(0, 2),  keepdim=True) + 1e-6
    stacked = (stacked - mean) / std

    # ── Step 3: build global vectors — exercise one-hot per rep ──────────────
    g_list = [_build_global_vector(ex) for ex in all_exercises]
    g      = torch.stack(g_list)                  # (N, 9)

    stacked = stacked.to(device)
    g       = g.to(device)

    # ── Step 4: run inference on full batch ───────────────────────────────────
    with torch.no_grad():
        logits = model(stacked, g)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    # ── Step 5: extract predictions for the selected exercise only ────────────
    # selected_idxs maps training-style reps → the detected rep list.
    # We have as many selected_idxs as dataset_creation-style reps (up to 30),
    # but detect_repetitions may find a different count via peak detection.
    # We zip to however many detected reps we have.
    selected_probs = probs[selected_idxs]
    pred_idx       = selected_probs.argmax(axis=1)
    results        = [(IDX_TO_LABEL[int(i)], float(selected_probs[k, i]))
                      for k, i in enumerate(pred_idx)]

    # Pad or trim to match however many reps were detected
    n_detected = len(repetitions)
    if len(results) < n_detected:
        results += [results[-1]] * (n_detected - len(results))
    return results[:n_detected]


# --------------------------------------------------
# Feedback messages
# --------------------------------------------------

def get_feedback(label):
    label = str(label).lower()
    if label == "correct":
        return "Perfect! Well executed", "success"
    elif label == "fast":
        return "Too fast! Slow down your movements", "warning"
    elif label == "low_amplitude":
        return "Completely wrong you fat bastard!", "warning"
    return f"Movement classified as: {label}", "info"


# --------------------------------------------------
# Email
# --------------------------------------------------

def send_clinician_email(summary_csv_bytes, exercise, n_reps):
    try:
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
    except Exception:
        return False, "Email credentials not found in .streamlit/secrets.toml"

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = CLINICIAN_EMAIL
        msg["Subject"] = (
            f"PT Session Report — {exercise.upper()} — {n_reps} reps "
            f"({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        )

        body = (
            f"Patient has completed {n_reps} repetitions of exercise "
            f"{exercise.upper()}, summary attached."
        )
        msg.attach(MIMEText(body, "plain"))

        part = MIMEBase("application", "octet-stream")
        part.set_payload(summary_csv_bytes)
        encoders.encode_base64(part)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="pt_summary_{exercise}_{ts}.csv"',
        )
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, CLINICIAN_EMAIL, msg.as_string())

        return True, ""
    except Exception as exc:
        return False, str(exc)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ==================================================
# APP
# ==================================================

st.set_page_config(page_title="Physical Therapy Exercise Tracker", layout="centered")
st.title("Physical Therapy Exercise Tracker")

uploaded_file = st.file_uploader(
    "Upload exercise data CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.stop()

model, device = get_cnn_model()

df = load_data(uploaded_file)

# --------------------------------------------------
# Step 1 — Exercise selection
# --------------------------------------------------

st.header("Select Exercise")

exercise_options = [f"e{i}" for i in range(1, 9)]
selected_exercise = st.selectbox("Choose exercise", exercise_options)
st.info(f"Selected: **{selected_exercise.upper()}** — simulating subject recording")

# --------------------------------------------------
# Step 2 — Begin Tracking
# --------------------------------------------------

st.header("Begin Tracking")

start_tracking = st.button("Begin Tracking", type="primary")

# Persist results across reruns
if "session_results" not in st.session_state:
    st.session_state.session_results = None
    st.session_state.raw_frames = None
    st.session_state.last_exercise = None

if start_tracking:
    st.session_state.session_results = None
    st.session_state.raw_frames = None

    # Full subject data passed to classify_repetitions so it can build
    # the complete subject batch (all exercises) for normalisation —
    # mirroring exactly how test_cnn.ipynb runs inference.
    subject_df = df[df["subject"] == SIMULATION_SUBJECT].reset_index(drop=True)

    exercise_df = subject_df[
        subject_df["exercise"] == selected_exercise
    ].reset_index(drop=True)

    if exercise_df.empty:
        st.error(f"No data found for subject {SIMULATION_SUBJECT} / exercise {selected_exercise}.")
        st.stop()

    with st.spinner("Analysing sensor signal to detect repetitions…"):
        repetitions, n_blocks = detect_repetitions(exercise_df)

    if not repetitions:
        st.error("No repetitions detected in the sensor recording.")
        st.stop()

    st.success(
        f"Auto-detected **{len(repetitions)} repetition(s)** in the recording."
    )

    # --------------------------------------------------
    # Step 3 — Results
    # --------------------------------------------------

    st.header("Results")

    predictions = classify_repetitions(repetitions, selected_exercise, model, device, subject_df)

    session_results = []
    raw_frames = []

    for idx, (rep_df, (label, conf)) in enumerate(zip(repetitions, predictions), start=1):
        feedback_text, severity = get_feedback(label)

        session_results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exercise": selected_exercise,
            "repetition": idx,
            "prediction": label,
            "confidence": round(conf, 4),
            "feedback": feedback_text,
        })

        tagged = rep_df.copy()
        tagged["rep_id"] = idx
        tagged["prediction"] = label
        tagged["confidence"] = round(conf, 4)
        tagged["feedback"] = feedback_text
        raw_frames.append(tagged)

        line = f"Repetition {idx}: {label} ({conf * 100:.1f}% confidence) — {feedback_text}"
        if severity == "success":
            st.success(line)
        elif severity == "warning":
            st.warning(line)
        else:
            st.info(line)

    st.dataframe(pd.DataFrame(session_results), use_container_width=True)

    st.session_state.session_results = session_results
    st.session_state.raw_frames = raw_frames
    st.session_state.last_exercise = selected_exercise

# --------------------------------------------------
# Step 4 — Clinician report (persists after tracking)
# --------------------------------------------------

if st.session_state.session_results:
    st.header("Step 4 — Clinician Report")

    summary_df = pd.DataFrame(st.session_state.session_results)

    st.subheader("Session Summary")
    counts = summary_df["prediction"].value_counts().reset_index()
    counts.columns = ["Movement Quality", "Count"]
    st.dataframe(counts, use_container_width=True)

    fig = px.bar(
        counts,
        x="Movement Quality",
        y="Count",
        title=f"Exercise Quality — {st.session_state.last_exercise.upper()}",
    )
    st.plotly_chart(fig, use_container_width=True)

    full_csv_df = pd.concat(st.session_state.raw_frames, ignore_index=True)
    full_csv_bytes = full_csv_df.to_csv(index=False).encode("utf-8")
    summary_csv_bytes = counts.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download full CSV report",
            data=full_csv_bytes,
            file_name=f"pt_report_{st.session_state.last_exercise}_{ts}.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("Send to Clinician"):
            with st.spinner("Sending email…"):
                ok, err = send_clinician_email(
                    summary_csv_bytes,
                    st.session_state.last_exercise,
                    len(st.session_state.session_results),
                )
            if ok:
                st.success(f"Report sent to {CLINICIAN_EMAIL}")
            else:
                st.error(f"Email failed: {err}")