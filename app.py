import sys
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utilities.src import (
    remove_short_true_segments,
    fill_short_false_gaps,
    extract_idle_periods,
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "notebooks/models/xgb_repetition_quality.json"
LE_PATH = "notebooks/models/label_encoder.pkl"
FEAT_COLS_PATH = "notebooks/models/feature_columns.pkl"
SIMULATION_SUBJECT = "s3"
CLINICIAN_EMAIL = "stefanoanthony.rizzuto01@universitadipavia.it"
ACTIVE_COLS = ["acc_mag_active", "gyr_mag_active"]
LABELS_ORDER = ["correct", "fast", "low_amplitude"]
N_REPS_PER_BLOCK = 10

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=6,
    gamma=0.2,
    reg_lambda=4.0,
    reg_alpha=1.0,
    random_state=42,
    eval_metric="mlogloss",
)

# --------------------------------------------------
# Feature extraction — v13 (from XGBoost.ipynb)
# --------------------------------------------------

def extract_features_v13(segment, sensor_cols=ACTIVE_COLS):
    features = {}

    for col in sensor_cols:
        x = segment[col].to_numpy().astype(float)
        n = len(x)
        x_c = x - np.mean(x)

        features[f"{col}_mean"] = np.mean(x)
        features[f"{col}_std"] = np.std(x)
        features[f"{col}_median"] = np.median(x)

        features[f"{col}_range"] = float(np.max(x) - np.min(x))
        q25, q75 = np.percentile(x, [25, 75])
        features[f"{col}_iqr"] = float(q75 - q25)
        p10, p90 = np.percentile(x, [10, 90])
        features[f"{col}_spread"] = float(p90 - p10)
        features[f"{col}_rms"] = float(np.sqrt(np.mean(x ** 2)))

        features[f"{col}_skew"] = float(skew(x))
        features[f"{col}_kurtosis"] = float(kurtosis(x))

        mid = n // 2
        first, second = x[:mid], x[mid:]
        rms_first = float(np.sqrt(np.mean(first ** 2))) if len(first) > 0 else 0.0
        rms_second = float(np.sqrt(np.mean(second ** 2))) if len(second) > 0 else 0.0
        features[f"{col}_energy_diff"] = rms_first - rms_second

        fft_mag = np.abs(np.fft.rfft(x_c))
        power = fft_mag ** 2
        total = power.sum()

        if total > 0 and len(power) > 1:
            pn = power / total
            dom_idx = int(np.argmax(power[1:]) + 1)
            features[f"{col}_dom_freq_idx"] = dom_idx
            features[f"{col}_dom_freq_power"] = float(power[dom_idx])
            features[f"{col}_dom_power_ratio"] = float(power[dom_idx] / total)
            features[f"{col}_spectral_entropy"] = float(-np.sum(pn * np.log(pn + 1e-12)))
            freqs = np.arange(len(power))
            features[f"{col}_spectral_centroid"] = float(np.sum(freqs * power) / (total + 1e-12))
            lo = np.sum(power[1:4]) if len(power) > 4 else np.sum(power[1:])
            hi = np.sum(power[4:]) if len(power) > 4 else 0.0
            features[f"{col}_band_ratio"] = float(hi / (lo + 1e-12))
            features[f"{col}_zcr"] = float(np.sum(np.diff(np.sign(x_c)) != 0) / max(n - 1, 1))
        else:
            for k in ("dom_freq_idx", "dom_freq_power", "dom_power_ratio",
                      "spectral_entropy", "spectral_centroid", "band_ratio", "zcr"):
                features[f"{col}_{k}"] = 0.0

    acc = segment[sensor_cols[0]].to_numpy().astype(float)
    gyr = segment[sensor_cols[1]].to_numpy().astype(float)
    features["acc_gyr_corr"] = float(np.corrcoef(acc, gyr)[0, 1]) if len(acc) > 1 else 0.0

    return features


def build_dataset(df, n_reps=N_REPS_PER_BLOCK):
    rows = []
    for (s, e, l), group in df.groupby(["subject", "exercise", "label"]):
        group = group.sort_values("time index").reset_index(drop=True)
        unit = group["most_active_unit"].iloc[0]
        raw_cols = [f"acc_mag_{unit}", f"gyr_mag_{unit}"]
        rep_df = group[raw_cols].rename(columns=dict(zip(raw_cols, ACTIVE_COLS)))
        n_ts = len(rep_df)
        cuts = np.linspace(0, n_ts, n_reps + 1, dtype=int)
        for i in range(n_reps):
            rep = rep_df.iloc[cuts[i]:cuts[i + 1]]
            if len(rep) == 0:
                continue
            feats = extract_features_v13(rep)
            feats.update(subject=s, exercise=e, label=l, rep_id=i)
            rows.append(feats)
    return pd.DataFrame(rows)


# --------------------------------------------------
# Model training — cached across reruns
# --------------------------------------------------

@st.cache_resource
def get_trained_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    le = joblib.load(LE_PATH)
    feature_cols = joblib.load(FEAT_COLS_PATH)
    return model, le, feature_cols


# --------------------------------------------------
# Repetition auto-detection
# --------------------------------------------------

def detect_repetitions(exercise_df):
    """
    Detect active exercise blocks via combined energy thresholding, then split
    each block into N_REPS_PER_BLOCK equal windows — matching the training pipeline.

    Returns (repetitions: list[DataFrame], n_blocks: int).
    """
    df = exercise_df.reset_index(drop=True)
    unit = df["most_active_unit"].iloc[0]

    acc_s = pd.Series(df[f"acc_mag_{unit}"].values)
    gyr_s = pd.Series(df[f"gyr_mag_{unit}"].values)

    energy = (
        acc_s.rolling(50, center=True).std() +
        gyr_s.rolling(50, center=True).std()
    ).bfill().ffill()
    energy_smooth = energy.rolling(30, center=True).median().bfill().ffill()

    low_val = energy_smooth.quantile(0.05)
    high_val = energy_smooth.quantile(0.60)
    threshold = low_val + 0.25 * (high_val - low_val)

    idle_mask = pd.Series(energy_smooth.values < threshold)
    idle_mask = remove_short_true_segments(idle_mask, min_len=50)
    idle_mask = fill_short_false_gaps(idle_mask, max_gap_len=100)
    idle_mask = remove_short_true_segments(idle_mask, min_len=50)

    active_mask = pd.Series(~idle_mask.values)
    active_mask = remove_short_true_segments(active_mask, min_len=150)

    block_periods = extract_idle_periods(active_mask)

    repetitions = []
    for start, end in block_periods:
        block = df.iloc[start:end + 1].reset_index(drop=True)
        n = len(block)
        cuts = np.linspace(0, n, N_REPS_PER_BLOCK + 1, dtype=int)
        for i in range(N_REPS_PER_BLOCK):
            rep = block.iloc[cuts[i]:cuts[i + 1]].reset_index(drop=True)
            if len(rep) > 0:
                repetitions.append(rep)

    return repetitions, len(block_periods)


# --------------------------------------------------
# Per-rep classification
# --------------------------------------------------

def classify_repetition(rep_df, exercise, model, le, feature_cols):
    unit = rep_df["most_active_unit"].iloc[0]
    renamed = rep_df.rename(columns={
        f"acc_mag_{unit}": "acc_mag_active",
        f"gyr_mag_{unit}": "gyr_mag_active",
    })

    feats = extract_features_v13(renamed)

    feat_row = {col: 0.0 for col in feature_cols}
    for k, v in feats.items():
        if k in feat_row:
            feat_row[k] = v
    ex_col = f"ex_{exercise}"
    if ex_col in feat_row:
        feat_row[ex_col] = 1.0

    X = pd.DataFrame([feat_row])[feature_cols]
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    label_str = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    return label_str, confidence


# --------------------------------------------------
# Feedback messages
# --------------------------------------------------

def get_feedback(label):
    label = str(label).lower()
    if label == "correct":
        return "Correct repetition. Good movement.", "success"
    elif label == "fast":
        return "Movement too fast. Please slow down.", "warning"
    elif label == "low_amplitude":
        return "Amplitude too low. Increase the range of motion.", "warning"
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
def load_data():
    return pd.read_csv(DATA_PATH)


# ==================================================
# APP
# ==================================================

st.set_page_config(page_title="Physical Therapy Exercise Tracker", layout="centered")
st.title("Physical Therapy Exercise Tracker")

model, le, feature_cols = get_trained_model()

df = load_data()

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

    exercise_df = df[
        (df["exercise"] == selected_exercise) &
        (df["subject"] == SIMULATION_SUBJECT)
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
        f"Auto-detected **{n_blocks} exercise block(s)** — "
        f"**{len(repetitions)} repetitions** to classify."
    )

    # --------------------------------------------------
    # Step 3 — Live feedback
    # --------------------------------------------------

    st.header("Live Feedback")

    feedback_box = st.empty()
    progress_bar = st.progress(0)
    session_table_box = st.empty()

    session_results = []
    raw_frames = []

    for idx, rep_df in enumerate(repetitions, start=1):
        label, conf = classify_repetition(rep_df, selected_exercise, model, le, feature_cols)
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

        line = f"Rep {idx}: {feedback_text}  —  {conf * 100:.1f}% confidence"
        if severity == "success":
            feedback_box.success(line)
        elif severity == "warning":
            feedback_box.warning(line)
        else:
            feedback_box.info(line)

        session_table_box.dataframe(pd.DataFrame(session_results), use_container_width=True)
        progress_bar.progress(idx / len(repetitions))
        time.sleep(0.3)

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
