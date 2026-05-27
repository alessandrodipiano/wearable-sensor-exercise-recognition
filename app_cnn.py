import sys
from pathlib import Path

<<<<<<< HEAD
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from python import phyphox_connection
from python.phyphox_connection import test_connection
from python.model_inference import build_model, run_inference, IDX_TO_LABEL
from python.rep_detection import record_repetitions
=======
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utilities.model import CNNMLPModel
from utilities.src import detect_peaks_and_valleys_clean
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800

# --------------------------------------------------
# Constants
# --------------------------------------------------

<<<<<<< HEAD
EXERCISES_ORDER = [f"e{i}" for i in range(1, 9)]
ARM_EXERCISES = {"e2", "e6", "e7", "e8"}

# Which live unit the worn phone/sensor represents (see model_inference).
SENSOR_UNIT = "u2"

STYLES = {
    "correct": {
        "bg":       "#FFFFFF",
        "accent":   "#16A34A",
=======
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
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
        "title":    "PERFECT FORM",
        "feedback": "Perfect! Well executed.",
    },
    "fast": {
<<<<<<< HEAD
        "bg":       "#FFFFFF",
        "accent":   "#D97706",
=======
        "bg":       "#FFF4E6",
        "accent":   "#E08E2A",
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
        "title":    "TOO FAST",
        "feedback": "Too fast! Slow down your movements.",
    },
    "low_amplitude": {
<<<<<<< HEAD
        "bg":       "#FFFFFF",
        "accent":   "#DC2626",
        "title":    "LOW AMPLITUDE",
        "feedback": "Range of motion too small — push through the full movement.",
=======
        "bg":       "#FCE8E6",
        "accent":   "#D14343",
        "title":    "LOW AMPLITUDE",
        "feedback": "Completely wrong you fat bastard!",
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
    },
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

<<<<<<< HEAD
def exercise_label(e):
    limb = "Arm" if e in ARM_EXERCISES else "Leg"
    return f"{e.upper()} · {limb}"


@st.cache_resource
def get_model():
    return build_model()
=======
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
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800


# ==================================================
# APP
# ==================================================

st.set_page_config(page_title="PT Exercise Tracker", layout="centered")

st.markdown(
    """
    <style>
<<<<<<< HEAD
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {
          font-family: 'Inter', sans-serif !important;
      }

      .hero {
          padding: 24px 0 16px;
          border-top: 4px solid #2563EB;
          margin-bottom: 28px;
      }
      .hero h1 {
          font-size: 2rem;
          font-weight: 800;
          color: #1E293B;
          margin: 12px 0 0;
          letter-spacing: -0.03em;
      }
      .hero .subtitle {
          color: #64748B;
          font-size: 0.95rem;
=======
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
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
          margin-top: 6px;
      }
      .hero .badge {
          display: inline-block;
<<<<<<< HEAD
          background: #2563EB;
          color: #FFFFFF;
          padding: 3px 10px;
          border-radius: 6px;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
      }

      div[data-testid="stFileUploader"] section {
          border: 2px dashed #2563EB !important;
          background: #EFF6FF !important;
          border-radius: 12px;
          padding: 18px;
      }
      div[data-testid="stFileUploader"] section:hover {
          background: #DBEAFE !important;
      }

      .stats-card {
          background: linear-gradient(135deg, #FFFFFF 0%, #F0F5FF 100%);
          border-left: 4px solid #2563EB;
          border-radius: 12px;
          padding: 18px 22px;
          margin-bottom: 20px;
          box-shadow: 0 1px 4px rgba(37,99,235,0.10);
      }
      .stats-card .n {
          font-size: 2.4rem;
          font-weight: 700;
          color: #2563EB;
          line-height: 1;
      }
      .stats-card .lbl {
          color: #64748B;
=======
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
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
          font-size: 0.9rem;
          margin-top: 4px;
      }

      .section-heading {
<<<<<<< HEAD
          font-size: 0.7rem;
          font-weight: 700;
          color: #94A3B8;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin: 28px 0 12px;
      }

      label[data-testid="stWidgetLabel"] p {
          color: #64748B;
=======
          font-size: 1.15rem;
          font-weight: 700;
          color: #2C3E50;
          margin: 24px 0 10px;
      }

      label[data-testid="stWidgetLabel"] p {
          color: #6B7C8C;
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
          font-weight: 600;
          letter-spacing: 0.04em;
          font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
<<<<<<< HEAD
    """
=======
    f"""
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
    <div class="hero">
      <h1>Physical Therapy Exercise Tracker</h1>
      <div class="subtitle">
        Movement-quality assessment from wearable sensor data &nbsp;·&nbsp;
<<<<<<< HEAD
        <span class="badge">LIVE</span>
=======
        <span class="badge">SUBJECT {SIMULATION_SUBJECT.upper()}</span>
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

<<<<<<< HEAD

# --------------------------------------------------
# Step 1 — Patient profile form
# --------------------------------------------------

if "profile" not in st.session_state:
    st.markdown('<div class="section-heading">Patient profile</div>', unsafe_allow_html=True)

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name")
        with c2:
            surname = st.text_input("Surname")

        c3, c4 = st.columns(2)
        with c3:
            gender_label = st.selectbox("Gender", ["Female", "Male"])
        with c4:
            age = st.number_input("Age", min_value=1, max_value=120, value=22, step=1)

        c5, c6 = st.columns(2)
        with c5:
            weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=60, step=1)
        with c6:
            height = st.number_input("Height (cm)", min_value=30, max_value=250, value=175, step=1)

        exercise = st.selectbox(
            "Exercise",
            options=EXERCISES_ORDER,
            format_func=exercise_label,
        )

        submitted = st.form_submit_button("Save profile", type="primary")

    if submitted:
        if not name.strip() or not surname.strip():
            st.error("Please enter both Name and Surname.")
            st.stop()

        st.session_state["profile"] = {
            "name": name.strip(),
            "surname": surname.strip(),
            "gender_label": gender_label,
            "gender": 1 if gender_label == "Male" else 0,
            "age": int(age),
            "weight": int(weight),
            "height": int(height),
            "exercise": exercise,
            # URL is set separately in Step 2 so it can be changed without re-entering the profile
            "phyphox_url": st.session_state.get("phyphox_url", phyphox_connection.BASE_URL),
        }
        st.rerun()

    st.stop()


# --------------------------------------------------
# Step 2 — Profile summary + recording
# --------------------------------------------------

profile = st.session_state["profile"]
=======
uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
if uploaded is None:
    st.stop()

with st.spinner("Detecting repetitions and building dataset…"):
    X_seq, X_glob, meta_df = build_peak_dataset(uploaded)

if len(meta_df) == 0:
    st.error(f"No reps detected for subject {SIMULATION_SUBJECT} in this CSV.")
    st.stop()
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800

st.markdown(
    f"""
    <div class="stats-card">
<<<<<<< HEAD
      <div class="n">{profile['name']} {profile['surname']}</div>
      <div class="lbl">
        {profile['gender_label']} · {profile['age']} yrs ·
        {profile['weight']:g} kg · {profile['height']:g} cm ·
        Exercise <strong>{exercise_label(profile['exercise'])}</strong>
=======
      <div style="display: flex; gap: 28px; align-items: baseline;">
        <div>
          <div class="n">{len(meta_df)}</div>
          <div class="lbl">repetitions detected</div>
        </div>
        <div>
          <div class="n">{meta_df['exercise'].nunique()}</div>
          <div class="lbl">exercises covered</div>
        </div>
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

<<<<<<< HEAD
phyphox_url = st.text_input(
    "phyphox URL",
    value=st.session_state.get("phyphox_url", profile["phyphox_url"]),
    help="From the phyphox app: enable 'Allow remote access' and copy the shown address.",
    key="phyphox_url_input",
)
# Keep the URL in session state so it survives reruns and profile edits
if phyphox_url.strip():
    st.session_state["phyphox_url"] = phyphox_url.strip()
    profile["phyphox_url"] = phyphox_url.strip()

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    start = st.button("▶ Start recording", type="primary", use_container_width=True)
with col_b:
    test_btn = st.button("🔌 Test connection", use_container_width=True)
with col_c:
    if st.button("Edit profile", use_container_width=True):
        del st.session_state["profile"]
        st.session_state.pop("results", None)
        st.session_state.pop("conn_status", None)
        st.rerun()

# Test-connection action
if test_btn:
    phyphox_connection.BASE_URL = profile["phyphox_url"]
    with st.spinner("Contacting phyphox…"):
        ok, latency, err = test_connection(url=profile["phyphox_url"])
    st.session_state["conn_status"] = {"ok": ok, "latency": latency, "err": err}

# Persist connection status badge across reruns
if "conn_status" in st.session_state:
    cs = st.session_state["conn_status"]
    if cs["ok"]:
        st.success(f"✅ phyphox reachable — {cs['latency']} ms")
    else:
        st.error(f"❌ Cannot reach phyphox at `{profile['phyphox_url']}`")
        with st.expander("Troubleshooting checklist"):
            st.markdown(
                """
1. **Enable remote access in phyphox**: open the experiment → tap ⋮ → *Allow remote access* must be **ON**.
2. **Same WiFi network**: phone and laptop must be on the **same** network — not mobile data or a different hotspot.
3. **Copy the URL exactly**: use the address shown in the phyphox remote-access screen (e.g. `http://172.x.x.x:8080`).
4. **Windows Firewall**: port 8080 may be blocked — try temporarily disabling the firewall or creating an inbound rule for port 8080.
5. **Retry**: once the above are confirmed, press **🔌 Test connection** again before recording.
                """
            )
        st.caption(f"Technical detail: {cs['err']}")

if start:
    # Apply the URL from the form so the live connection targets the right device.
    phyphox_connection.BASE_URL = profile["phyphox_url"]

    status = st.empty()
    try:
        with st.spinner("Connecting to phyphox…"):
            repetitions, original_lengths, _, fig = record_repetitions(
                progress=lambda msg: status.info(msg),
            )
    except Exception as exc:  # noqa: BLE001 — surface any connection/recording failure to the user
        status.empty()
        st.error(
            f"Could not record from phyphox at {profile['phyphox_url']}.\n\n"
            f"Check the phone is on the same network and the URL is correct.\n\n"
            f"Details: {exc}"
        )
        st.stop()

    status.empty()

    if len(original_lengths) == 0:
        st.error("No repetitions were detected in the recording. Try again.")
        st.stop()

    model, device = get_model()
    preds, probs = run_inference(
        repetitions,
        original_lengths,
        gender=profile["gender"],
        age=profile["age"],
        weight=profile["weight"],
        height=profile["height"],
        exercise=profile["exercise"],
        unit=SENSOR_UNIT,
        model=model,
        device=device,
    )

    st.session_state["results"] = {"preds": preds, "probs": probs, "fig": fig}


# --------------------------------------------------
# Step 3 — Results
# --------------------------------------------------

CARD_HTML = """
<div style="
    background: #FFFFFF;
    border-left: 5px solid {accent};
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
">
  <div style="font-size: 0.65rem; color: #94A3B8; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
    Repetition {rep_num}
  </div>
  <div style="font-size: 1.25rem; font-weight: 800; color: {accent}; margin-top: 4px;">
    {title}
  </div>
  <div style="color: #475569; margin-top: 4px; font-size: 0.9rem;">
    {feedback}
  </div>
  <div style="font-size: 0.65rem; color: #94A3B8; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; margin-top: 14px;">
    Confidence
  </div>
  <div style="margin-top: 6px; display: flex; align-items: center; gap: 12px;">
    <div style="flex: 1; height: 10px; background: #F1F5F9; border-radius: 6px; overflow: hidden;">
      <div style="width: {pct:.1f}%; height: 100%; background: {accent}; border-radius: 6px;"></div>
    </div>
    <div style="font-weight: 700; color: #1E293B; min-width: 56px; text-align: right; font-size: 1rem;">
      {pct:.1f}%
    </div>
  </div>
</div>
"""

if "results" in st.session_state:
    preds = st.session_state["results"]["preds"]
    probs = st.session_state["results"]["probs"]
    n_reps = len(preds)

    st.markdown(
        f'<div class="section-heading">Results — {exercise_label(profile["exercise"])} '
        f'<span style="color:#6B7C8C; font-weight:500;">({n_reps} reps)</span></div>',
        unsafe_allow_html=True,
    )

    for k in range(n_reps):
        pred_label = IDX_TO_LABEL[int(preds[k])]
        style = STYLES[pred_label]
        st.markdown(
            CARD_HTML.format(
                bg=style["bg"],
                accent=style["accent"],
                title=style["title"],
                feedback=style["feedback"],
                rep_num=k + 1,
                pct=float(probs[k, preds[k]]) * 100,
            ),
            unsafe_allow_html=True,
        )

    fig = st.session_state["results"].get("fig")
    if fig is not None:
        with st.expander("📈 Peak detection graph"):
            st.pyplot(fig)
=======
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
>>>>>>> 09952647504d5f0ab85b9442594071faf7c76800
