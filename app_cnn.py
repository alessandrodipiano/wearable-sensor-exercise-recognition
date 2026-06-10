import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from python import phyphox_connection
from python.phyphox_connection import test_connection
from python.model_inference import build_model, run_inference, IDX_TO_LABEL
from python.rep_detection import record_repetitions
from python.send_report import send_report_background

# --------------------------------------------------
# Constants
# --------------------------------------------------

EXERCISES_ORDER = [f"e{i}" for i in range(1, 9)]
ARM_EXERCISES = {"e2", "e6", "e7", "e8"}

# Which live unit the worn phone/sensor represents (see model_inference).
SENSOR_UNIT = "u2"

STYLES = {
    "correct": {
        "bg":       "#FFFFFF",
        "accent":   "#16A34A",
        "title":    "PERFECT FORM",
        "feedback": "Perfect! Well executed.",
    },
    "fast": {
        "bg":       "#FFFFFF",
        "accent":   "#D97706",
        "title":    "TOO FAST",
        "feedback": "Too fast! Slow down your movements.",
    },
    "low_amplitude": {
        "bg":       "#FFFFFF",
        "accent":   "#DC2626",
        "title":    "LOW AMPLITUDE",
        "feedback": "Range of motion too small — push through the full movement.",
    },
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def exercise_label(e):
    limb = "Arm" if e in ARM_EXERCISES else "Leg"
    return f"{e.upper()} · {limb}"


@st.cache_resource
def get_model():
    return build_model()


# ==================================================
# APP
# ==================================================

st.set_page_config(page_title="PT Exercise Tracker", layout="centered")

st.markdown(
    """
    <style>
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
          margin-top: 6px;
      }
      .hero .badge {
          display: inline-block;
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
          font-size: 0.9rem;
          margin-top: 4px;
      }

      .section-heading {
          font-size: 0.7rem;
          font-weight: 700;
          color: #94A3B8;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin: 28px 0 12px;
      }

      label[data-testid="stWidgetLabel"] p {
          color: #64748B;
          font-weight: 600;
          letter-spacing: 0.04em;
          font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Physical Therapy Exercise Tracker</h1>
      <div class="subtitle">
        Movement-quality assessment from wearable sensor data &nbsp;·&nbsp;
        <span class="badge">LIVE</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
            gender_label = st.selectbox("Gender", ["Female", "Male"], index=None, placeholder="Select gender")
        with c4:
            age = st.number_input("Age", min_value=1, max_value=120, value=None, step=1)

        c5, c6 = st.columns(2)
        with c5:
            weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=None, step=1)
        with c6:
            height = st.number_input("Height (cm)", min_value=30, max_value=250, value=None, step=1)

        exercise = st.selectbox(
            "Exercise",
            options=EXERCISES_ORDER,
            format_func=exercise_label,
        )

        submitted = st.form_submit_button("Save profile", type="primary")

    if submitted:
        missing = []
        if not name.strip(): missing.append("Name")
        if not surname.strip(): missing.append("Surname")
        if not gender_label: missing.append("Gender")
        if age is None: missing.append("Age")
        if weight is None: missing.append("Weight")
        if height is None: missing.append("Height")
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}.")
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

st.markdown(
    f"""
    <div class="stats-card">
      <div class="n">{profile['name']} {profile['surname']}</div>
      <div class="lbl">
        {profile['gender_label']} · {profile['age']} yrs ·
        {profile['weight']:g} kg · {profile['height']:g} cm ·
        Exercise <strong>{exercise_label(profile['exercise'])}</strong>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

_cs = st.session_state.get("conn_status", {})
_conn_ok = bool(_cs.get("ok")) and (
    st.session_state.get("last_tested_url", "") == profile["phyphox_url"]
)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    start = st.button("▶ Start recording", type="primary", use_container_width=True, disabled=not _conn_ok)
    if not _conn_ok:
        st.caption("Test the connection successfully before recording.")
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
    st.session_state["last_tested_url"] = profile["phyphox_url"]
    st.rerun()

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

    pred_labels = [IDX_TO_LABEL[int(p)] for p in preds]
    send_report_background(profile, pred_labels)
    st.toast("Session report sent to clinician.", icon="📧")


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
