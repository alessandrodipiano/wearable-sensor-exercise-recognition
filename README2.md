# app.py — Code Walkthrough

This document explains every section of `app.py` in plain language.

---

## 1. Imports (lines 1–25)

```python
import sys, time, smtplib, joblib, numpy, pandas, plotly, streamlit, scipy, sklearn, xgboost
from utilities.src import remove_short_true_segments, fill_short_false_gaps, extract_idle_periods
```

Standard library and third-party packages are loaded first.

| Import | Purpose |
|---|---|
| `smtplib`, `email.*` | Sending the clinician email with a CSV attachment |
| `joblib` | Loading the saved LabelEncoder and feature column list from `.pkl` files |
| `numpy` / `pandas` | Numerical computation and data manipulation |
| `plotly.express` | Interactive bar chart in the clinician report |
| `streamlit` | The entire UI framework — every `st.*` call renders something in the browser |
| `scipy.stats.skew/kurtosis` | Distribution shape features used during feature extraction |
| `sklearn.preprocessing.LabelEncoder` | Converts numeric model output (0/1/2) back to text labels |
| `xgboost.XGBClassifier` | The classification model |
| `utilities.src` | Three helper functions for the repetition detector (see section 5) |

---

## 2. Constants (lines 31–55)

```python
DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "notebooks/models/xgb_repetition_quality.json"
LE_PATH    = "notebooks/models/label_encoder.pkl"
FEAT_COLS_PATH = "notebooks/models/feature_columns.pkl"
SIMULATION_SUBJECT = "s3"
CLINICIAN_EMAIL = "stefanoanthony.rizzuto01@universitadipavia.it"
ACTIVE_COLS = ["acc_mag_active", "gyr_mag_active"]
LABELS_ORDER = ["correct", "fast", "low_amplitude"]
N_REPS_PER_BLOCK = 10
XGB_PARAMS = { ... }
```

All hard-coded values live here so they are easy to find and change without touching logic.

| Constant | Meaning |
|---|---|
| `DATA_PATH` | The preprocessed sensor dataset (158 k rows, 65 columns) |
| `MODEL_PATH` | Saved XGBoost model in JSON format |
| `LE_PATH` | Saved `LabelEncoder` that maps class integers → label strings |
| `FEAT_COLS_PATH` | Saved ordered list of feature column names the model was trained on |
| `SIMULATION_SUBJECT` | The subject whose recording is played back (`s3`) |
| `CLINICIAN_EMAIL` | Fixed destination address for the emailed report |
| `ACTIVE_COLS` | Canonical names for the two sensor signals used for features: accelerometer magnitude and gyroscope magnitude of the most active unit |
| `LABELS_ORDER` | The three possible movement quality labels |
| `N_REPS_PER_BLOCK` | How many equal windows each detected exercise block is split into (10, matching the training pipeline) |
| `XGB_PARAMS` | Hyperparameters for the XGBoost model — kept here for reference even though the model is now loaded from disk |

---

## 3. `extract_features_v13` (lines 61–115)

```python
def extract_features_v13(segment, sensor_cols=ACTIVE_COLS) -> dict
```

Takes a short time-series segment (one repetition window) and returns a flat dictionary of ~35 numerical features. This is called once per repetition during inference, and was also used to build the training dataset.

The function loops over two sensor signals (`acc_mag_active`, `gyr_mag_active`) and computes:

### Baseline statistics
| Feature | Formula | Detects |
|---|---|---|
| `_mean` | arithmetic mean | average signal level |
| `_std` | standard deviation | overall variability |
| `_median` | 50th percentile | robust central value |

### Amplitude
| Feature | Formula | Detects |
|---|---|---|
| `_range` | max − min | peak-to-peak swing |
| `_iqr` | Q75 − Q25 | spread of the middle 50 % |
| `_spread` | P90 − P10 | spread excluding outliers |
| `_rms` | √mean(x²) | signal energy — low for low-amplitude reps |

### Distribution shape
| Feature | Formula | Detects |
|---|---|---|
| `_skew` | 3rd standardised moment | asymmetry — fast reps often accelerate then brake suddenly |
| `_kurtosis` | 4th moment − 3 | spiky / impulsive motion |

### Temporal asymmetry
| Feature | Formula | Detects |
|---|---|---|
| `_energy_diff` | RMS(first half) − RMS(second half) | whether effort is front- or back-loaded |

### Spectral (frequency-domain)
The signal is mean-subtracted (`x_c`) and transformed with `np.fft.rfft`. Power = magnitude².

| Feature | Detects |
|---|---|
| `_dom_freq_idx` | which frequency bin carries the most power |
| `_dom_freq_power` | raw power at that bin |
| `_dom_power_ratio` | how dominant that frequency is (rhythmicity) |
| `_spectral_entropy` | complexity — low for rhythmic, high for chaotic |
| `_spectral_centroid` | power-weighted average frequency (proxy for speed) |
| `_band_ratio` | high-frequency / low-frequency power ratio — high for fast reps |
| `_zcr` | zero-crossing rate — how often the signal crosses its mean |

### Cross-sensor correlation
```python
features["acc_gyr_corr"] = np.corrcoef(acc, gyr)[0, 1]
```
Pearson correlation between the accelerometer and gyroscope magnitudes. Captures whether linear and rotational motion are in sync — a quality indicator that differs across execution types.

**Total output: ~35 features** (17 per sensor × 2 sensors + 1 cross-sensor).

---

## 4. `build_dataset` (lines 118–134)

```python
def build_dataset(df, n_reps=10) -> pd.DataFrame
```

This function was used to build the training dataset and is kept in the file for reference. It is **not called at runtime** — the model is loaded from disk instead.

**What it does:**
1. Groups `processed_data.csv` by `(subject, exercise, label)` — one group per execution type per exercise per subject.
2. For each group, identifies the most active sensor unit.
3. Renames that unit's columns to `acc_mag_active` / `gyr_mag_active`.
4. Splits the group into `n_reps=10` equal time windows using `np.linspace`.
5. Calls `extract_features_v13` on each window.
6. Returns one row per window — the flat feature matrix used for training.

---

## 5. `get_trained_model` (lines 141–147)

```python
@st.cache_resource
def get_trained_model() -> (model, le, feature_cols)
```

Loads the three pre-saved model artefacts from disk. `@st.cache_resource` means this runs **once per app session** — Streamlit reuses the result on every subsequent page interaction without reloading from disk.

| File loaded | What it contains |
|---|---|
| `xgb_repetition_quality.json` | Trained XGBoost classifier (100 trees, 3 classes) |
| `label_encoder.pkl` | Maps integers 0/1/2 → `correct` / `fast` / `low_amplitude` |
| `feature_columns.pkl` | Ordered list of column names the model was trained on — needed to build the inference feature vector in the correct order |

---

## 6. `detect_repetitions` (lines 154–197)

```python
def detect_repetitions(exercise_df) -> (list[DataFrame], int)
```

The repetition auto-detection pipeline. Given a subject's full exercise recording, returns a list of DataFrames (one per detected repetition window) and the number of exercise blocks found.

### Step-by-step

**1. Compute combined energy**
```python
energy = rolling_std(acc_mag, window=50) + rolling_std(gyr_mag, window=50)
```
Rolling standard deviation measures local signal variability. High variability = movement happening. The two sensors are summed to get a single energy signal.

**2. Smooth**
```python
energy_smooth = energy.rolling(30).median()
```
A median filter removes short spikes, making the threshold step more stable.

**3. Adaptive threshold**
```python
threshold = q05 + 0.25 * (q60 - q05)
```
The threshold is computed from the signal's own distribution, so it adapts to each exercise's energy level rather than using a fixed value.

**4. Idle mask**
```python
idle_mask = energy_smooth < threshold
```
Timesteps below the threshold are labelled as "idle" (patient at rest). Three cleaning passes are applied:
- `remove_short_true_segments(min_len=50)` — removes brief resting labels that are just noise
- `fill_short_false_gaps(max_gap_len=100)` — merges nearby idle periods separated by a short active blip
- `remove_short_true_segments(min_len=50)` — second pass after gap-filling

**5. Active mask**
```python
active_mask = ~idle_mask
active_mask = remove_short_true_segments(min_len=150)
```
Invert the idle mask to get active (exercising) periods. Segments shorter than 150 timesteps are removed — these are noise bursts between the real exercise blocks.

**6. Extract block boundaries**
```python
block_periods = extract_idle_periods(active_mask)
```
Returns a list of `(start_index, end_index)` tuples, one per continuous active block. For most exercises this is 3 blocks (one for correct, one for fast, one for low-amplitude execution).

**7. Split each block into windows**
```python
cuts = np.linspace(0, block_length, N_REPS_PER_BLOCK + 1)
```
Each block is divided into 10 equal windows. This matches exactly how the training data was constructed in `build_dataset`, so the features the model sees at inference time have the same statistical distribution as at training time.

**Returns:** a flat list of 10 × n_blocks repetition DataFrames, plus the block count.

---

## 7. `classify_repetition` (lines 204–226)

```python
def classify_repetition(rep_df, exercise, model, le, feature_cols) -> (str, float)
```

Runs inference on a single repetition window. Returns the predicted label and the model's confidence.

### Steps

**1. Rename active unit columns**
```python
rep_df.rename({f"acc_mag_{unit}": "acc_mag_active", ...})
```
The raw data uses unit-specific column names (`acc_mag_u2`). These are renamed to the generic `acc_mag_active` / `gyr_mag_active` names that `extract_features_v13` expects.

**2. Extract features**
```python
feats = extract_features_v13(renamed)   # → dict of ~35 values
```

**3. Build the full feature vector**
```python
feat_row = {col: 0.0 for col in feature_cols}   # initialise all to 0
feat_row.update(feats)                            # fill in computed features
feat_row[f"ex_{exercise}"] = 1.0                 # one-hot encode the exercise
```
The model was trained with exercise as a one-hot encoded column (`ex_e1` … `ex_e8`). The current exercise gets a `1`, all others stay `0`. Any feature column that wasn't produced by the extractor defaults to `0.0`.

**4. Predict**
```python
proba = model.predict_proba(X)[0]    # shape (3,) — one probability per class
pred_idx = np.argmax(proba)
label_str = le.inverse_transform([pred_idx])[0]
confidence = proba[pred_idx]
```
`predict_proba` returns a probability for each of the three classes. The class with the highest probability wins. The confidence value is that winning probability (0–1).

---

## 8. `get_feedback` (lines 233–241)

```python
def get_feedback(label) -> (str, severity)
```

Simple lookup that converts a label string into a human-readable message and a Streamlit severity level (`"success"` renders green, `"warning"` renders yellow).

| Label | Message | Colour |
|---|---|---|
| `correct` | "Correct repetition. Good movement." | green |
| `fast` | "Movement too fast. Please slow down." | yellow |
| `low_amplitude` | "Amplitude too low. Increase the range of motion." | yellow |

---

## 9. `send_clinician_email` (lines 248–286)

```python
def send_clinician_email(summary_csv_bytes, exercise, n_reps) -> (bool, str)
```

Sends the session summary CSV to the clinician via Gmail SMTP.

### Steps

**1. Load credentials**
```python
sender   = st.secrets["email"]["sender"]
password = st.secrets["email"]["password"]
```
Credentials are read from `.streamlit/secrets.toml` — never hard-coded. If the file is missing or incomplete, returns `(False, error_message)` gracefully.

**2. Build the email**
- **Subject:** `PT Session Report — E1 — 30 reps (2026-04-27 14:32)`
- **Body:** `"Patient has completed 30 repetitions of exercise E1, summary attached."`
- **Attachment:** the summary CSV (`pt_summary_e1_TIMESTAMP.csv`) containing the movement quality counts table

**3. Send**
```python
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(sender, password)
    server.sendmail(sender, CLINICIAN_EMAIL, msg.as_string())
```
Uses Gmail's encrypted SMTP port (465). Returns `(True, "")` on success or `(False, exception_message)` on any error.

---

## 10. `load_data` (lines 293–295)

```python
@st.cache_data
def load_data() -> pd.DataFrame
```

Reads `processed_data.csv` once and caches the result. `@st.cache_data` serialises the DataFrame and reuses it on every Streamlit rerun — the file is not re-read from disk on each user interaction.

---

## 11. App entry point and page setup (lines 302–307)

```python
st.set_page_config(...)
st.title("Physical Therapy Exercise Tracker")
model, le, feature_cols = get_trained_model()
df = load_data()
```

These four lines run every time the page loads (or the user interacts with a widget). Because both functions are cached, they return instantly after the first call.

---

## 12. Step 1 — Exercise selection (lines 313–317)

```python
exercise_options = ["e1", "e2", ..., "e8"]
selected_exercise = st.selectbox("Choose exercise", exercise_options)
```

Renders a dropdown. The selected value (`"e1"` … `"e8"`) is used downstream to filter the dataset and to set the exercise one-hot feature during inference. An `st.info` banner confirms the selection.

---

## 13. Step 2 — Begin Tracking (lines 323–356)

```python
start_tracking = st.button("Begin Tracking", type="primary")
```

**Session state initialisation:**
```python
if "session_results" not in st.session_state: ...
```
Streamlit reruns the entire script on every interaction. `st.session_state` is a dictionary that persists across reruns, so results from a completed session remain visible after the button press is gone.

**When the button is pressed:**
1. Clears any previous session results.
2. Filters `processed_data.csv` to rows where `subject == "s3"` and `exercise == selected_exercise`.
3. Calls `detect_repetitions` (with a spinner).
4. Displays a green banner: *"Auto-detected N blocks — M repetitions to classify."*

---

## 14. Step 3 — Live feedback loop (lines 362–405)

```python
for idx, rep_df in enumerate(repetitions, start=1):
    label, conf = classify_repetition(...)
    feedback_text, severity = get_feedback(label)
    ...
    time.sleep(0.3)
```

Iterates over every detected repetition window in order.

For each repetition:
- Calls `classify_repetition` → gets a label and confidence score.
- Calls `get_feedback` → gets a human message and a colour.
- Updates `feedback_box` (single placeholder updated in place — only the latest rep is shown).
- Appends a row to `session_results` (the summary table) and a raw-data frame to `raw_frames`.
- Updates `session_table_box` with the growing results table.
- Advances `progress_bar`.
- Sleeps 0.3 s to produce a live, animated effect.

After the loop, both lists are saved into `st.session_state` so they survive future reruns.

---

## 15. Step 4 — Clinician report (lines 411–453)

```python
if st.session_state.session_results:
    ...
```

This block renders only if a session has been completed. It persists on the page even after subsequent widget interactions.

**Summary table:** counts of `correct` / `fast` / `low_amplitude` predictions using `value_counts()`.

**Bar chart:** Plotly bar chart of those counts.

**Two CSV objects are generated:**
| Variable | Contents | Used for |
|---|---|---|
| `full_csv_bytes` | All raw sensor timesteps, tagged with `rep_id`, `prediction`, `confidence`, `feedback` | "Download full CSV report" button |
| `summary_csv_bytes` | Just the two-column counts table (`Movement Quality`, `Count`) | "Send to Clinician" email attachment |

**Download button:** `st.download_button` — triggers a browser file download of the full CSV.

**Send to Clinician button:** calls `send_clinician_email` with the summary CSV. Shows `st.success` or `st.error` depending on the result.
