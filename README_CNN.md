# CNN Pipeline & Live Inference — Technical Reference

This document explains the **CNN-based** movement-quality system end to end: how the
training dataset is built, how the model is defined and trained, how repetitions are
detected, and how live data flows from a phone (via phyphox) into a prediction.

> The repo also contains an **XGBoost** pipeline documented in `README.md` and a
> walkthrough of the XGBoost Streamlit app (`app.py`) in `README2.md`. The CNN system
> described here is a **separate, parallel track** built around `CNNMLPModel`,
> `fold_results.pth`, and `app_cnn.py`. The two pipelines share the same raw data and
> the same three quality labels but nothing else.

---

## 1. What the system does

Given wearable IMU (accelerometer + gyroscope + magnetometer) recordings of a physical
therapy exercise, the model classifies **each repetition** into one of three movement
quality classes:

| Index | Label | Meaning |
|------:|-------|---------|
| 0 | `correct` | Repetition performed properly |
| 1 | `low_amplitude` | Range of motion too small |
| 2 | `fast` | Performed too quickly |

> ⚠️ **The index→label mapping `{0: correct, 1: low_amplitude, 2: fast}` is the
> contract** that every consumer of the model must respect. It is defined in
> `dataset_creation.py` as `label_to_idx = {'correct':0, 'low_amplitude':1, 'fast':2}`
> and mirrored in `model_inference.py` and `app_cnn.py`. Do not reorder it.

There are **8 exercises** (`e1`–`e8`) and **5 subjects** (`s1`–`s5`). Exercises
`e2, e6, e7, e8` are **arm** exercises; the rest are **leg** exercises (`limb` feature).

---

## 2. Two pipelines at a glance

```
                          ┌─────────────────────────────────────────────┐
   OFFLINE (training)     │  data/processed_data.csv  (wide, 65 cols)    │
                          └───────────────────┬─────────────────────────┘
                                              │
                    python/dataset_creation.py│  segment reps, resample→128,
                                              │  build X_seq(45) / X_ex(9) / X_info(5) / y
                                              ▼
                    notebooks/CNN/CV_lopo_cnn.ipynb
                       - LOPO cross-validation (diagnostic)
                       - single "s2" fold trained & saved
                                              │
                                              ▼
                            notebooks/CNN/fold_results.pth   ← the only saved weights ("s2")
                                              │
        ┌─────────────────────────────────────┴───────────────────────────┐
        ▼                                                                   ▼
 LIVE (phyphox)                                                    app_cnn.py (Streamlit)
 python/phyphox_connection.py  → read phone sensors                replays a CSV, predicts,
 python/rep_detection.py       → record, detect reps, segment      renders feedback cards
 python/model_inference.py     → expand 9→45 ch, load model, run    (⚠ currently out of sync —
                                                                     see §10)
```

The two pipelines **detect repetitions differently** and **shape the input differently**;
the only shared artifact is the trained model in `fold_results.pth`.

---

## 3. Data layout

```
data/
├── raw/
│   └── s{1..5}/e{1..8}/
│       ├── template_times.txt          # start;end;execution type per segment
│       └── u{1..5}/template_session.txt # one IMU unit's full session (semicolon-sep)
└── processed_data.csv                   # wide table built from raw (65 columns)
```

- **Raw** is the original UCI-style layout. `utilities/src.py::build_df()` walks it and
  slices each session into labeled segments using `template_times.txt`
  (`execution type` 1/2/3 → `correct`/`fast`/`low_amplitude`).
- **`processed_data.csv`** is the wide table everything downstream actually reads. Each
  row is one timestep for one `(subject, exercise, label)` trial, with **all 5 units**
  side by side. Key columns:
  - `subject`, `exercise`, `label`, `time index`, `most_active_unit`
  - 45 raw channels: `{acc,gyr,mag}_{x,y,z}_u{1..5}`
  - 15 magnitude channels: `{acc,gyr,mag}_mag_u{1..5}` (per-unit L2 norm of the 3 axes)
  - `most_active_unit` is the sensor unit (`u1`–`u5`) with the most signal energy for
    that trial — chosen so rep detection always runs on the limb that actually moves.

> The notebook that produces `processed_data.csv` (preprocessing / magnitude
> computation / most-active-unit selection) is `notebooks/preprocessing.ipynb`; the
> magnitude and per-subject activity logic also lives in `utilities/src.py`
> (`analyze_subject_exercises`, `compare_sensor_means`, idle-period helpers).

---

## 4. The model — `utilities/model.py`

The file defines three classes; **only `CNNMLPModel` is current.**

| Class | Status | Branches | `forward` signature |
|-------|--------|----------|---------------------|
| `CNNmodel_base` | legacy/experiment | seq only | `forward(x)` |
| `CNNMLPModel0` | **legacy** | seq + one combined "global" vector | `forward(x_seq, x_global)` |
| `CNNMLPModel` | **current** | seq + exercise + subject-info (3 separate branches) | `forward(x_seq, x_ex, x_info)` |

### `CNNMLPModel` architecture

A three-branch network with late fusion:

1. **Sequence branch (`use_seq`)** — 1-D CNN over the IMU time series.
   Input shape `(B, input_dim_seq, T)`.
   `Conv1d(→64,k7) → BN → ReLU → Conv1d(→128,k5) → BN → ReLU → Conv1d(→128,k3) → BN →
   ReLU → MaxPool1d(2) → Dropout(0.3) → Conv1d(→256,k3) → BN → ReLU → AdaptiveAvgPool1d(1)`
   → flatten → **256-d** vector.
2. **Exercise branch (`use_ex`)** — MLP over the exercise descriptor.
   Input `(B, input_dim_exercise)` → `Linear(→32) → ReLU → LayerNorm → Dropout →
   Linear(→16) → ReLU` → **16-d**.
3. **Subject-info branch (`use_info`)** — identical MLP shape over subject info.
   Input `(B, input_dim_info)` → **16-d**.

The active branch outputs are concatenated (`fusion_dim = 256 + 16 + 16 = 288` when all
three are on) and passed through the classifier
`Linear(fusion_dim→32) → ReLU → Dropout → Linear(32→num_classes)`.

Each branch is optional via `use_seq` / `use_ex` / `use_info`, which is what makes the
class reusable for ablations.

### The three input tensors (the contract)

These shapes are fixed by how `dataset_creation.py` builds the training data and how the
saved `fold_results.pth` was trained:

| Tensor | Shape | Content |
|--------|-------|---------|
| `X_seq`  | `(N, 45, 128)` | 45 raw channels (`{acc,gyr,mag}_{x,y,z}_u{1..5}`), time resampled to 128 |
| `X_ex`   | `(N, 9)`  | `[limb, e1_onehot, …, e8_onehot]` |
| `X_info` | `(N, 5)`  | `[gender, age, weight, height, original_rep_length]` |
| `y`      | `(N,)`    | class index 0/1/2 |

The saved model was therefore built as:
```python
CNNMLPModel(input_dim_seq=45, input_dim_exercise=9, input_dim_info=5, num_classes=3)
```

---

## 5. Building the training dataset — `python/dataset_creation.py`

Run-on-import script (it executes top to bottom when imported) that turns
`processed_data.csv` into the four tensors above. It exports
`X_seq, X_ex, X_info, y, all_subjects, all_active_units, unique_subjects, inputs_seq, …`
which the CV notebook imports directly.

Pipeline:

1. **Load** `processed_data.csv`.
2. **Exercise features** — `limb_map` adds a leg/arm flag; `pd.get_dummies` adds
   `exercise_e1…e8` one-hot columns.
3. **Subject info** — a hard-coded `subject_info` table (gender/age/weight/height for
   s1–s5) becomes `subject_info_dict`.
4. **Rep detection** — for every `(subject, exercise, label)` trial it runs
   `detect_peaks_and_valleys_clean` (from `utilities/src.py`) on the **gyroscope** axis
   of the trial's `most_active_unit`. Valleys mark rep boundaries
   (one rep = `valley → peak → valley`).
   - Two trials are manually patched because detection misses their first boundary:
     ```python
     valleys_indexes["s2-e1-correct"] = np.insert(..., 0, 1)
     valleys_indexes["s3-e1-correct"] = np.insert(..., 0, 1)
     ```
5. **Label reps** — each timestep gets a `rep` id (1–10) by slicing valley-to-valley.
6. **Segment + resample** — each rep's 45 raw channels are resampled to **`target_len=128`**
   via linear interpolation (`resample_sequence`), so all reps are the same length.
   The original (pre-resample) length is kept and appended to the info vector.
7. **Stack** into `X_seq (N,45,128)`, `X_ex (N,9)`, `X_info (N,5)`, `y (N,)`.
   Final dataset is **N = 1193** reps (≈ 80 per subject per class).

> Note: `inputs_seq` here is the **45 raw channels**. There is a commented-out
> 15-channel magnitude variant in the file — that older choice is what `app_cnn.py`
> still assumes (see §10).

---

## 6. Training & cross-validation — `notebooks/CNN/CV_lopo_cnn.ipynb`

Two distinct sections:

### (a) LOPO cross-validation (diagnostic only)
Leave-One-Subject-Out loop over all 5 subjects. For each held-out subject:

- **Split** by subject (no subject appears in both train and val — the honest test of
  generalization to a new patient).
- **Normalize** with **training-fold statistics only**:
  - `normalize_seq_train_val` — per-channel mean/std over train.
  - `normalize_info_train_val` — standardizes age/weight/height/rep_length; leaves
    gender (binary) untouched.
- **Augment** — `make_most_active_sensor_version` zeroes every channel except the
  most-active unit's, then concatenates that masked copy to the training set
  (**doubling** it). This teaches the model to also work from a single unit — directly
  relevant to live single-sensor inference.
- Train a fresh `CNNMLPModel` (AdamW, lr 5e-5, wd 1e-3, 70 epochs), track best val loss.

This loop **stores only metrics** (`accuracy`, `macro_f1`, `confusion_matrix`) in a list
— **it does not save model weights.** Reported per-subject accuracy varies widely
(s2 ≈ 0.83, s3 ≈ 0.81, s5 ≈ 0.70, s1/s4 much lower), which is the realistic
new-patient generalization picture.

### (b) Single "s2" fold — the saved model
A separate section repeats the split with `val_subject = "s2"`, trains
(Adam, lr 1e-4, 20 epochs), keeps the **best-epoch** `state_dict`, and saves:

```python
results = {"s2": {"model_state_dict": ..., "accuracy": ..., "confusion_matrix": ..., ...}}
torch.save(results, "fold_results.pth")
```

> ⚠️ **`fold_results.pth` contains only the `"s2"` fold.** The LOPO loop never persisted
> the other folds. Every consumer keys it as `checkpoint["s2"]["model_state_dict"]`.
> This is why `s2` is the only *honest* held-out evaluation subject; predictions on
> s1/s3/s4/s5 are partly in-sample and will look inflated.

---

## 7. Peak / rep detection — `notebooks/CNN/peak_detection.ipynb` + `utilities/src.py`

This notebook is where the valley-based rep detector was **developed and tuned**; the
final functions were copied into `utilities/src.py` (used offline) and a phyphox-adapted
copy into `python/phyphox_connection.py` (used live).

Core idea (`detect_peaks_and_valleys_clean`):

1. Smooth the three gyroscope axes of the active unit (`safe_savgol`, a guarded
   Savitzky–Golay that no-ops on too-short signals).
2. Pick the axis with the **largest peak-to-peak range** (`np.ptp`) — the dominant
   rotation axis for that exercise.
3. Normalize (z-score).
4. `scipy.signal.find_peaks` for peaks and (on the negated signal) valleys, with a
   minimum `distance` derived from `expected_reps`.
5. `enforce_alternating_extrema` cleans the result so extrema strictly alternate
   `valley → peak → valley`: consecutive same-type extrema are collapsed to the
   stronger one, and the number of valleys is capped at `max_valleys` (≈ `reps + 1`).

Returns `(peaks, valleys, info)`. **Valleys are the rep boundaries**; N valleys →
N−1 reps. The notebook deliberately only plots trials where `len(valleys) != 11` so you
can eyeball the failures. The repeated console lines like
`subject=s1, exercise=e1, ... valleys=11` are this detector's per-trial output.

The earlier `detect_rep_valleys_acc` in the same notebook is a simpler first attempt
(valleys only, no peak/alternation logic) kept for comparison.

---

## 8. Live pipeline — phyphox → prediction

### 8.1 `python/phyphox_connection.py`
Talks to the **phyphox** phone app over its HTTP "Remote access" API.

- `BASE_URL` — **must be set** to the phone's address shown in phyphox after enabling
  Remote access, e.g. `http://192.168.1.11:8080`. The phone and the PC must be on the
  same network.
- `SENSOR_CHANNELS` — the 9 live channels: `accX/Y/Z, gyrX/Y/Z, magX/Y/Z`.
- `send_command(cmd)` — `GET /control?cmd=…` (`clear`/`start`/`stop`).
- `read_values()` — `GET /get?accX&accY&…` and returns the latest sample of each channel
  plus a `computer_time` timestamp.
- It also carries a **phyphox-adapted copy** of `detect_peaks_and_valleys_clean` that
  works on live column names (`gyrX/gyrY/gyrZ`, `computer_time`) instead of the offline
  `gyr_x_u2` / `time index` names. (A second, commented-out variant is kept below it.)

### 8.2 `python/rep_detection.py`
The live **recording + segmentation** orchestrator (also runs on import):

1. `send_command("clear")`, wait 5 s, `send_command("start")`.
2. Poll `read_values()` every `SAMPLE_INTERVAL` (0.1 s) for `DURATION` (30 s),
   collecting rows; `send_command("stop")` in a `finally`.
3. Build a DataFrame, `dropna()`.
4. `detect_peaks_and_valleys_clean(...)` → valleys.
5. `segment_repetitions_from_valleys(...)` slices valley-to-valley and
   `resample_rep`s each rep to **`target_length=128`**, producing:
   - `repetitions` — `(N, 128, 9)` (note: **9** live channels, time-major)
   - `original_lengths` — `(N,)`

### 8.3 `python/model_inference.py`
Turns the live reps into the model's 3-tensor input and runs the classifier:

1. Imports `repetitions, original_lengths` from `rep_detection`.
2. Hard-coded **user inputs** at the top: `gender, age, weight, height, exercise`, and
   the worn `unit` (default `"u2"`). Edit these per session.
3. `repetitions (N,128,9) → transpose → (N,9,128)`.
4. **`expand_single_unit_to_45`** — the live phone provides one unit's 9 channels, but
   the model expects 45. This places the 9 live channels into the chosen unit's slots
   (`target_ch = live_ch*5 + unit_index`) and leaves the other four units zero —
   exactly matching the most-active-unit masking used during training augmentation
   (§6a). Output `(N,45,128)`.
5. Builds `X_info (N,5)` from the user inputs + each rep's `original_len`, and
   `X_ex (N,9)` from `make_exercise_dummy(exercise)`.
6. Loads `CNNMLPModel(input_dim_seq=45, input_dim_exercise=9, input_dim_info=5,
   num_classes=3)`, restores `fold_results["s2"]["model_state_dict"]`, runs inference,
   prints per-rep predicted label + class probabilities.

> ⚠️ **Known gap:** `model_inference.py` does **not** apply the seq/info normalization
> that training used (§6a). For best results it should normalize with the saved
> training-fold statistics. Those stats are currently **not persisted** in
> `fold_results.pth`, so they would need to be saved from the notebook (or recomputed)
> for live inference to match training conditions.

---

## 9. Package `__init__.py` files

`python/__init__.py` and `utilities/__init__.py` are **empty package markers** — they
exist only so `python.*` and `utilities.*` import as packages. The real path setup
(adding the project root to `sys.path`) is done explicitly at the top of each entry
script/notebook (`PROJECT_ROOT` logic).

---

## 10. `app_cnn.py` — current state and why it needs fixing

`app_cnn.py` is a Streamlit app that **replays a CSV** (not live data): the user uploads
`processed_data.csv`, the app detects reps for `SIMULATION_SUBJECT` (default `"s3"`),
predicts on a random sample, and renders styled feedback cards comparing prediction vs
the ground-truth label.

**It is currently out of sync with the trained model and will not run as written.**
Concretely:

| What `app_cnn.py` does now | What the saved model / pipeline requires |
|---|---|
| `CNNMLPModel(input_dim_seq=15, input_dim_global=9, num_classes=3)` | `CNNMLPModel(input_dim_seq=45, input_dim_exercise=9, input_dim_info=5, num_classes=3)` — **`input_dim_global` is not a valid argument** (it belonged to the legacy `CNNMLPModel0`) → `TypeError` |
| `model(X_seq, X_glob)` — 2 inputs | `forward(x_seq, x_ex, x_info)` — **3 inputs** |
| 15 **magnitude** channels (`acc/gyr/mag_mag_u1..u5`) | 45 **raw** channels (`{acc,gyr,mag}_{x,y,z}_u{1..5}`) |
| Pads/truncates each rep to `CNN_MAX_LEN = 237` | Reps **resampled to 128** |
| No normalization | Training used seq + info normalization |
| Builds a single 9-d "global" vector | Needs separate `X_ex (9)` and `X_info (5)` |

It also uses an unprofessional placeholder feedback string for `low_amplitude` that
should be replaced.

**To make it work**, `app_cnn.py` needs to be brought in line with
`dataset_creation.py` / `model_inference.py`:
construct the model with the 3-branch signature, feed 45 raw channels resampled to 128,
build the `X_ex` (limb + exercise one-hot) and `X_info` (subject demographics +
rep length) tensors, apply the same normalization, and call
`model(x_seq, x_ex, x_info)`. (This is the follow-up task.)

The parts of `app_cnn.py` that are already fine: the page styling/HTML, the
`IDX_TO_LABEL` mapping (matches the contract), the `LIMB_MAP`, and the rep-detection
call into `detect_peaks_and_valleys_clean`.

---

## 11. How to run

> Requires the project root on `PYTHONPATH`. The entry scripts add it themselves; for
> ad-hoc use, run from the repo root with the venv active.

**Build the dataset / inspect it** (imports run the pipeline):
```bash
python -c "import python.dataset_creation"
```

**Train & save the model:** open `notebooks/CNN/CV_lopo_cnn.ipynb`, run the LOPO section
for diagnostics, then the `s2` training section, then the `torch.save(results, "fold_results.pth")`
cell.

**Live inference from a phone:**
1. Install phyphox, add an experiment exposing accelerometer/gyroscope/magnetometer,
   enable **Remote access**, and copy the shown URL into `BASE_URL` in
   `python/phyphox_connection.py`.
2. Set the user inputs at the top of `python/model_inference.py`.
3. Run `python -m python.model_inference` (this transitively records via
   `rep_detection.py` — be ready to move when it says "Recording starts in 5 seconds").

**Streamlit (CSV replay):**
```bash
streamlit run app_cnn.py     # ⚠ needs the §10 fixes first
```

---

## 12. File map

| File | Role |
|------|------|
| `data/processed_data.csv` | Wide processed dataset everything reads |
| `utilities/src.py` | Dataset build (`build_df`), magnitude/idle analysis, **rep detector** (`detect_peaks_and_valleys_clean`), shared constants (`subjects`, `exercises`, `labels`) |
| `utilities/model.py` | `CNNMLPModel` (current) + legacy `CNNMLPModel0`, `CNNmodel_base` |
| `python/dataset_creation.py` | Builds `X_seq/X_ex/X_info/y` tensors for training |
| `notebooks/CNN/CV_lopo_cnn.ipynb` | LOPO CV + trains/saves the `s2` model |
| `notebooks/CNN/peak_detection.ipynb` | Development/tuning of the rep detector |
| `notebooks/CNN/fold_results.pth` | Saved weights — **`"s2"` fold only** |
| `python/phyphox_connection.py` | Phone HTTP API client + live rep detector |
| `python/rep_detection.py` | Record 30 s, detect reps, segment → `(N,128,9)` |
| `python/model_inference.py` | Expand 9→45 ch, build tensors, load model, predict |
| `app_cnn.py` | Streamlit CSV-replay UI (**needs §10 fixes**) |
| `python/__init__.py`, `utilities/__init__.py` | Empty package markers |
```
