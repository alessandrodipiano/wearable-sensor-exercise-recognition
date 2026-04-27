# Wearable Sensor Exercise Recognition

Classifies physical therapy exercise quality (`correct`, `fast`, `low_amplitude`) from wearable IMU data using Leave-One-Subject-Out (LOSO) cross-validation with XGBoost.

---

## Functions

### `extract_features_v13(segment, sensor_cols)`
Converts a raw time-series segment into a flat feature vector. Runs once per repetition window. Computes per-sensor statistical, amplitude, shape, temporal, and spectral descriptors, plus one cross-sensor feature. Returns a dict of named floats.

### `build_dataset(df, extract_fn, n_reps=10)`
Splits each `(subject, exercise, label)` block into `n_reps` equal-length windows along the time axis, then calls `extract_fn` on each window. Uses only the most-active sensor unit's accelerometer and gyroscope magnitudes, renamed to `acc_mag_active` / `gyr_mag_active` so the feature extractor is unit-agnostic. Returns a DataFrame of one row per window.

### `evaluate_loso(dataset, xgb_params)`
Runs Leave-One-Subject-Out cross-validation: for each subject, trains XGBoost on all other subjects and tests on the held-out subject. One-hot encodes exercise type before training. Returns per-fold accuracy results, a summary dict, and the full arrays of true/predicted labels for confusion matrix plotting.

---

## Features

All per-sensor features are computed independently for `acc_mag_active` (accelerometer magnitude) and `gyr_mag_active` (gyroscope magnitude). Feature names are prefixed accordingly, e.g. `acc_mag_active_mean`, `gyr_mag_active_mean`. Explanations below apply to both sensors unless stated otherwise.

### Baseline statistics

| Feature | Formula | Purpose |
|---|---|---|
| `_mean` | arithmetic mean | Central tendency of the signal; dominated by gravity for accelerometer |
| `_std` | standard deviation | Overall signal variability; higher for dynamic movements |
| `_median` | 50th percentile | Robust central tendency, less sensitive to outliers than mean |

### Amplitude

| Feature | Formula | Purpose |
|---|---|---|
| `_range` | max − min | Peak-to-peak swing; directly reflects range of motion |
| `_iqr` | Q75 − Q25 | Robust spread; less affected by brief spikes than range |
| `_spread` | P90 − P10 | Wider percentile band; captures moderate excursions missed by IQR |
| `_rms` | √( mean(x²) ) | Signal energy; for gyroscope (resting ≈ 0) this scales almost linearly with rotation intensity, making it a strong discriminator for `low_amplitude` |

### Distribution shape

| Feature | Formula | Purpose |
|---|---|---|
| `_skew` | third standardised moment | Asymmetry of the distribution; a `fast` rep that accelerates then brakes sharply produces a skewed signal |
| `_kurtosis` | fourth standardised moment − 3 | Peakedness / heavy tails; spiky, impulsive motion (e.g. abrupt stops in `fast` reps) yields high kurtosis |

### Temporal asymmetry

| Feature | Formula | Purpose |
|---|---|---|
| `_energy_diff` | RMS(first half) − RMS(second half) | Detects whether energy is front- or back-loaded within a window; useful for reps that slow down or speed up over time |

### Spectral

Computed on the mean-subtracted signal (DC removed). All FFT features use the one-sided power spectrum.

| Feature | Definition | Purpose |
|---|---|---|
| `_dom_freq_idx` | bin index of the highest-power frequency (DC excluded) | Captures tempo: faster reps shift this index upward |
| `_dom_freq_power` | raw power at the dominant bin | Absolute energy of the primary oscillation |
| `_dom_power_ratio` | dominant bin power / total power | How concentrated energy is at one frequency; high for rhythmic motion |
| `_spectral_entropy` | −∑ p·log(p) over normalised spectrum | Low entropy = energy focused at a few frequencies (rhythmic); high entropy = spread across many (erratic or noisy) |
| `_spectral_centroid` | power-weighted mean bin index | Effective centre of mass of the spectrum; rises with movement speed |
| `_band_ratio` | power(bins 4+) / power(bins 1–3) | High-to-low frequency energy ratio; `fast` reps push more energy into higher bands |
| `_zcr` | zero-crossing rate (mean-subtracted signal) | Counts oscillations per sample; correlated with movement frequency |

### Cross-sensor

| Feature | Formula | Purpose |
|---|---|---|
| `acc_gyr_corr` | Pearson r between `acc_mag_active` and `gyr_mag_active` | Linear coupling between linear acceleration and rotation magnitude within a rep; quality differences between `correct`, `fast`, and `low_amplitude` alter how the two sensors co-vary |

---

## Model

XGBoost multi-class classifier (`multi:softprob`, 3 classes).

| Hyperparameter | Value | Reason |
|---|---|---|
| `max_depth` | 3 | Shallow trees reduce overfitting across subjects |
| `n_estimators` | 100 | Sufficient ensemble size given strong regularisation |
| `learning_rate` | 0.05 | Low rate pairs with moderate tree count |
| `min_child_weight` | 6 | Prevents splits on small leaf groups |
| `gamma` | 0.2 | Minimum gain required to make a split |
| `reg_lambda` | 4.0 | L2 weight regularisation |
| `reg_alpha` | 1.0 | L1 weight regularisation |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.5 | Feature subsampling per tree |

---

## Results (v13, n_reps=10)

| Subject | Train | Test | Gap |
|---|---|---|---|
| s1 | — | — | — |
| s2 | — | — | — |
| s3 | — | — | — |
| s4 | — | — | — |
| s5 | — | — | — |
| **Mean** | | | |

*Fill in after running `notebooks/Test copy.ipynb`.*
