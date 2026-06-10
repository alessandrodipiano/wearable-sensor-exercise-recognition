import pandas as pd 


exercises = ['e1','e2','e3','e4','e5','e6','e7','e8',]
subjects=['s1', 's2', 's3', 's4', 's5',]
labels=['correct', 'fast', 'low_amplitude']

def build_df(root):

    templates = []

    label_map = {
        1: "correct",
        2: "fast",
        3: "low_amplitude"
    }

    sample_id = 0

    for subject_dir in root.glob("s*"):
        for exercise_dir in subject_dir.glob("e*"):
            times_path = exercise_dir / "template_times.txt"
            

            template_times = pd.read_csv(times_path, sep=";")

            for unit_dir in exercise_dir.glob("u*"):
                file_path = unit_dir / "template_session.txt"
                

                df = pd.read_csv(file_path, sep=";")
                df["subject"] = subject_dir.name
                df["exercise"] = exercise_dir.name
                df["unit"] = unit_dir.name
            

                for _, row in template_times.iterrows():
                    start = row["start"]   
                    end = row["end"]
                    exec_type = row["execution type"]

                    segment = df[
                        (df["time index"] >= start) &
                        (df["time index"] <= end)
                    ].copy()

                    segment["execution_type"] = label_map[exec_type]
                    segment["sample_id"] = sample_id

                    templates.append(segment)

                    sample_id += 1   
    templates_df = pd.concat(templates, ignore_index=True)

    return templates_df






 
def analyze_segment_lengths(templates_df):
    """
    For each sample_id, count the number of timesteps.
    Then summarize by execution_type: mean, std, min, max.
    """
 
    # Step 1: count timesteps per sample
    lengths = (
        templates_df
        .groupby([ "execution_type", "subject", "exercise"])
        .size()
        .reset_index(name="n_timesteps")
    )
 
    # Step 2: summary stats grouped by execution type
    summary = (
        lengths
        .groupby("execution_type")["n_timesteps"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(1)
    )
 
    print("=== Timesteps per execution type ===")
    print(summary)
    print()
 
    # Step 3: per exercise breakdown (to see if pattern holds across all 8)
    per_exercise_mean = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .mean()
        .round(1)
        .unstack("execution_type")
    )
    per_exercise_mean.columns = [f"{c}_mean" for c in per_exercise_mean.columns]
 
    per_exercise_max = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .max()
        .unstack("execution_type")
    )
    per_exercise_max.columns = [f"{c}_max" for c in per_exercise_max.columns]
 
    per_exercise_min = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .min()
        .unstack("execution_type")
    )
    per_exercise_min.columns = [f"{c}_min" for c in per_exercise_min.columns]
 
    per_exercise = pd.concat([per_exercise_mean, per_exercise_max, per_exercise_min], axis=1).sort_index(axis=1)
 
    print("=== Mean, min and max timesteps per exercise × execution type ===")
    print(per_exercise)
 
    return lengths, summary, per_exercise






def compare_sensor_means(templates_df, exec_types=("correct", "low_amplitude")):
    """
    Compare average sensor readings between two execution types,
    broken down by exercise × unit combination.
 
    For each (exercise, unit) pair and each sensor group (acc, gyr, mag),
    computes the mean per axis per execution type, plus the diff row.
 
    Returns a dict: { (exercise, unit): { sensor: DataFrame } }
    """
 
    sensor_cols = {
        "acc": ["acc_x", "acc_y", "acc_z"],
        "gyr": ["gyr_x", "gyr_y", "gyr_z"],
        "mag": ["mag_x", "mag_y", "mag_z"],
    }
 
    subset = templates_df[templates_df["execution_type"].isin(exec_types)]
 
    results = {}
 
    for (exercise, unit), group in subset.groupby(["exercise", "unit"]):
        key = (exercise, unit)
        results[key] = {}
 
        print(f"\n{'='*50}")
        print(f"Exercise: {exercise}  |  Unit: {unit}")
        print(f"{'='*50}")
 
        for sensor, cols in sensor_cols.items():
            sensor_means = (
                group
                .groupby("execution_type")[cols]
                .mean()
                .round(4)
            )
 
            # diff row: second exec_type minus first
            if all(t in sensor_means.index for t in exec_types):
                sensor_means.loc[f"diff ({exec_types[1]} - {exec_types[0]})"] = (
                    sensor_means.loc[exec_types[1]] - sensor_means.loc[exec_types[0]]
                )
 
            results[key][sensor] = sensor_means
 
            print(f"\n  {sensor.upper()}")
            print(sensor_means.to_string())
 
    return results



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_short_true_segments(mask, min_len):
    mask = mask.copy()
    groups = (mask != mask.shift()).cumsum()

    for _, group in mask.groupby(groups):
        if bool(group.iloc[0]) and len(group) < min_len:
            mask.loc[group.index] = False

    return mask


def fill_short_false_gaps(mask, max_gap_len):
    mask = mask.copy()
    groups = (mask != mask.shift()).cumsum()

    segments = []
    for _, group in mask.groupby(groups):
        segments.append({
            "value": bool(group.iloc[0]),
            "start": group.index[0],
            "end": group.index[-1],
            "length": len(group)
        })

    for i in range(1, len(segments) - 1):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]
        next_seg = segments[i + 1]

        if (
            prev_seg["value"] is True and
            curr_seg["value"] is False and
            next_seg["value"] is True and
            curr_seg["length"] <= max_gap_len
        ):
            mask.loc[curr_seg["start"]:curr_seg["end"]] = True

    return mask


def extract_idle_periods(mask):
    groups = (mask != mask.shift()).cumsum()
    idle_groups = mask[mask].groupby(groups[mask])
    return [(group.index[0], group.index[-1]) for _, group in idle_groups]





def analyze_subject_exercises(subject, merged_df):
    subject_df = merged_df[merged_df['subject'] == subject]
    exercises = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8']

    all_subject_ex = {}

    for exercise in exercises:
        subject_ex = subject_df[subject_df['exercise'] == exercise].copy()

        if subject_ex.empty:
            print(f"No data for {exercise}")
            continue

        unit_activity = {}

        # Compute magnitudes for all units
        for u in range(1, 6):
            subject_ex[f"acc_mag_u{u}"] = np.sqrt(
                subject_ex[f"acc_x_u{u}"]**2 +
                subject_ex[f"acc_y_u{u}"]**2 +
                subject_ex[f"acc_z_u{u}"]**2
            )

            subject_ex[f"gyr_mag_u{u}"] = np.sqrt(
                subject_ex[f"gyr_x_u{u}"]**2 +
                subject_ex[f"gyr_y_u{u}"]**2 +
                subject_ex[f"gyr_z_u{u}"]**2
            )

            subject_ex[f"mag_mag_u{u}"] = np.sqrt(
                subject_ex[f"mag_x_u{u}"]**2 +
                subject_ex[f"mag_y_u{u}"]**2 +
                subject_ex[f"mag_z_u{u}"]**2
            )

            total_std = (
                subject_ex[f"acc_mag_u{u}"].std() +
                subject_ex[f"gyr_mag_u{u}"].std() +
                subject_ex[f"mag_mag_u{u}"].std()
            )
            unit_activity[f"u{u}"] = total_std

        # Determine most active unit AFTER all units are processed
        most_active_unit = max(unit_activity, key=unit_activity.get)

        print(f"\n{exercise} — most active unit: {most_active_unit}")
        print({k: round(v, 4) for k, v in unit_activity.items()})

        # Add the column to the dataframe
        subject_ex["most_active_unit"] = most_active_unit

        # Save dataframe
        all_subject_ex[exercise] = subject_ex

        threshold = 0.1

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Subject {subject} - {exercise}")

        sensor_info = [
            ("acc", "Acceleration magnitude", axes[0]),
            ("gyr", "Gyroscope magnitude", axes[1]),
            ("mag", "Magnetometer magnitude", axes[2]),
        ]

        anything_plotted = False

        for sensor_prefix, title, ax in sensor_info:
            plotted_here = False

            for u in range(1, 6):
                std = subject_ex[f"{sensor_prefix}_mag_u{u}"].std()

                if std > threshold:
                    ax.plot(
                        subject_ex["time index"],
                        subject_ex[f"{sensor_prefix}_mag_u{u}"],
                        label=f"u{u} (std={std:.2f})"
                    )
                    plotted_here = True
                    anything_plotted = True

            ax.set_title(title)
            ax.set_xlabel("time index")
            ax.grid(True)

            if sensor_prefix == "acc":
                ax.set_ylabel("magnitude")

            if plotted_here:
                ax.legend()
            else:
                ax.text(
                    0.5, 0.5, "No signal above threshold",
                    transform=ax.transAxes,
                    ha="center", va="center"
                )

        if anything_plotted:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            plt.close()
            print(f"No sensor exceeded threshold for {exercise}")

    return all_subject_ex


def find_idle_periods_adaptive(
    df,
    window_size=100,
    smooth_window=50,
    q_low=0.10,
    q_high=0.60,
    alpha=0.20,
    min_idle_len=100,
    max_gap_len=200,
    plot=True
):
    """
    Detect 2 idle periods using the most active unit, with combined energy from
    acc + gyr + mag, and an adaptive threshold.

    threshold = q_low_value + alpha * (q_high_value - q_low_value)
    """

    unit = df["most_active_unit"].iloc[0]

    acc_col = f"acc_mag_{unit}"
    gyr_col = f"gyr_mag_{unit}"
    mag_col = f"mag_mag_{unit}"

    required_cols = [acc_col, gyr_col, mag_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Rolling local energy per modality
    acc_energy = pd.Series(df[acc_col].to_numpy()).rolling(window_size, center=True).std()
    gyr_energy = pd.Series(df[gyr_col].to_numpy()).rolling(window_size, center=True).std()
    mag_energy = pd.Series(df[mag_col].to_numpy()).rolling(window_size, center=True).std()

    # Combined energy
    energy = acc_energy + gyr_energy + mag_energy
    energy = energy.bfill().ffill()

    # Smooth
    energy_smooth = energy.rolling(smooth_window, center=True).median().bfill().ffill()

    # Adaptive threshold
    low_val = energy_smooth.quantile(q_low)
    high_val = energy_smooth.quantile(q_high)
    threshold = low_val + alpha * (high_val - low_val)

    # Raw idle mask
    idle_mask = energy_smooth < threshold

    # Clean mask
    idle_mask = remove_short_true_segments(idle_mask, min_len=min_idle_len)
    idle_mask = fill_short_false_gaps(idle_mask, max_gap_len=max_gap_len)
    idle_mask = remove_short_true_segments(idle_mask, min_len=min_idle_len)

    # Extract intervals
    idle_periods = extract_idle_periods(idle_mask)

    # Keep 2 longest
    idle_periods = sorted(idle_periods, key=lambda x: x[1] - x[0], reverse=True)[:2]
    idle_periods = sorted(idle_periods, key=lambda x: x[0])

    result = {
        "most_active_unit": unit,
        "threshold": threshold,
        "low_val": low_val,
        "high_val": high_val,
        "energy_smooth": energy_smooth,
        "idle_mask": idle_mask,
        "idle_periods": idle_periods,
        "columns_used": {
            "acc": acc_col,
            "gyr": gyr_col,
            "mag": mag_col
        }
    }

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(energy_smooth, label="combined smoothed energy")
        plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.3f}")

        for start, end in idle_periods:
            plt.axvspan(start, end, alpha=0.2)

        plt.xlabel("time_step")
        plt.ylabel("energy")
        plt.title(f"Combined energy from {acc_col}, {gyr_col}, {mag_col}")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(idle_mask.astype(int))
        plt.xlabel("time_step")
        plt.ylabel("idle")
        plt.title(f"Idle mask using most active unit {unit}")
        plt.grid(True)
        plt.show()

    return result


###########################ààà
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


def safe_savgol(x, window_length=51, polyorder=3):
    """
    Apply Savitzky-Golay smoothing safely.

    The window length must be:
    - odd
    - <= len(x)
    - > polyorder
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n <= polyorder + 2:
        return x

    window_length = min(window_length, n if n % 2 == 1 else n - 1)

    if window_length <= polyorder:
        return x

    return savgol_filter(x, window_length, polyorder)


def enforce_alternating_extrema(sig_n, peaks, valleys, max_valleys=11):
    """
    Enforce an alternating sequence of extrema.

    This removes:
    - peak -> peak
    - valley -> valley

    If two consecutive peaks occur, keep the higher one.
    If two consecutive valleys occur, keep the deeper one.
    """
    peaks = np.asarray(peaks, dtype=int)
    valleys = np.asarray(valleys, dtype=int)

    extrema = []

    for p in peaks:
        extrema.append((p, "peak"))

    for v in valleys:
        extrema.append((v, "valley"))

    # Sort all extrema by time/sample index
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
            # Same type twice in a row: keep the stronger one
            if kind == "peak":
                # Higher peak is stronger
                if sig_n[idx] > sig_n[prev_idx]:
                    cleaned[-1] = (idx, kind)

            elif kind == "valley":
                # Deeper valley is stronger
                if sig_n[idx] < sig_n[prev_idx]:
                    cleaned[-1] = (idx, kind)

    cleaned_peaks = np.array(
        [idx for idx, kind in cleaned if kind == "peak"],
        dtype=int
    )

    cleaned_valleys = np.array(
        [idx for idx, kind in cleaned if kind == "valley"],
        dtype=int
    )

    # Limit the number of valleys
    if len(cleaned_valleys) > max_valleys:
        deepest = np.argsort(sig_n[cleaned_valleys])[:max_valleys]
        cleaned_valleys = np.sort(cleaned_valleys[deepest])

        # Re-apply alternation after removing excess valleys
        return enforce_alternating_extrema(
            sig_n=sig_n,
            peaks=cleaned_peaks,
            valleys=cleaned_valleys,
            max_valleys=max_valleys
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
    valley_prominence=0.5,
    plot=True
):
    """
    Detect clean repetition peaks and valleys from the most active gyroscope unit.

    Assumption:
        One repetition has the form

            valley -> peak -> valley

    Therefore:
        - there cannot be two consecutive peaks
        - there cannot be two consecutive valleys
        - max number of valleys is usually expected_reps + 1

    Parameters
    ----------
    used : pandas.DataFrame
        Data for one subject-exercise-label combination.
    u : str
        Unit name, for example "u2".
    expected_reps : int
        Expected number of repetitions.
    max_valleys : int
        Maximum allowed number of valleys. For 10 repetitions, usually 11.
    window_length : int
        Savitzky-Golay smoothing window.
    polyorder : int
        Savitzky-Golay polynomial order.
    peak_prominence : float
        Prominence threshold for peaks.
    valley_prominence : float
        Prominence threshold for valleys.
    plot : bool
        Whether to plot the detected extrema.

    Returns
    -------
    peaks : np.ndarray
        Clean peak indices, relative to `used`.
    valleys : np.ndarray
        Clean valley indices, relative to `used`.
    info : dict
        Diagnostic information.
    """

    time = used["time index"].to_numpy()

    # Smooth gyroscope axes
    gx = safe_savgol(
        used[f"gyr_x_{u}"].to_numpy(),
        window_length=window_length,
        polyorder=polyorder
    )

    gy = safe_savgol(
        used[f"gyr_y_{u}"].to_numpy(),
        window_length=window_length,
        polyorder=polyorder
    )

    gz = safe_savgol(
        used[f"gyr_z_{u}"].to_numpy(),
        window_length=window_length,
        polyorder=polyorder
    )

    signals = {
        "x": gx,
        "y": gy,
        "z": gz
    }

    # Choose the axis with the largest oscillation
    axis = max(signals, key=lambda a: np.ptp(signals[a]))
    sig = signals[axis]

    # Normalize signal
    std = np.std(sig)

    if std == 0:
        sig_n = sig - np.mean(sig)
    else:
        sig_n = (sig - np.mean(sig)) / std



    # Detect candidate peaks
    peaks, peak_props = find_peaks(
        sig_n,
        
        prominence=peak_prominence
    )

    # Detect candidate valleys
    valleys, valley_props = find_peaks(
        -sig_n,
       
        prominence=valley_prominence
    )

    # Enforce:
    #   no peak -> peak
    #   no valley -> valley
    #   max number of valleys
    peaks, valleys = enforce_alternating_extrema(
        sig_n=sig_n,
        peaks=peaks,
        valleys=valleys,
        max_valleys=max_valleys
    )

    info = {
        "axis": axis,
        "unit": u,
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
        "expected_reps": expected_reps,
        "max_valleys": max_valleys,
        
        "peak_prominence": peak_prominence,
        "valley_prominence": valley_prominence
    }

    if plot:
        plt.figure(figsize=(12, 4))

        plt.plot(
            time,
            sig_n,
            label=f"normalized gyr_{axis}_{u}"
        )

        plt.plot(
            time[peaks],
            sig_n[peaks],
            "o",
            label="clean peaks"
        )

        plt.plot(
            time[valleys],
            sig_n[valleys],
            "x",
            label="clean valleys"
        )

        plt.title(
            f"unit={u}, axis={axis}, "
            f"peaks={len(peaks)}, valleys={len(valleys)}, "
            f"expected={expected_reps}"
        )

        plt.xlabel("time index")
        plt.ylabel("normalized signal")
        plt.legend()
        plt.show()

    return peaks, valleys, info