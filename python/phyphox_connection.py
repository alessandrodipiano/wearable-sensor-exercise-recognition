import requests
import time


# Copy this from the phyphox app after enabling Remote access
BASE_URL = "http://192.168.1.11:8080"

# Change these depending on the experiment
SENSOR_CHANNELS = (
    "accX", "accY", "accZ",
    "gyrX", "gyrY", "gyrZ",
    "magX", "magY", "magZ",
)
DURATION = 30
SAMPLE_INTERVAL = 0.1

def send_command(command):
    response = requests.get(f"{BASE_URL}/control?cmd={command}", timeout=5)
    response.raise_for_status()

def read_values():
    query = "&".join(SENSOR_CHANNELS)
    response = requests.get(f"{BASE_URL}/get?{query}", timeout=5)
    response.raise_for_status()
    data = response.json()

    row = {"computer_time": time.time()}

    for channel in SENSOR_CHANNELS:
        buffer = data["buffer"][channel]["buffer"]
        row[channel] = buffer[0] if buffer else None

    return row






import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


def safe_savgol(x, window_length=11, polyorder=2):
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
    expected_reps=10,
    max_valleys=11,
    window_length=51,
    polyorder=3,
    peak_prominence=0.5,
    valley_prominence=0,
    plot=True
):
    """
    Detect clean repetition peaks and valleys from the most active gyroscope axis.

    Assumption:
        One repetition has the form:

            valley -> peak -> valley

    Therefore:
        - there cannot be two consecutive peaks
        - there cannot be two consecutive valleys
        - max number of valleys is usually expected_reps + 1
    """

    time = used["computer_time"].to_numpy()

    # Extract gyroscope axes as 1-D NumPy arrays.
    # IMPORTANT: no trailing commas.
    gx = used["gyrX"].to_numpy(dtype=float)
    gy = used["gyrY"].to_numpy(dtype=float)
    gz = used["gyrZ"].to_numpy(dtype=float)

    signals = {
        "x": gx,
        "y": gy,
        "z": gz,
    }

    # Choose the axis with the largest oscillation
    axis = max(signals, key=lambda a: np.ptp(signals[a]))
    sig = signals[axis]

    # Ensure signal is genuinely 1-D
    sig = np.asarray(sig, dtype=float).ravel()


    # Normalize signal
    std = np.std(sig)

    if std == 0:
        sig_n = sig - np.mean(sig)
    else:
        sig_n = (sig - np.mean(sig)) / std

    # Estimate minimum distance between extrema in samples
    expected_period = len(sig_n) / expected_reps
    distance = max(1, int(0.5 * expected_period))

    # Detect candidate peaks
    peaks, peak_props = find_peaks(
        sig_n,
        distance=distance,
        prominence=peak_prominence
    )

    # Detect candidate valleys
    valleys, valley_props = find_peaks(
        -sig_n,
        distance=distance,
        prominence=valley_prominence
    )

    # Enforce alternating extrema and max valleys
    peaks, valleys = enforce_alternating_extrema(
        sig_n=sig_n,
        peaks=peaks,
        valleys=valleys,
        max_valleys=max_valleys
    )

    info = {
        "axis": axis,
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
        "expected_reps": expected_reps,
        "max_valleys": max_valleys,
        "distance": distance,
        "peak_prominence": peak_prominence,
        "valley_prominence": valley_prominence,
        "window_length": window_length,
        "polyorder": polyorder,
    }

    if plot:
        plt.figure(figsize=(12, 4))

        plt.plot(
            time,
            sig_n,
            label=f"normalized gyr_{axis}"
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
            f"axis={axis}, "
            f"peaks={len(peaks)}, valleys={len(valleys)}, "
            f"expected={expected_reps}"
        )

        plt.xlabel("computer_time")
        plt.ylabel("normalized signal")
        plt.legend()
        plt.show()

    return peaks, valleys, info
















'''

def detect_peaks_and_valleys_clean(
    used,
    expected_reps=10,
    max_valleys=11,
    window_length=51,
    polyorder=3,
    peak_prominence=0.5,
    valley_prominence=0,
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

    time = used["computer_time"].to_numpy()

    # Smooth gyroscope axes
    gx = safe_savgol(
        used[f"gyrX"].to_numpy(),
        window_length=window_length,
        polyorder=polyorder
    )

    gy = safe_savgol(
        used[f"gyrY"].to_numpy(),
        window_length=window_length,
        polyorder=polyorder
    )

    gz = safe_savgol(
        used[f"gyrZ"].to_numpy(),
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

    # Estimate minimum distance between extrema in samples
    expected_period = len(sig_n) / expected_reps
    distance = max(1, int(0.5 * expected_period))

    # Detect candidate peaks
    peaks, peak_props = find_peaks(
        sig_n,
        distance=distance,
        prominence=peak_prominence
    )

    # Detect candidate valleys
    valleys, valley_props = find_peaks(
        -sig_n,
        distance=distance,
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
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
        "expected_reps": expected_reps,
        "max_valleys": max_valleys,
        "distance": distance,
        "peak_prominence": peak_prominence,
        "valley_prominence": valley_prominence
    }

    if plot:
        plt.figure(figsize=(12, 4))

        plt.plot(
            time,
            sig_n,
            label=f"normalized gyr_{axis}"
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
            f"axis={axis}, "
            f"peaks={len(peaks)}, valleys={len(valleys)}, "
            f"expected={expected_reps}"
        )

        plt.xlabel("computer_time")
        plt.ylabel("normalized signal")
        plt.legend()
        plt.show()

    return peaks, valleys, info'''











