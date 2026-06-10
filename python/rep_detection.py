from python.phyphox_connection import (
    send_command,
    read_values,
    detect_peaks_and_valleys_clean,
    
    DURATION,
    SAMPLE_INTERVAL,
    SENSOR_CHANNELS
)


import time
import numpy as np
import pandas as pd


def resample_rep(rep, target_length):
    old_x = np.linspace(0, 1, len(rep))
    new_x = np.linspace(0, 1, target_length)

    rep_resampled = np.zeros((target_length, rep.shape[1]))

    for c in range(rep.shape[1]):
        rep_resampled[:, c] = np.interp(new_x, old_x, rep[:, c])

    return rep_resampled


def segment_repetitions_from_valleys(
    df,
    valleys,
    channels=SENSOR_CHANNELS,
    target_length=128
):
    valleys = np.asarray(valleys, dtype=int)
    valleys = np.sort(valleys)

    signal = df[list(channels)].to_numpy(dtype=float)

    reps = []
    original_lengths = []

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i + 1]

        if end <= start:
            continue

        rep_raw = signal[start:end]

        if len(rep_raw) < 2:
            continue

        original_len = rep_raw.shape[0]
        rep_resampled = resample_rep(rep_raw, target_length)

        reps.append(rep_resampled)
        original_lengths.append(original_len)

    reps = np.stack(reps, axis=0)
    original_lengths = np.array(original_lengths, dtype=np.float32)

    return reps, original_lengths




def record_repetitions(
    duration=DURATION,
    sample_interval=SAMPLE_INTERVAL,
    channels=SENSOR_CHANNELS,
    countdown=5,
    expected_reps=17,
    max_valleys=18,
    target_length=128,
    plot=False,
    progress=None,
):
    """
    Record a live session from the connected phyphox device, detect repetition
    boundaries, and slice the signal into fixed-length repetitions.

    progress: optional callable(str) used to report status (the Streamlit app
    passes a callback that updates a status placeholder).

    Returns (repetitions, original_lengths, df, fig):
        repetitions      : np.ndarray (N, target_length, len(channels))
        original_lengths : np.ndarray (N,)
        df               : the full recorded DataFrame
        fig              : matplotlib Figure with the peak/valley plot (always generated)
    """

    def _report(msg):
        if progress is not None:
            progress(msg)
        else:
            print(msg)

    send_command("clear")

    for remaining in range(countdown, 0, -1):
        _report(f"Get ready — recording starts in {remaining}s…")
        time.sleep(1)

    send_command("start")
    start_time = time.time()
    rows = []

    try:
        while time.time() - start_time < duration:
            rows.append(read_values())
            elapsed = time.time() - start_time
            _report(f"Recording… {int(elapsed)}/{int(duration)}s")
            time.sleep(sample_interval)
    finally:
        send_command("stop")
        _report("Recording stopped. Detecting repetitions…")

    df = pd.DataFrame(rows).dropna()

    peaks, valleys, info = detect_peaks_and_valleys_clean(
        used=df,
        expected_reps=expected_reps,
        max_valleys=max_valleys,
        window_length=51,
        polyorder=3,
        peak_prominence=1,
        valley_prominence=0,
        plot=True,  # always generate the figure so the app can display it
    )
    fig = info.pop("fig", None)  # extract figure; keep info dict clean

    repetitions, original_lengths = segment_repetitions_from_valleys(
        df=df,
        valleys=valleys,
        channels=channels,
        target_length=target_length,
    )

    return repetitions, original_lengths, df, fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    repetitions, original_lengths, df, fig = record_repetitions()

    if fig is not None:
        plt.show()  # display the peak-detection plot in the terminal

    print("Repetitions shape:", repetitions.shape)
    print("Original lengths:", original_lengths)
    print("Original lengths shape:", original_lengths.shape)


