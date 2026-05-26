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


send_command("clear")

print("Prepare. Recording starts in 5 seconds...")
time.sleep(5)

send_command("start")
start_time = time.time()

rows = []

try:
    while time.time() - start_time < DURATION:
        row = read_values()
        rows.append(row)

        print(row)

        time.sleep(SAMPLE_INTERVAL)

finally:
    send_command("stop")
    print("Experiment stopped")


df = pd.DataFrame(rows)
df=df.dropna()
peaks, valleys, info = detect_peaks_and_valleys_clean(
    used=df,
    expected_reps=10,
    max_valleys=11,
    window_length=51,
    polyorder=3,
    peak_prominence=0.5,
    valley_prominence=0,
    plot=True
)

print(info)
print("Peaks:", peaks)
print("Valleys:", valleys)

repetitions, original_lengths = segment_repetitions_from_valleys(
    df=df,
    valleys=valleys,
    channels=SENSOR_CHANNELS,
    target_length=128
)

print("Repetitions shape:", repetitions.shape)
print("Original lengths:", original_lengths)
print("Original lengths shape:", original_lengths.shape)


