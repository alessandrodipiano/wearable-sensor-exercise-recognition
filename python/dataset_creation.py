import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from utilities.src import subjects, exercises, labels, detect_peaks_and_valleys_clean



import numpy as np
import torch


def resample_sequence(seq, target_len=128):
    """
    seq shape: (time, features)
    returns:   (target_len, features)
    """
    seq = np.asarray(seq, dtype=np.float32)

    old_len = seq.shape[0]

    if old_len == target_len:
        return seq

    old_time = np.linspace(0.0, 1.0, old_len)
    new_time = np.linspace(0.0, 1.0, target_len)

    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)

    for j in range(seq.shape[1]):
        resampled[:, j] = np.interp(new_time, old_time, seq[:, j])

    return resampled

########################################################## general dataset with timeseries and exercise type 


df=pd.read_csv(DATA_DIR / 'processed_data.csv')

limb_map = {
    "e1": 0, #leg
    "e2": 1, #arm
    "e3": 0,
    "e4": 0,
    "e5": 0,
    "e6": 1,
    "e7": 1,
    "e8": 1
}

df["limb"] = df["exercise"].map(limb_map)
df['limb'] = df['limb'].astype(np.float32)

exercise_dummies = pd.get_dummies(df['exercise'], prefix='exercise').astype(np.float32)
df = pd.concat([df, exercise_dummies], axis=1)


################################################## subject info dataframe 


subject_info = pd.DataFrame({
    "subject": ["s1", "s2", "s3", "s4", "s5"],
    "gender": ["Female", "Male", "Male", "Female", "Male"],
    "age": [55, 61, 23, 48, 53],
    "weight": [73, 85, 95, 55, 98],
    "height": [169, 180, 180, 158, 175],
})
subject_info["gender"] = subject_info["gender"].map({"Female": 0, "Male": 1})

inputs_subject_info = [
    "gender",
    "age",
    "weight",
    "height",
]

subject_info_dict = {}

for _, row in subject_info.iterrows():
    subject_info_dict[row["subject"]] = row[inputs_subject_info].to_numpy(
        dtype=np.float32
    )
#####################################################

import numpy as np
valleys_indexes={}
for s in subjects:
    for e in exercises:
        for l in labels:

            used = df[
                (df['subject'] == s) &
                (df['exercise'] == e) &
                (df['label'] == l)
            ]

            u = used['most_active_unit'].iloc[0]
            time = used["time index"].to_numpy()

            peaks, valleys, info = detect_peaks_and_valleys_clean(
                                                used=used,
                                                u=u,
                                                expected_reps=10,
                                                max_valleys=11,
                                                peak_prominence=0.5,
                                                valley_prominence=0.05,
                                                plot=False
                                            )
            
            key=f'{s}-{e}-{l}'

            valleys_indexes[key] = time[valleys]

            print(
                f"subject={s}, exercise={e}, label={l}, "
                f"unit={u}, axis={info['axis']}, valleys={info['n_valleys']}"
            )

valleys_indexes["s2-e1-correct"] = np.insert(valleys_indexes["s2-e1-correct"], 0, 1)
valleys_indexes["s3-e1-correct"] = np.insert(valleys_indexes["s3-e1-correct"], 0, 1)

df["rep"] = np.nan

for key, boundaries in valleys_indexes.items():
    s, e, label = key.split("-")

    boundaries = np.asarray(boundaries, dtype=int)
    boundaries = np.sort(boundaries)

    base_mask = (
        (df["subject"] == s) &
        (df["exercise"] == e) &
        (df["label"] == label)
    )

    if base_mask.sum() == 0:
        continue

    for rep_id, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
        start=1
    ):
        if rep_id > 10:
            break

        rep_mask = (
            base_mask &
            (df["time index"] >= start) &
            (df["time index"] < end)
        )

        df.loc[rep_mask, "rep"] = rep_id


df = df.dropna(subset=["rep"]).copy()





inputs_seq=['acc_x_u1', 'acc_x_u2', 'acc_x_u3',
       'acc_x_u4', 'acc_x_u5', 'acc_y_u1', 'acc_y_u2', 'acc_y_u3', 'acc_y_u4',
       'acc_y_u5', 'acc_z_u1', 'acc_z_u2', 'acc_z_u3', 'acc_z_u4', 'acc_z_u5',
       'gyr_x_u1', 'gyr_x_u2', 'gyr_x_u3', 'gyr_x_u4', 'gyr_x_u5', 'gyr_y_u1',
       'gyr_y_u2', 'gyr_y_u3', 'gyr_y_u4', 'gyr_y_u5', 'gyr_z_u1', 'gyr_z_u2',
       'gyr_z_u3', 'gyr_z_u4', 'gyr_z_u5', 'mag_x_u1', 'mag_x_u2', 'mag_x_u3',
       'mag_x_u4', 'mag_x_u5', 'mag_y_u1', 'mag_y_u2', 'mag_y_u3', 'mag_y_u4',
       'mag_y_u5', 'mag_z_u1', 'mag_z_u2', 'mag_z_u3', 'mag_z_u4', 'mag_z_u5',]

'''inputs_seq = [
    'acc_mag_u1', 'gyr_mag_u1', 'mag_mag_u1',
    'acc_mag_u2', 'gyr_mag_u2', 'mag_mag_u2',
    'acc_mag_u3', 'gyr_mag_u3', 'mag_mag_u3',
    'acc_mag_u4', 'gyr_mag_u4', 'mag_mag_u4',
    'acc_mag_u5', 'gyr_mag_u5', 'mag_mag_u5'
]'''

inputs_exercise = [
    "limb",
    "exercise_e1", "exercise_e2", "exercise_e3", "exercise_e4",
    "exercise_e5", "exercise_e6", "exercise_e7", "exercise_e8",
]

inputs_subject_info = [
    "gender",
    "age",
    "weight",
    "height",
]


label_to_idx = {
    'correct': 0,
    'low_amplitude': 1,
    'fast': 2
}



target_len = 128

all_sequences = []
all_labels = []
all_subjects = []
all_exercises = []
all_exercise_features = []
all_subject_info_features = []
all_rep_lengths = []
all_rep_ids = []
all_active_units = []

for s in subjects:
    for e in exercises:
        for l in labels:

            df_trial = df[
                (df["subject"] == s) &
                (df["exercise"] == e) &
                (df["label"] == l)
            ].copy()

            if df_trial.empty:
                continue

            df_trial = df_trial.sort_values("time index")

            for rep_id in range(1, 11):

                rep = df_trial[df_trial["rep"] == rep_id].copy()

                if rep.empty:
                    continue

                rep = rep.sort_values("time index")

                # Original variable-length sequence
                X_rep_raw = rep[inputs_seq].to_numpy(dtype=np.float32)
                original_len = X_rep_raw.shape[0]

                # Resample instead of padding
                X_rep = resample_sequence(X_rep_raw, target_len=target_len)

                # Exercise/limb features
                x_ex_rep = rep[inputs_exercise].iloc[0].to_numpy(
                    dtype=np.float32
                )

                # Subject info
                x_info_rep = subject_info_dict[s]

                # Add original repetition length to info branch
                x_info_rep = np.concatenate([
                    x_info_rep,
                    np.array([original_len], dtype=np.float32)
                ])

                # Store most active unit for augmentation later
                active_unit = rep["most_active_unit"].iloc[0]

                all_sequences.append(X_rep)
                all_exercise_features.append(
                    torch.tensor(x_ex_rep, dtype=torch.float32)
                )
                all_subject_info_features.append(
                    torch.tensor(x_info_rep, dtype=torch.float32)
                )

                all_rep_lengths.append(original_len)
                all_labels.append(label_to_idx[l])
                all_subjects.append(s)
                all_exercises.append(e)
                all_rep_ids.append(rep_id)
                all_active_units.append(active_unit)

# ---------------------------------------------------------
# Final tensors
# ---------------------------------------------------------

X_seq = torch.tensor(np.stack(all_sequences), dtype=torch.float32)
X_seq = X_seq.transpose(1, 2)  # (N, seq_features, target_len)

X_ex = torch.stack(all_exercise_features)       # (N, 9)
X_info = torch.stack(all_subject_info_features) # (N, 5)
y = torch.tensor(all_labels, dtype=torch.long)

all_subjects = np.array(all_subjects)
all_exercises = np.array(all_exercises)
all_rep_ids = np.array(all_rep_ids)
all_active_units = np.array(all_active_units)
all_rep_lengths = np.array(all_rep_lengths)

unique_subjects = np.unique(all_subjects)

print("X_seq:", X_seq.shape)
print("X_ex:", X_ex.shape)
print("X_info:", X_info.shape)
print("y:", y.shape)
print("all_active_units:", all_active_units.shape)
print("all_rep_lengths:", all_rep_lengths.shape)

























'''
all_sequences = []
all_labels = []
all_subjects = []
all_exercises = []
all_global_features = []
all_rep_ids = []

for s in subjects:   
    for e in exercises:
        for l in labels:

            df_trial = df[
                (df["subject"] == s) &
                (df["exercise"] == e) &
                (df["label"] == l)
            ].copy()

            if df_trial.empty:
                continue

            df_trial = df_trial.sort_values("time index")

            
            for rep_id in range(1, 11):

                rep = df_trial[df_trial["rep"] == rep_id].copy()

                
                if rep.empty:
                    continue

                rep = rep.sort_values("time index")

                X_rep = rep[inputs_seq].to_numpy()   # (time, seq_features)

                g_rep = torch.tensor(
                    rep[inputs_global].iloc[0].to_numpy(),
                    dtype=torch.float32
                )

                all_sequences.append(X_rep)
                all_global_features.append(g_rep)
                all_labels.append(label_to_idx[l])
                all_subjects.append(s)
                all_exercises.append(e)
                all_rep_ids.append(rep_id)

max_len = max(seq.shape[0] for seq in all_sequences)

padded = []

for seq in all_sequences:
    seq = torch.tensor(seq, dtype=torch.float32)  # (time, features)
    seq = seq.transpose(0, 1)                     # (features, time)

    pad_len = max_len - seq.shape[1]
    seq = F.pad(seq, (0, pad_len))                # pad time dimension

    padded.append(seq)

X_seq = torch.stack(padded)        # (N, features, max_len)
X_glob = torch.stack(all_global_features)



y = torch.tensor(all_labels, dtype=torch.long)

all_subjects = np.array(all_subjects)
all_exercises = np.array(all_exercises)
all_rep_ids = np.array(all_rep_ids)

unique_subjects = np.unique(all_subjects)'''


            



