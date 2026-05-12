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

##########################################################àà


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







inputs_seq = [
    'acc_mag_u1', 'gyr_mag_u1', 'mag_mag_u1',
    'acc_mag_u2', 'gyr_mag_u2', 'mag_mag_u2',
    'acc_mag_u3', 'gyr_mag_u3', 'mag_mag_u3',
    'acc_mag_u4', 'gyr_mag_u4', 'mag_mag_u4',
    'acc_mag_u5', 'gyr_mag_u5', 'mag_mag_u5'
]

inputs_global = [
    'limb',
    'exercise_e1', 'exercise_e2', 'exercise_e3', 'exercise_e4',
    'exercise_e5', 'exercise_e6', 'exercise_e7', 'exercise_e8'


]


label_to_idx = {
    'correct': 0,
    'low_amplitude': 1,
    'fast': 2
}



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

unique_subjects = np.unique(all_subjects)


            



