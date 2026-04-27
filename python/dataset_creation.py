import sys 
from pathlib import Path
sys.path.append(str(Path().resolve().parent))

import torch.nn.functional as F







from utilities.model import CNNmodel_base, CNNMLPModel
import torch
import numpy as np
import pandas as pd


BASE_DIR = Path().resolve()
PROJECT_ROOT = BASE_DIR.parent   #to go un up the the root
DATA_DIR = PROJECT_ROOT / "data" 



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

exercises = ['e1','e2','e3','e4','e5','e6','e7','e8']
subject=['s1', 's2', 's3','s4', 's5']
labels=['correct', 'low_amplitude', 'fast']

for s in  subject:
    for e in exercises:
        for l in labels:


            df_r=df[(df['subject'] == s) & (df['exercise'] == e) & (df['label']==l)]

            
            duration = df_r['time index'].max() - df_r['time index'].min()

            repetition_duration= duration /10

            #print(repetition_duration, duration, l)


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
all_global_features=[]


for s in subject:
    for e in exercises:
        for l in labels:

            df_r = df[
                (df['subject'] == s) &
                (df['exercise'] == e) &
                (df['label'] == l)
            ].copy()

            df_r = df_r.sort_values('time index')

            duration = df_r['time index'].max() - df_r['time index'].min()
            rep_len = int(duration / 10)

            for i in range(10):
                rep = df_r.iloc[i * rep_len:(i + 1) * rep_len]

                X_rep = rep[inputs_seq].to_numpy()              # (time, seq_features)
                g_rep = torch.tensor(rep[inputs_global].iloc[0].to_numpy(), dtype=torch.float32)  # (global_features,)

                all_sequences.append(X_rep)
                all_global_features.append(g_rep)
                all_labels.append(label_to_idx[l])
                all_subjects.append(s)
                all_exercises.append(e)

max_len = max(seq.shape[0] for seq in all_sequences)

padded = []

for seq in all_sequences:
    seq = torch.tensor(seq, dtype=torch.float32)  # (time, features)
    seq = seq.transpose(0, 1)                     # (features, time)

    pad_len = max_len - seq.shape[1]
    seq = F.pad(seq, (0, pad_len))

    padded.append(seq)

X_seq = torch.stack(padded)  # (N, features, max_len)
X_glob=torch.stack(all_global_features)
y = torch.tensor(all_labels, dtype=torch.long)

all_subjects = np.array(all_subjects)
unique_subjects = np.unique(all_subjects)


