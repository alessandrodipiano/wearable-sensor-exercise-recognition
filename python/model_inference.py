import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.model import CNNMLPModel
from python.rep_detection import repetitions, original_lengths


# ============================================================
# User inputs
# ============================================================

gender = 1      # 0 = female, 1 = male
age = 23
weight = 85     # kg
height = 185    # cm

exercise = "e7"   # user input


# ============================================================
# Exercise features: limb + one-hot exercise
# Shape per repetition: (9,)
# ============================================================

EXERCISES = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
ARM_EXERCISES = {"e2", "e6", "e7", "e8"}


def make_exercise_dummy(exercise):
    """
    Returns:
        vector of shape (9,)

    Components:
        [limb, exercise_e1, exercise_e2, ..., exercise_e8]

    limb:
        0 = leg
        1 = arm
    """

    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise: {exercise}")

    x = np.zeros(len(EXERCISES) + 1, dtype=np.float32)

    # limb feature
    x[0] = 1.0 if exercise in ARM_EXERCISES else 0.0

    # one-hot exercise identity
    exercise_idx = EXERCISES.index(exercise)
    x[1 + exercise_idx] = 1.0

    return x


# ============================================================
# Expand live 9-channel sensor input to 45 channels
# ============================================================

def expand_single_unit_to_45(X_seq_9, unit="u2"):
    """
    Converts live single-sensor input to the 45-channel format used in training.

    Input:
        X_seq_9 shape: (N, 9, 128)

    Output:
        X_seq_45 shape: (N, 45, 128)

    Live channel order:
        0 accX
        1 accY
        2 accZ
        3 gyrX
        4 gyrY
        5 gyrZ
        6 magX
        7 magY
        8 magZ

    Training channel order:
        acc_x_u1, acc_x_u2, acc_x_u3, acc_x_u4, acc_x_u5,
        acc_y_u1, acc_y_u2, acc_y_u3, acc_y_u4, acc_y_u5,
        acc_z_u1, acc_z_u2, acc_z_u3, acc_z_u4, acc_z_u5,
        gyr_x_u1, ..., gyr_z_u5,
        mag_x_u1, ..., mag_z_u5
    """

    unit_to_index = {
        "u1": 0,
        "u2": 1,
        "u3": 2,
        "u4": 3,
        "u5": 4,
    }

    if unit not in unit_to_index:
        raise ValueError(f"Unknown unit: {unit}")

    u = unit_to_index[unit]

    N, C, T = X_seq_9.shape

    if C != 9:
        raise ValueError(f"Expected 9 live channels, got {C}")

    X_seq_45 = torch.zeros((N, 45, T), dtype=X_seq_9.dtype)

    # For each live channel, place it into the chosen unit position.
    # Unused unit channels remain zero.
    for live_ch in range(9):
        target_ch = live_ch * 5 + u
        X_seq_45[:, target_ch, :] = X_seq_9[:, live_ch, :]

    return X_seq_45


# ============================================================
# Build tensors
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# repetitions comes from rep_detection
# expected original shape: (N, 128, 9)
repetitions = torch.tensor(repetitions, dtype=torch.float32)

# Convert to Conv1d format: (N, 128, 9) -> (N, 9, 128)
X_seq_val = repetitions.transpose(1, 2)

# Expand from one live unit to 45 training channels
# Choose the unit that represents where the phone/live sensor is worn.
X_seq_val = expand_single_unit_to_45(X_seq_val, unit="u2")


# Build X_info_val: (N, 5)
# Components: [gender, age, weight, height, original_len]
X_info_rows = []

for original_len in original_lengths:
    x_info = np.array(
        [gender, age, weight, height, original_len],
        dtype=np.float32
    )
    X_info_rows.append(x_info)

X_info_val = torch.tensor(
    np.stack(X_info_rows),
    dtype=torch.float32
)


# Build X_ex_val: (N, 9)
exercise_dummy = make_exercise_dummy(exercise)
exercise_dummy = torch.tensor(exercise_dummy, dtype=torch.float32)

n_reps = X_seq_val.shape[0]
X_ex_val = exercise_dummy.unsqueeze(0).repeat(n_reps, 1)


# Sanity checks
print("X_seq_val:", X_seq_val.shape)
print("X_ex_val:", X_ex_val.shape)
print("X_info_val:", X_info_val.shape)

assert X_seq_val.shape[0] == X_ex_val.shape[0] == X_info_val.shape[0]
assert X_seq_val.shape[1] == 45
assert X_ex_val.shape[1] == 9
assert X_info_val.shape[1] == 5


# ============================================================
# Load model
# ============================================================

model = CNNMLPModel(
    input_dim_seq=45,
    input_dim_exercise=9,
    input_dim_info=5,
    num_classes=3,
    use_seq=True,
    use_ex=True,
    use_info=True,
).to(device)

fold_results = torch.load(
    PROJECT_ROOT / "notebooks" / "CNN" / "fold_results.pth",
    map_location=device
)

checkpoint = fold_results["s2"]
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ============================================================
# Inference
# ============================================================

X_seq_val = X_seq_val.to(device)
X_ex_val = X_ex_val.to(device)
X_info_val = X_info_val.to(device)

with torch.no_grad():
    outputs = model(X_seq_val, X_ex_val, X_info_val)

    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)


# ============================================================
# Results
# ============================================================

idx_to_label = {
    0: "correct",
    1: "low_amplitude",
    2: "fast",
}

preds_np = preds.cpu().numpy()
probs_np = probs.cpu().numpy()

print("Predictions:", preds_np)

for i, pred in enumerate(preds_np):
    print(
        f"Rep {i + 1}: "
        f"{idx_to_label[pred]} | "
        f"probabilities = {probs_np[i]}"
    )




