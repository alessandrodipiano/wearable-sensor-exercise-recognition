import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.model import CNNMLPModel


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
# Model loading
# ============================================================

IDX_TO_LABEL = {
    0: "correct",
    1: "low_amplitude",
    2: "fast",
}


def build_model(device=None):
    """Load the trained CNN+MLP model and return (model, device)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        PROJECT_ROOT / "fold_results.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(fold_results["s2"]["model_state_dict"])
    model.eval()

    return model, device


# ============================================================
# Inference
# ============================================================

def run_inference(
    repetitions,
    original_lengths,
    gender,
    age,
    weight,
    height,
    exercise,
    unit="u2",
    model=None,
    device=None,
):
    """
    Run the trained model on a batch of detected repetitions.

    Dynamic inputs (set by the app's form):
        gender   : 0 = female, 1 = male
        age      : years
        weight   : kg
        height   : cm
        exercise : one of "e1".."e8"

    Sensor inputs (from rep_detection.record_repetitions):
        repetitions      : array-like (N, 128, 9)
        original_lengths : array-like (N,)

    Returns (preds, probs) as numpy arrays.
    """
    if model is None:
        model, device = build_model(device)
    if device is None:
        device = next(model.parameters()).device

    # (N, 128, 9) -> (N, 9, 128) -> (N, 45, 128)
    X_seq = torch.as_tensor(np.asarray(repetitions), dtype=torch.float32)
    X_seq = X_seq.transpose(1, 2)
    X_seq = expand_single_unit_to_45(X_seq, unit=unit)

    n_reps = X_seq.shape[0]
    if n_reps == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)

    # X_info: (N, 5) = [gender, age, weight, height, original_len]
    X_info = torch.tensor(
        np.array(
            [[gender, age, weight, height, ol] for ol in original_lengths],
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )

    # X_ex: (N, 9) = limb + one-hot exercise, repeated per rep
    exercise_dummy = torch.tensor(make_exercise_dummy(exercise), dtype=torch.float32)
    X_ex = exercise_dummy.unsqueeze(0).repeat(n_reps, 1)

    X_seq = X_seq.to(device)
    X_ex = X_ex.to(device)
    X_info = X_info.to(device)

    with torch.no_grad():
        outputs = model(X_seq, X_ex, X_info)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()


if __name__ == "__main__":
    # Standalone live run: record from the phone, then classify.
    from python.rep_detection import record_repetitions

    repetitions, original_lengths, _ = record_repetitions(plot=True)

    preds_np, probs_np = run_inference(
        repetitions,
        original_lengths,
        gender=0,
        age=22,
        weight=60,
        height=175,
        exercise="e7",
    )

    print("Predictions:", preds_np)
    for i, pred in enumerate(preds_np):
        print(f"Rep {i + 1}: {IDX_TO_LABEL[int(pred)]} | probabilities = {probs_np[i]}")




