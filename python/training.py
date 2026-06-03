
import sys 
from pathlib import Path
sys.path.append(str(Path().resolve().parents[1]))

BASE_DIR = Path().resolve()
PROJECT_ROOT = BASE_DIR.parent   #to go un up the the root
DATA_DIR = PROJECT_ROOT / "data" 



from dataset_creation import (all_subjects, 
                                    X_ex,X_seq, y, X_info, all_active_units, inputs_seq                                       
                                    )



from utilities.model import  CNNMLPModel





import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_unit_name(unit):
    unit = str(unit)

    if unit.startswith("u"):
        return unit

    if unit.endswith(".0"):
        unit = unit[:-2]

    return f"u{unit}"


def make_most_active_sensor_version(X_seq, active_units, inputs_seq):
    """
    X_seq shape: (N, features, time)

    Keeps only the channels belonging to the most active unit.
    Example: u2 keeps acc_mag_u2, gyr_mag_u2, mag_mag_u2.
    """
    X_masked = torch.zeros_like(X_seq)

    for i, unit in enumerate(active_units):
        unit = normalize_unit_name(unit)

        keep_indices = [
            j for j, col in enumerate(inputs_seq)
            if col.endswith(f"_{unit}")
        ]

        if len(keep_indices) == 0:
            raise ValueError(
                f"No columns found for active unit {unit}. "
                f"Check most_active_unit values and inputs_seq."
            )

        X_masked[i, keep_indices, :] = X_seq[i, keep_indices, :]

        return X_masked



def normalize_seq_train_val(X_train, X_val):
    """
    X_train, X_val shape: (N, channels, time)
    Normalizes using training-set statistics only.
    """
    mean = X_train.mean(dim=(0, 2), keepdim=True)
    std = X_train.std(dim=(0, 2), keepdim=True) + 1e-6

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm


def normalize_info_train_val(X_info_train, X_info_val):
    """
    X_info columns assumed:
        0 = gender
        1 = age
        2 = weight
        3 = height
        4 = original_rep_length

    Normalizes only numeric continuous columns.
    Leaves gender unchanged.
    """
    X_info_train = X_info_train.clone()
    X_info_val = X_info_val.clone()

    numeric_cols = [1, 2, 3, 4]

    mean = X_info_train[:, numeric_cols].mean(dim=0, keepdim=True)
    std = X_info_train[:, numeric_cols].std(dim=0, keepdim=True) + 1e-6

    X_info_train[:, numeric_cols] = (
        X_info_train[:, numeric_cols] - mean
    ) / std

    X_info_val[:, numeric_cols] = (
        X_info_val[:, numeric_cols] - mean
    ) / std

    return X_info_train, X_info_val

val_subject = "s2"   # subject/patient left out for validation

train_indices = [
    i for i, s in enumerate(all_subjects)
    if s != val_subject
]

val_indices = [
    i for i, s in enumerate(all_subjects)
    if s == val_subject
]

# 0. Split data according to subject
X_seq_train = X_seq[train_indices]
X_seq_val   = X_seq[val_indices]

X_ex_train = X_ex[train_indices]
X_ex_val   = X_ex[val_indices]

X_info_train = X_info[train_indices]
X_info_val   = X_info[val_indices]

y_train = y[train_indices]
y_val   = y[val_indices]

# Active units must also be split, if they are sample-level
train_active_units = [all_active_units[i] for i in train_indices]
val_active_units   = [all_active_units[i] for i in val_indices]

# 1. Normalize using training-fold statistics only
X_seq_train, X_seq_val = normalize_seq_train_val(
    X_seq_train,
    X_seq_val
)

X_info_train, X_info_val = normalize_info_train_val(
    X_info_train,
    X_info_val
)

# Usually X_ex should NOT be normalized if it is one-hot exercise encoding.
# Normalize X_ex only if it contains continuous numerical features.

# 2. Create most-active-sensor-only version of TRAINING data only
X_seq_train_masked = make_most_active_sensor_version(
    X_seq_train,
    train_active_units,
    inputs_seq
)

# 3. Double the training set
X_seq_train_aug = torch.cat(
    [X_seq_train, X_seq_train_masked],
    dim=0
)

X_ex_train_aug = torch.cat(
    [X_ex_train, X_ex_train],
    dim=0
)

X_info_train_aug = torch.cat(
    [X_info_train, X_info_train],
    dim=0
)

y_train_aug = torch.cat(
    [y_train, y_train],
    dim=0
)


train_dataset = TensorDataset(
    X_seq_train_aug,
    X_ex_train_aug,
    X_info_train_aug,
    y_train_aug
)

val_dataset = TensorDataset(
    X_seq_val,
    X_ex_val,
    X_info_val,
    y_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


model = CNNMLPModel(
    input_dim_seq=X_seq_train.shape[1],
    input_dim_exercise=X_ex_train.shape[1],
    input_dim_info=X_info_train.shape[1],
    num_classes=3,
    use_seq=True,
    use_ex=True,
    use_info=True
).to(device)


import copy

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float("inf")
best_preds = None
best_labels = None
best_epoch = None

for epoch in range(20):

    # TRAIN
    model.train()
    running_loss = 0.0

    for x_seq_b, x_ex_b, x_info_b, y_b in train_loader:
        x_seq_b = x_seq_b.to(device)
        x_ex_b = x_ex_b.to(device)
        x_info_b = x_info_b.to(device)
        y_b = y_b.to(device)

       

        optimizer.zero_grad()

        outputs = model(x_seq_b, x_ex_b, x_info_b)
        loss = criterion(outputs, y_b)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_seq_b.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # VALIDATION
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_seq_b, x_ex_b, x_info_b, y_b in val_loader:
            x_seq_b = x_seq_b.to(device)
            x_ex_b = x_ex_b.to(device)
            x_info_b = x_info_b.to(device)
            y_b = y_b.to(device)

            outputs = model(x_seq_b, x_ex_b, x_info_b)
            loss = criterion(outputs, y_b)


            val_running_loss += loss.item() * x_seq_b.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)

    print(
        f"Fold {val_subject} | Epoch {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # keep best epoch for this fold
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_preds = np.array(all_preds)
        best_labels = np.array(all_labels)
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1

# fold-level metrics
fold_acc = accuracy_score(best_labels, best_preds)
fold_f1 = f1_score(best_labels, best_preds, average="macro")
fold_cm = confusion_matrix(best_labels, best_preds)

results = {}

results[val_subject] = {
    "subject": val_subject,
    "best_epoch": best_epoch,
    "val_loss": best_val_loss,
    "accuracy": fold_acc,
    "macro_f1": fold_f1,
    "confusion_matrix": fold_cm,
    "model_state_dict": best_model_state,
    "predictions": best_preds,
    "labels": best_labels
}


torch.save(results, "fold_results.pth")