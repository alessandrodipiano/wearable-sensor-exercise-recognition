import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



class CNNmodel_base(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
    




class CNNMLPModel(nn.Module):
    def __init__(self, input_dim_seq, input_dim_global, num_classes):
        super().__init__()

        # CNN branch for sequential features
        self.features = nn.Sequential(
            nn.Conv1d(input_dim_seq, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        # MLP branch for global features
        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim_global, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )

        # Final classifier after concatenation
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_global):
        # x_seq shape: (B, input_dim_seq, T)
        # x_global shape: (B, input_dim_global)

        # CNN branch
        h_seq = self.features(x_seq)            # (B, 256, 1)
        h_seq = torch.flatten(h_seq, start_dim=1)   # (B, 256)

        # MLP branch
        h_global = self.global_mlp(x_global)    # (B, 32)

        # Concatenate
        h = torch.cat([h_seq, h_global], dim=1) # (B, 288)

        # Final output
        out = self.classifier(h)                # (B, num_classes)
        return out
