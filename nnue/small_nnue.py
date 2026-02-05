#!/usr/bin/env python3
"""
Small NNUE training script using heuristic features as input.

This network takes precomputed heuristic features (24 floats) instead of
raw board positions (193 bits). The heuristics encode strategic knowledge
that would otherwise require many more weights to learn.

Architecture: 24 -> 32 -> 16 -> 3 (softmax: black_win, draw, white_win)
Total weights: ~700 (vs ~200k for full NNUE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from pathlib import Path

# Feature names for documentation
FEATURE_NAMES = [
    "MATERIAL_DIFF", "SCORE_DIFF", "MOBILITY_DIFF", "FREE_SQUARES",
    "CENTER_CONTROL", "INNER_CENTER", "EXTENDED_CENTER",
    "EDGE_PIECES", "CORNER_PIECES", "NEAR_CORNER",
    "BLOCKED_PIECES", "FRONTIER_PIECES", "GROUPS_DIFF", "TENSION",
    "CONNECTIVITY_DIFF", "CONNECTIVITY_SET_DIFF", "ALONE_DIFF",
    "FIRST_DIFF", "INFLUENCE_DIFF",
    "FREEDOM_LOW", "FREEDOM_MID", "FREEDOM_HIGH",
    "GAME_PHASE", "TURN"
]

NB_FEATURES = 24
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 3

# Quantization factor (same as full NNUE)
FACTOR = 64


class SmallNNUE(nn.Module):
    """
    Small neural network that learns to combine heuristic features.

    The network uses clipped ReLU (ClippedReLU6 / min(max(x,0),1)) for
    compatibility with int8 quantization. The output uses softmax for
    win/draw/loss probabilities.
    """

    def __init__(self):
        super(SmallNNUE, self).__init__()

        # Input normalization (learned)
        self.input_norm = nn.BatchNorm1d(NB_FEATURES)

        # Hidden layers
        self.fc1 = nn.Linear(NB_FEATURES, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc3 = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize input features
        x = self.input_norm(x)

        # Hidden layers with clipped ReLU
        x = torch.clamp(torch.relu(self.fc1(x)), max=1.0)
        x = torch.clamp(torch.relu(self.fc2(x)), max=1.0)

        # Output (raw logits for cross-entropy loss)
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        """Get win/draw/loss probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class SmallNNUENoNorm(nn.Module):
    """
    Version without batch normalization for easier quantization.
    Features should be pre-normalized before training.
    """

    def __init__(self):
        super(SmallNNUENoNorm, self).__init__()
        self.fc1 = nn.Linear(NB_FEATURES, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc3 = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.clamp(torch.relu(self.fc1(x)), max=1.0)
        x = torch.clamp(torch.relu(self.fc2(x)), max=1.0)
        x = self.fc3(x)
        return x


class HeuristicFeaturesDataset(Dataset):
    """
    Dataset for training the small NNUE.

    Expected file format (text):
    - Each line: 24 floats (features) followed by 3 floats (probabilities)
    - Features are space-separated
    - Probabilities are: P(black_win), P(draw), P(white_win)

    Alternative: binary format for faster loading
    """

    def __init__(self, filepath, binary=False):
        self.filepath = filepath

        if binary:
            data = np.load(filepath)
            self.features = torch.tensor(data['features'], dtype=torch.float32)
            self.targets = torch.tensor(data['targets'], dtype=torch.float32)
        else:
            self._load_text(filepath)

        # Compute normalization statistics
        self.feature_mean = self.features.mean(dim=0)
        self.feature_std = self.features.std(dim=0) + 1e-6

    def _load_text(self, filepath):
        features_list = []
        targets_list = []

        with open(filepath, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                if len(values) == NB_FEATURES + 3:
                    features_list.append(values[:NB_FEATURES])
                    targets_list.append(values[NB_FEATURES:])

        self.features = torch.tensor(features_list, dtype=torch.float32)
        self.targets = torch.tensor(targets_list, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_normalized(self):
        """Return normalized features for training without BatchNorm."""
        return (self.features - self.feature_mean) / self.feature_std


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        # Cross-entropy with soft targets
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

        # Accuracy (argmax match)
        pred = outputs.argmax(dim=1)
        target_class = targets.argmax(dim=1)
        correct += (pred == target_class).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


def validate_model(model, dataloader, criterion, device):
    model.train(False)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * features.size(0)

            pred = outputs.argmax(dim=1)
            target_class = targets.argmax(dim=1)
            correct += (pred == target_class).sum().item()
            total += features.size(0)

    return total_loss / total, correct / total


def quantize_and_export(model, output_path, feature_mean=None, feature_std=None):
    """
    Export model weights in quantized format for C++ inference.

    Format:
    - Normalization params (if provided): mean and std for each feature
    - For each layer: W m n (weights), B n (bias)
    - Weights scaled to int8 range [-127, 127]
    - Biases scaled to int16
    """

    with open(output_path, 'w') as f:
        # Write normalization parameters if provided
        if feature_mean is not None and feature_std is not None:
            f.write(f"NORM {NB_FEATURES}\n")
            f.write(" ".join(f"{v:.6f}" for v in feature_mean.numpy()) + "\n")
            f.write(" ".join(f"{v:.6f}" for v in feature_std.numpy()) + "\n")

        # Quantize and write each layer
        layers = [
            ('fc1', model.fc1, HIDDEN1_SIZE, NB_FEATURES),
            ('fc2', model.fc2, HIDDEN2_SIZE, HIDDEN1_SIZE),
            ('fc3', model.fc3, OUTPUT_SIZE, HIDDEN2_SIZE),
        ]

        for name, layer, out_size, in_size in layers:
            weight = layer.weight.detach().cpu()
            bias = layer.bias.detach().cpu()

            # Scale weights to int8 range
            w_scale = 127.0 / (weight.abs().max() + 1e-6)
            w_quantized = torch.clamp(torch.round(weight * w_scale), -127, 127).to(torch.int8)

            # Scale bias to int16 range (accounting for weight scaling and FACTOR)
            b_scale = FACTOR * w_scale
            b_quantized = torch.clamp(torch.round(bias * b_scale), -32767, 32767).to(torch.int16)

            # Write weight matrix
            f.write(f"W {out_size} {in_size}\n")
            for i in range(out_size):
                f.write(" ".join(str(int(v)) for v in w_quantized[i].numpy()) + "\n")

            # Write bias vector
            f.write(f"B {out_size}\n")
            f.write(" ".join(str(int(v)) for v in b_quantized.numpy()) + "\n")

            # Write scale factor for this layer
            f.write(f"S {w_scale:.6f}\n")

    print(f"Quantized model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train small NNUE with heuristic features')
    parser.add_argument('--train', type=str, required=True, help='Training data file')
    parser.add_argument('--val', type=str, help='Validation data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='small_nnue_quantized.txt',
                        help='Output file for quantized weights')
    parser.add_argument('--checkpoint', type=str, default='small_nnue.pt',
                        help='Checkpoint file')
    parser.add_argument('--no-norm', action='store_true',
                        help='Use version without BatchNorm')
    parser.add_argument('--binary', action='store_true',
                        help='Load data from binary .npz format')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading training data from {args.train}...")
    train_dataset = HeuristicFeaturesDataset(args.train, binary=args.binary)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if args.val:
        print(f"Loading validation data from {args.val}...")
        val_dataset = HeuristicFeaturesDataset(args.val, binary=args.binary)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    if args.no_norm:
        model = SmallNNUENoNorm()
    else:
        model = SmallNNUE()
    model = model.to(device)

    print(f"Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        scheduler.step()

        log = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"

        if val_loader:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            log += f" - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.checkpoint)
                log += " *"
        else:
            torch.save(model.state_dict(), args.checkpoint)

        print(log)

    # Load best model and export
    if val_loader:
        model.load_state_dict(torch.load(args.checkpoint))

    # Export quantized weights
    if args.no_norm:
        quantize_and_export(model, args.output,
                           train_dataset.feature_mean,
                           train_dataset.feature_std)
    else:
        quantize_and_export(model, args.output)

    print("Training complete!")


if __name__ == '__main__':
    main()
