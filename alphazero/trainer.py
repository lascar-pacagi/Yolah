#!/usr/bin/env python3
"""
AlphaZero Training Pipeline with Multi-GPU Support

This script trains the AlphaZero neural network using self-play data.
It supports multi-GPU training with PyTorch DDP (DistributedDataParallel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os
import struct
from pathlib import Path
from tqdm import tqdm
import argparse


class YolahDataset(Dataset):
    """Dataset loader for AlphaZero training examples"""

    def __init__(self, data_file):
        self.data_file = data_file
        self.example_size = (
            3 * 8 +  # 3 uint64_t for bitboards
            2 +      # 1 uint16_t for ply
            75 * 4 + # 75 floats for policy
            4        # 1 float for value
        )

        # Calculate number of examples
        file_size = os.path.getsize(data_file)
        self.num_examples = file_size // self.example_size

        # Memory map for efficient loading
        self.data = np.memmap(data_file, dtype=np.uint8, mode='r')

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        offset = idx * self.example_size
        example_bytes = self.data[offset:offset + self.example_size]

        # Parse game state
        black = struct.unpack('Q', example_bytes[0:8])[0]
        white = struct.unpack('Q', example_bytes[8:16])[0]
        empty = struct.unpack('Q', example_bytes[16:24])[0]
        ply = struct.unpack('H', example_bytes[24:26])[0]

        # Convert bitboards to planes
        state = self._bitboards_to_planes(black, white, empty)

        # Parse policy
        policy_offset = 26
        policy = np.frombuffer(
            example_bytes[policy_offset:policy_offset + 75*4],
            dtype=np.float32
        )

        # Parse value
        value_offset = policy_offset + 75*4
        value = np.frombuffer(
            example_bytes[value_offset:value_offset + 4],
            dtype=np.float32
        )[0]

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(policy).float(),
            torch.tensor(value, dtype=torch.float32)
        )

    @staticmethod
    def _bitboards_to_planes(black, white, empty):
        """Convert bitboards to 3-plane representation"""
        planes = np.zeros((3, 8, 8), dtype=np.float32)

        for sq in range(64):
            row = sq // 8
            col = sq % 8
            mask = 1 << sq

            if black & mask:
                planes[0, row, col] = 1.0
            elif white & mask:
                planes[1, row, col] = 1.0
            elif empty & mask:
                planes[2, row, col] = 1.0

        return planes


class ResidualBlock(nn.Module):
    """Residual block for AlphaZero network"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network architecture"""

    def __init__(self, input_channels=3, num_residual_blocks=10, num_filters=256):
        super().__init__()

        # Initial convolution
        self.conv_initial = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 75)  # 75 max moves

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual tower
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value.squeeze(1)


class AlphaZeroLoss(nn.Module):
    """Combined loss for AlphaZero: policy loss + value loss"""

    def __init__(self, value_weight=1.0):
        super().__init__()
        self.value_weight = value_weight

    def forward(self, policy_pred, value_pred, policy_target, value_target):
        # Policy loss: cross-entropy
        policy_loss = -torch.mean(torch.sum(policy_target * policy_pred, dim=1))

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, value_target)

        # Combined loss
        total_loss = policy_loss + self.value_weight * value_loss

        return total_loss, policy_loss, value_loss


def ddp_setup(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    """AlphaZero trainer with multi-GPU support"""

    def __init__(self, gpu_id, model, train_loader, optimizer, scheduler,
                 loss_fn, epochs, save_every, checkpoint_dir):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)

        # Wrap model with DDP
        self.model = DDP(model, device_ids=[gpu_id])

    def train(self):
        """Training loop"""
        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            if self.gpu_id == 0:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            else:
                pbar = self.train_loader

            for state, policy_target, value_target in pbar:
                state = state.to(self.gpu_id)
                policy_target = policy_target.to(self.gpu_id)
                value_target = value_target.to(self.gpu_id)

                # Forward pass
                policy_pred, value_pred = self.model(state)

                # Compute loss
                loss, policy_loss, value_loss = self.loss_fn(
                    policy_pred, value_pred, policy_target, value_target
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Accumulate losses
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

                if self.gpu_id == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'policy': f'{policy_loss.item():.4f}',
                        'value': f'{value_loss.item():.4f}'
                    })

            # Update learning rate
            self.scheduler.step()

            # Log epoch statistics
            if self.gpu_id == 0:
                avg_loss = total_loss / num_batches
                avg_policy_loss = total_policy_loss / num_batches
                avg_value_loss = total_value_loss / num_batches

                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Policy Loss: {avg_policy_loss:.4f}")
                print(f"  Value Loss: {avg_value_loss:.4f}")
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

                # Save checkpoint
                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(epoch + 1)

        # Save final model
        if self.gpu_id == 0:
            self.save_checkpoint(self.epochs)

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")


def train_worker(rank, world_size, args):
    """Worker function for distributed training"""
    ddp_setup(rank, world_size)

    # Create dataset and dataloader
    dataset = YolahDataset(args.data_file)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    model = AlphaZeroNet(
        num_residual_blocks=args.num_blocks,
        num_filters=args.num_filters
    )

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{rank}')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank == 0:
            print(f"Loaded checkpoint from epoch {start_epoch}")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # Load optimizer and scheduler state if checkpoint exists
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Create loss function
    loss_fn = AlphaZeroLoss(value_weight=args.value_weight)

    # Create trainer and train
    trainer = Trainer(
        gpu_id=rank,
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=args.epochs,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.train()

    destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training Pipeline')

    # Data arguments
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Model arguments
    parser.add_argument('--num-blocks', type=int, default=10,
                       help='Number of residual blocks')
    parser.add_argument('--num-filters', type=int, default=256,
                       help='Number of filters in conv layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=20,
                       help='Learning rate scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                       help='Learning rate scheduler gamma')
    parser.add_argument('--value-weight', type=float, default=1.0,
                       help='Weight for value loss')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')

    args = parser.parse_args()

    # Get number of GPUs
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")

    # Launch distributed training
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
