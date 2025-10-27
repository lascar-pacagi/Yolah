#!/usr/bin/env python3
"""
Export PyTorch AlphaZero model to binary format for C++ inference

This script converts a trained PyTorch model to a binary format that can be
loaded by the C++ inference engine.
"""

import torch
import argparse
import struct
from trainer import AlphaZeroNet


def export_model_to_binary(checkpoint_path, output_path):
    """Export PyTorch model to binary format for C++ inference"""

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    # Create model and load weights
    model = AlphaZeroNet()
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    print(f"Exporting to {output_path}...")

    # Open output file
    with open(output_path, 'wb') as f:
        # Export each layer's weights and biases
        for name, param in model.named_parameters():
            print(f"  Exporting {name}: {param.shape}")

            # Convert to float32 and flatten
            weights = param.detach().cpu().float().numpy().flatten()

            # Write weights as binary float32
            for value in weights:
                f.write(struct.pack('f', value))

    print(f"\nExport complete!")
    print(f"Binary file size: {output_path.stat().st_size / (1024**2):.2f} MB")


def export_model_to_text(checkpoint_path, output_path):
    """Export PyTorch model to text format for debugging"""

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    # Create model and load weights
    model = AlphaZeroNet()
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    print(f"Exporting to {output_path}...")

    # Open output file
    with open(output_path, 'w') as f:
        # Export each layer's weights and biases
        for name, param in model.named_parameters():
            f.write(f"# {name} {list(param.shape)}\n")

            # Convert to float32 and write
            weights = param.detach().cpu().float().numpy().flatten()

            for i, value in enumerate(weights):
                f.write(f"{value:.8f}\n")
                if i > 0 and i % 10 == 0:
                    f.write("\n")

            f.write("\n\n")

    print(f"\nExport complete!")


def print_model_info(checkpoint_path):
    """Print model architecture and parameter count"""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    model = AlphaZeroNet()
    model.load_state_dict(model_state)

    print(f"\nModel Information:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"\nArchitecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size (FP32): {total_params * 4 / (1024**2):.2f} MB")

    print(f"\nLayer Details:")
    for name, param in model.named_parameters():
        print(f"  {name:50s} {str(list(param.shape)):30s} {param.numel():>10,}")


def main():
    parser = argparse.ArgumentParser(description='Export AlphaZero model for C++ inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for binary weights')
    parser.add_argument('--format', type=str, default='binary',
                       choices=['binary', 'text'],
                       help='Export format (binary or text)')
    parser.add_argument('--info', action='store_true',
                       help='Print model information')

    args = parser.parse_args()

    # Print info if requested
    if args.info:
        print_model_info(args.checkpoint)
        return

    # Export model
    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'binary':
        export_model_to_binary(args.checkpoint, output_path)
    else:
        export_model_to_text(args.checkpoint, output_path)

    print(f"\nTo use in C++:")
    print(f"  AlphaZeroNetwork network;")
    print(f"  network.initialize(\"{output_path}\");")


if __name__ == '__main__':
    main()
