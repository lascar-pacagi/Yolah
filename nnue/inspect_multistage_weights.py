"""
Script to inspect the structure of MultiStageNet weights.
Shows how weights are organized for each of the three networks.
"""

import torch

def inspect_multistage_checkpoint(checkpoint_path):
    """Inspect the structure of a MultiStageNet checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    print("=" * 70)
    print(f"Inspecting: {checkpoint_path}")
    print("=" * 70)

    # Group by network
    networks = {'opening_net': [], 'middle_net': [], 'end_net': []}

    for key in state_dict.keys():
        for net_name in networks.keys():
            if key.startswith(net_name):
                networks[net_name].append(key)

    # Print structure for each network
    for net_name, keys in networks.items():
        print(f"\n{net_name.upper()}:")
        print("-" * 70)
        for key in sorted(keys):
            tensor = state_dict[key]
            print(f"  {key:<40} shape: {str(tensor.shape):<20} dtype: {tensor.dtype}")

    print("\n" + "=" * 70)
    print("WEIGHT ORGANIZATION FOR C++ IMPLEMENTATION")
    print("=" * 70)

    # Show what each network contains
    print("\nEach network (opening/middle/end) contains:")
    print("  fc1.weight: [1024, 193]  - First layer weights")
    print("  fc1.bias:   [1024]       - First layer biases")
    print("  fc2.weight: [64, 1024]   - Second layer weights")
    print("  fc2.bias:   [64]         - Second layer biases")
    print("  fc3.weight: [32, 64]     - Third layer weights")
    print("  fc3.bias:   [32]         - Third layer biases")
    print("  fc4.weight: [3, 32]      - Output layer weights")
    print("  fc4.bias:   [3]          - Output layer biases")

    print("\nTo use in C++:")
    print("  1. Load one of: opening_net, middle_net, or end_net")
    print("  2. Access weights as: state_dict['fc1.weight'], etc.")
    print("  3. Convert tensors to your C++ format")

    # Show example of accessing weights
    print("\n" + "=" * 70)
    print("EXAMPLE: Accessing Opening Network Weights")
    print("=" * 70)

    for layer in ['fc1', 'fc2', 'fc3', 'fc4']:
        weight_key = f'opening_net.{layer}.weight'
        bias_key = f'opening_net.{layer}.bias'

        if weight_key in state_dict:
            weight = state_dict[weight_key]
            bias = state_dict[bias_key]
            print(f"\n{layer}:")
            print(f"  Weight shape: {weight.shape}")
            print(f"  Bias shape:   {bias.shape}")
            print(f"  Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            print(f"  Bias range:   [{bias.min().item():.6f}, {bias.max().item():.6f}]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inspect_multistage_weights.py <checkpoint_path>")
        print("\nExample:")
        print("  python inspect_multistage_weights.py nnue_multistage_1024x64x32x3.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    inspect_multistage_checkpoint(checkpoint_path)
