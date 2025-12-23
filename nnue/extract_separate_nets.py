"""
Script to extract the three separate neural networks from a MultiStageNet checkpoint.
Each network (opening, middle, end) will be saved as a separate .pt file.
"""

import torch
from nnue_multi_stage_multigpu import MultiStageNet, Net, INPUT_SIZE

def extract_networks(checkpoint_path, output_prefix="net"):
    """
    Extract the three separate networks from a MultiStageNet checkpoint.

    Args:
        checkpoint_path: Path to the MultiStageNet checkpoint file
        output_prefix: Prefix for output files (default: "net")

    Saves:
        {output_prefix}_opening.pt - Opening network weights
        {output_prefix}_middle.pt - Middle game network weights
        {output_prefix}_end.pt - End game network weights
    """
    # Load the full multi-stage model state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    print("Keys in state dict:")
    for key in state_dict.keys():
        print(f"  {key}")
    print()

    # Extract weights for each network
    opening_state = {}
    middle_state = {}
    end_state = {}

    for key, value in state_dict.items():
        if key.startswith('opening_net.'):
            # Remove 'opening_net.' prefix
            new_key = key.replace('opening_net.', '')
            opening_state[new_key] = value
        elif key.startswith('middle_net.'):
            # Remove 'middle_net.' prefix
            new_key = key.replace('middle_net.', '')
            middle_state[new_key] = value
        elif key.startswith('end_net.'):
            # Remove 'end_net.' prefix
            new_key = key.replace('end_net.', '')
            end_state[new_key] = value

    # Verify we got all the weights
    print(f"Opening network has {len(opening_state)} parameter tensors")
    print(f"Middle network has {len(middle_state)} parameter tensors")
    print(f"End network has {len(end_state)} parameter tensors")
    print()

    # Save each network separately
    torch.save(opening_state, f"{output_prefix}_opening.pt")
    torch.save(middle_state, f"{output_prefix}_middle.pt")
    torch.save(end_state, f"{output_prefix}_end.pt")

    print(f"Saved networks to:")
    print(f"  {output_prefix}_opening.pt")
    print(f"  {output_prefix}_middle.pt")
    print(f"  {output_prefix}_end.pt")

    # Optionally verify by loading into individual Net instances
    verify_extraction(opening_state, middle_state, end_state)

    return opening_state, middle_state, end_state


def verify_extraction(opening_state, middle_state, end_state):
    """Verify that extracted weights can be loaded into individual Net instances."""
    print("\nVerifying extraction by loading into Net instances...")

    try:
        opening_net = Net()
        opening_net.load_state_dict(opening_state)
        print("✓ Opening network loaded successfully")

        middle_net = Net()
        middle_net.load_state_dict(middle_state)
        print("✓ Middle network loaded successfully")

        end_net = Net()
        end_net.load_state_dict(end_state)
        print("✓ End network loaded successfully")

        # Test forward pass
        test_input = torch.randn(1, INPUT_SIZE)
        with torch.no_grad():
            out1 = opening_net(test_input)
            out2 = middle_net(test_input)
            out3 = end_net(test_input)

        print("✓ All networks can perform forward pass")
        print(f"  Output shapes: {out1.shape}, {out2.shape}, {out3.shape}")

    except Exception as e:
        print(f"✗ Verification failed: {e}")


def print_network_weights(state_dict, network_name):
    """Print the structure and shapes of weights in a network."""
    print(f"\n{network_name} structure:")
    for key, value in state_dict.items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_separate_nets.py <checkpoint_path> [output_prefix]")
        print("\nExample:")
        print("  python extract_separate_nets.py nnue_multistage_1024x64x32x3.pt")
        print("  python extract_separate_nets.py nnue_multistage_1024x64x32x3.49.pt my_nets")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "net"

    opening_state, middle_state, end_state = extract_networks(checkpoint_path, output_prefix)

    # Optionally print detailed weight information
    if "--verbose" in sys.argv:
        print_network_weights(opening_state, "Opening Network")
        print_network_weights(middle_state, "Middle Network")
        print_network_weights(end_state, "End Network")
