# Multi-Stage Neural Network Implementation

## Overview

The `MultiStageNet` contains three separate neural networks:
- **Opening Network**: Used when free squares > 36
- **Middle Network**: Used when 18 < free squares ≤ 36
- **End Network**: Used when free squares ≤ 18

All three networks share the same architecture: 193 → 1024 → 64 → 32 → 3

## Saved Weights Structure

When you save a `MultiStageNet`, the state dictionary contains keys like:

```
opening_net.fc1.weight    [1024, 193]
opening_net.fc1.bias      [1024]
opening_net.fc2.weight    [64, 1024]
opening_net.fc2.bias      [64]
opening_net.fc3.weight    [32, 64]
opening_net.fc3.bias      [32]
opening_net.fc4.weight    [3, 32]
opening_net.fc4.bias      [3]

middle_net.fc1.weight     [1024, 193]
middle_net.fc1.bias       [1024]
... (same structure)

end_net.fc1.weight        [1024, 193]
end_net.fc1.bias          [1024]
... (same structure)
```

## Extracting Networks for C++

### Method 1: Using extract_separate_nets.py

```bash
python extract_separate_nets.py nnue_multistage_1024x64x32x3.pt
```

This creates three separate files:
- `net_opening.pt` - Opening network only
- `net_middle.pt` - Middle network only
- `net_end.pt` - End network only

Each file contains a standard `Net` state dict that can be loaded with your existing C++ code.

### Method 2: Direct Access in Python

```python
import torch

# Load the full multi-stage model
state_dict = torch.load('nnue_multistage_1024x64x32x3.pt')

# Extract opening network weights
opening_weights = {
    'fc1.weight': state_dict['opening_net.fc1.weight'],
    'fc1.bias': state_dict['opening_net.fc1.bias'],
    'fc2.weight': state_dict['opening_net.fc2.weight'],
    'fc2.bias': state_dict['opening_net.fc2.bias'],
    'fc3.weight': state_dict['opening_net.fc3.weight'],
    'fc3.bias': state_dict['opening_net.fc3.bias'],
    'fc4.weight': state_dict['opening_net.fc4.weight'],
    'fc4.bias': state_dict['opening_net.fc4.bias'],
}

# Save as a standalone network
torch.save(opening_weights, 'opening_net.pt')
```

## C++ Implementation

In your C++ code, you'll need to:

1. **Load all three networks** at initialization
2. **Determine game phase** using the same logic as `GameDataset.calculate_phase()`:
   ```cpp
   int calculate_phase(const Yolah& yolah) {
       int free_squares = popcount(yolah.free_squares());
       if (free_squares <= 18) return 2;  // End game
       if (free_squares <= 36) return 1;  // Middle game
       return 0;  // Opening
   }
   ```
3. **Select the appropriate network** based on phase:
   ```cpp
   int evaluate(const Yolah& yolah) {
       int phase = calculate_phase(yolah);
       switch(phase) {
           case 0: return opening_net.forward(encode(yolah));
           case 1: return middle_net.forward(encode(yolah));
           case 2: return end_net.forward(encode(yolah));
       }
   }
   ```

## Training

The multi-stage model trains all three networks simultaneously:
- Positions are automatically routed to the correct network based on their phase
- Each network only receives gradients from positions in its phase
- All three networks share the same loss function and optimizer

## Utilities

- `extract_separate_nets.py` - Extract individual networks from checkpoint
- `inspect_multistage_weights.py` - Inspect weight structure and ranges
- `nnue_multi_stage_multigpu.py` - Training script

## Example Usage

```bash
# Train the multi-stage model
python nnue_multi_stage_multigpu.py

# After training, extract individual networks
python extract_separate_nets.py nnue_multistage_1024x64x32x3.49.pt

# Inspect the weights
python inspect_multistage_weights.py nnue_multistage_1024x64x32x3.49.pt

# Use in your C++ engine
# Load net_opening.pt, net_middle.pt, net_end.pt
# Select network based on game phase
```
