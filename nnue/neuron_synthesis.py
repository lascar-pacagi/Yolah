#!/usr/bin/env python3
"""
Neuron Logic Synthesis Prototype

This script attempts to synthesize Boolean formulas that approximate
individual neurons in a quantized NNUE network.

Approach:
1. Load NNUE weights
2. Generate samples from real game positions
3. Compute binary neuron outputs (threshold at 0)
4. Use logic synthesis (ESPRESSO-style) to find compact formulas
"""

import numpy as np
import sys
import struct
from pathlib import Path
from collections import defaultdict
import itertools

# Try to import PyEDA for ESPRESSO synthesis
try:
    from pyeda.inter import exprvars, Or, And, espresso_exprs, expr2dimacscnf
    from pyeda.boolalg.expr import expr
    PYEDA_AVAILABLE = True
except ImportError:
    print("Warning: pyeda not found. Install with: pip install pyeda")
    PYEDA_AVAILABLE = False

sys.path.append("../server")
try:
    from yolah import Yolah, Move
except ImportError:
    print("Warning: yolah module not found, using mock")
    Yolah = None

# Configuration
INPUT_SIZE = 193  # 64 black + 64 white + 64 empty + 1 turn
H1_SIZE = 1024


def load_nnue_weights(filename):
    """Load quantized NNUE weights from text file.

    Format:
        W M N
        <M*N int8 values>
        B N
        <N int16 values>
        ...
    """
    weights = {}
    with open(filename, 'r') as f:
        tokens = f.read().split()
        idx = 0

        def read_matrix(m, n, transpose=False):
            nonlocal idx
            assert tokens[idx] == 'W', f"Expected W, got {tokens[idx]}"
            idx += 1
            assert int(tokens[idx]) == m and int(tokens[idx+1]) == n
            idx += 2
            mat = np.array([int(tokens[idx + i]) for i in range(m * n)], dtype=np.int8)
            idx += m * n
            if transpose:
                return mat.reshape(m, n).T  # (n, m) after transpose
            return mat.reshape(m, n)

        def read_bias(n):
            nonlocal idx
            assert tokens[idx] == 'B', f"Expected B, got {tokens[idx]}"
            idx += 1
            assert int(tokens[idx]) == n
            idx += 1
            bias = np.array([int(tokens[idx + i]) for i in range(n)], dtype=np.int16)
            idx += n
            return bias

        # input_to_h1: W[H1_SIZE, INPUT_SIZE] transposed -> (INPUT_SIZE, H1_SIZE)
        weights['input_to_h1'] = read_matrix(H1_SIZE, INPUT_SIZE, transpose=True)
        weights['h1_bias'] = read_bias(H1_SIZE)

        # Read remaining layers for completeness
        weights['h1_to_h2'] = read_matrix(64, H1_SIZE)
        weights['h2_bias'] = read_bias(64)
        weights['h2_to_h3'] = read_matrix(32, 64)
        weights['h3_bias'] = read_bias(32)
        weights['h3_to_output'] = read_matrix(3, 32)
        weights['output_bias'] = read_bias(3)

    return weights


def encode_position(black_bb, white_bb, empty_bb, turn):
    """Encode a position as a binary vector."""
    bits = []
    for bb in [black_bb, white_bb, empty_bb]:
        for i in range(64):
            bits.append((bb >> i) & 1)
    bits.append(turn)
    return np.array(bits, dtype=np.int8)


def compute_h1_activations(inputs, weights):
    """Compute H1 layer activations (pre-ReLU)."""
    # inputs: (batch, 193) binary
    # weights: (193, 1024) int8
    # bias: (1024,) int16
    return inputs @ weights['input_to_h1'] + weights['h1_bias']


def generate_random_positions(n_samples, seed=42):
    """Generate random game positions for sampling."""
    np.random.seed(seed)
    positions = []

    for _ in range(n_samples):
        # Generate a random valid position
        # Each square is black, white, or empty (but not multiple)
        squares = np.random.choice(3, 64)  # 0=empty, 1=black, 2=white

        black_bb = sum(1 << i for i in range(64) if squares[i] == 1)
        white_bb = sum(1 << i for i in range(64) if squares[i] == 2)
        empty_bb = sum(1 << i for i in range(64) if squares[i] == 0)
        turn = np.random.randint(0, 2)

        positions.append(encode_position(black_bb, white_bb, empty_bb, turn))

    return np.array(positions)


class NeuronSynthesizer:
    """Synthesize Boolean formula for a single neuron."""

    def __init__(self, neuron_idx, weights, samples, labels):
        """
        Args:
            neuron_idx: Index of neuron in H1 layer
            weights: NNUE weights dict
            samples: (n, 193) binary input samples
            labels: (n,) binary output labels (activation > 0)
        """
        self.neuron_idx = neuron_idx
        self.weights = weights
        self.samples = samples
        self.labels = labels

        # Get weights for this specific neuron
        self.neuron_weights = weights['input_to_h1'][:, neuron_idx]
        self.neuron_bias = weights['h1_bias'][neuron_idx]

        # Analyze weight distribution
        self.analyze_weights()

    def analyze_weights(self):
        """Analyze which inputs are most important."""
        w = self.neuron_weights
        self.nonzero_inputs = np.where(w != 0)[0]
        self.positive_inputs = np.where(w > 0)[0]
        self.negative_inputs = np.where(w < 0)[0]

        print(f"Neuron {self.neuron_idx}:")
        print(f"  Bias: {self.neuron_bias}")
        print(f"  Non-zero weights: {len(self.nonzero_inputs)}/{INPUT_SIZE}")
        print(f"  Positive: {len(self.positive_inputs)}, Negative: {len(self.negative_inputs)}")
        print(f"  Weight range: [{w.min()}, {w.max()}]")

        # Top contributing inputs
        top_k = 10
        sorted_idx = np.argsort(np.abs(w))[::-1][:top_k]
        print(f"  Top {top_k} weights: {list(zip(sorted_idx, w[sorted_idx]))}")

    def compute_output(self, x):
        """Compute binary output for input x."""
        activation = np.dot(x, self.neuron_weights) + self.neuron_bias
        return 1 if activation > 0 else 0

    def find_minterms(self):
        """Find input patterns that produce output=1 (minterms)."""
        minterms = []
        for i, (sample, label) in enumerate(zip(self.samples, self.labels)):
            if label == 1:
                # Only track nonzero inputs to reduce dimensionality
                minterm = tuple(sample[self.nonzero_inputs])
                minterms.append(minterm)
        return minterms

    def synthesize_dnf(self, max_terms=100):
        """
        Attempt to synthesize a DNF (sum of products) formula.

        This is a simplified version - for production use ESPRESSO or ABC.
        """
        minterms = self.find_minterms()

        if len(minterms) == 0:
            return "FALSE"
        if len(minterms) >= len(self.samples) // 2:
            # More than half are 1s, might be easier to synthesize complement
            pass

        # Group similar minterms
        minterm_set = set(minterms)
        print(f"  Unique minterms: {len(minterm_set)}")

        # Try to find common patterns (very simplified)
        # A real implementation would use Quine-McCluskey or ESPRESSO

        # For now, just report statistics
        return f"DNF with ~{len(minterm_set)} terms (use ESPRESSO for minimization)"

    def synthesize_espresso(self, max_inputs=20, max_samples=500):
        """
        Synthesize Boolean formula using ESPRESSO via PyEDA.

        Args:
            max_inputs: Maximum number of inputs to consider (dimensionality reduction)
            max_samples: Maximum samples to use for synthesis

        Returns:
            Minimized Boolean expression as string
        """
        if not PYEDA_AVAILABLE:
            return "PyEDA not available. Install with: pip install pyeda"

        # Select most important inputs (by weight magnitude)
        important_inputs = np.argsort(np.abs(self.neuron_weights))[::-1][:max_inputs]
        print(f"  Using top {len(important_inputs)} inputs for ESPRESSO")

        # Create PyEDA variables
        X = exprvars('x', len(important_inputs))

        # Build truth table from samples
        on_set = []   # Minterms where output = 1
        off_set = []  # Minterms where output = 0

        # Limit samples for tractability
        sample_indices = np.random.choice(
            len(self.samples),
            min(max_samples, len(self.samples)),
            replace=False
        )

        for idx in sample_indices:
            sample = self.samples[idx]
            label = self.labels[idx]

            # Extract relevant input bits
            input_bits = tuple(sample[important_inputs])

            if label == 1:
                on_set.append(input_bits)
            else:
                off_set.append(input_bits)

        # Remove duplicates
        on_set = list(set(on_set))
        off_set = list(set(off_set))

        print(f"  ON-set: {len(on_set)} minterms")
        print(f"  OFF-set: {len(off_set)} minterms")

        if len(on_set) == 0:
            return "FALSE (no positive samples)"

        # Build product terms for ON-set
        products = []
        for minterm in on_set:
            literals = []
            for i, bit in enumerate(minterm):
                if bit == 1:
                    literals.append(X[i])
                else:
                    literals.append(~X[i])
            if literals:
                products.append(And(*literals))

        if not products:
            return "FALSE"

        # Create DNF expression
        f = Or(*products) if len(products) > 1 else products[0]

        # Apply ESPRESSO minimization
        try:
            print("  Running ESPRESSO minimization...")
            f_min = espresso_exprs(f)
            minimized = f_min[0] if f_min else f

            # Convert to readable string with original input indices
            result_str = str(minimized)

            # Map back to original input indices
            for i, orig_idx in enumerate(important_inputs):
                result_str = result_str.replace(f'x[{i}]', f'in[{orig_idx}]')

            return result_str

        except Exception as e:
            return f"ESPRESSO failed: {e}"

    def synthesize_espresso_bitwise(self, max_inputs=30, max_samples=1000):
        """
        Synthesize and convert to bitwise operations.

        This version groups inputs by bitboard (black/white/empty) and
        generates optimized bitwise code.
        """
        if not PYEDA_AVAILABLE:
            return "PyEDA not available"

        # Analyze per-bitboard contributions
        black_w = self.neuron_weights[0:64]
        white_w = self.neuron_weights[64:128]
        empty_w = self.neuron_weights[128:192]

        # Find significant inputs per bitboard
        def top_inputs(w, k=10):
            idx = np.argsort(np.abs(w))[::-1][:k]
            return idx[w[idx] != 0]

        black_important = top_inputs(black_w)
        white_important = top_inputs(white_w)
        empty_important = top_inputs(empty_w)

        print(f"  Important inputs - Black: {len(black_important)}, "
              f"White: {len(white_important)}, Empty: {len(empty_important)}")

        # For each bitboard, try to find a simple pattern
        result = {
            'black_mask_pos': 0,
            'black_mask_neg': 0,
            'white_mask_pos': 0,
            'white_mask_neg': 0,
            'empty_mask_pos': 0,
            'empty_mask_neg': 0,
        }

        for i in black_important:
            if black_w[i] > 0:
                result['black_mask_pos'] |= (1 << i)
            elif black_w[i] < 0:
                result['black_mask_neg'] |= (1 << i)

        for i in white_important:
            if white_w[i] > 0:
                result['white_mask_pos'] |= (1 << i)
            elif white_w[i] < 0:
                result['white_mask_neg'] |= (1 << i)

        for i in empty_important:
            if empty_w[i] > 0:
                result['empty_mask_pos'] |= (1 << i)
            elif empty_w[i] < 0:
                result['empty_mask_neg'] |= (1 << i)

        # Generate C code
        code = f"""
// Neuron {self.neuron_idx} - ESPRESSO optimized
// Approximation using top weighted inputs
bool neuron_{self.neuron_idx}_approx(uint64_t black, uint64_t white, uint64_t empty, int turn) {{
    int score = {self.neuron_bias};

    // Positive contributions
    score += popcount(black & 0x{result['black_mask_pos']:016x}ULL) * {int(np.mean(np.abs(black_w[black_w > 0]))) if np.any(black_w > 0) else 0};
    score += popcount(white & 0x{result['white_mask_pos']:016x}ULL) * {int(np.mean(np.abs(white_w[white_w > 0]))) if np.any(white_w > 0) else 0};
    score += popcount(empty & 0x{result['empty_mask_pos']:016x}ULL) * {int(np.mean(np.abs(empty_w[empty_w > 0]))) if np.any(empty_w > 0) else 0};

    // Negative contributions
    score -= popcount(black & 0x{result['black_mask_neg']:016x}ULL) * {int(np.mean(np.abs(black_w[black_w < 0]))) if np.any(black_w < 0) else 0};
    score -= popcount(white & 0x{result['white_mask_neg']:016x}ULL) * {int(np.mean(np.abs(white_w[white_w < 0]))) if np.any(white_w < 0) else 0};
    score -= popcount(empty & 0x{result['empty_mask_neg']:016x}ULL) * {int(np.mean(np.abs(empty_w[empty_w < 0]))) if np.any(empty_w < 0) else 0};

    // Turn
    score += turn * {self.neuron_weights[192]};

    return score > 0;
}}
"""
        return code

    def synthesize_threshold(self):
        """
        Synthesize as a threshold function: sum(w_i * x_i) > threshold

        This is the natural form for a neuron and might be more compact
        than arbitrary Boolean formulas.
        """
        # The neuron IS a threshold function by definition
        # We can potentially simplify by:
        # 1. Removing inputs with zero weight
        # 2. Grouping inputs with same weight
        # 3. Finding equivalent simpler threshold functions

        # Group by weight value
        weight_groups = defaultdict(list)
        for i in self.nonzero_inputs:
            weight_groups[self.neuron_weights[i]].append(i)

        formula_parts = []
        for w, inputs in sorted(weight_groups.items(), key=lambda x: -abs(x[0])):
            if len(inputs) == 1:
                formula_parts.append(f"{w}*x[{inputs[0]}]")
            else:
                formula_parts.append(f"{w}*popcount({inputs})")

        return f"({' + '.join(formula_parts)}) + {self.neuron_bias} > 0"

    def to_bitwise_ops(self):
        """
        Convert to bitwise operations exploiting the board structure.

        Since inputs are bitboards, we can use:
        - popcount for counting
        - AND/OR for masking
        """
        # Inputs 0-63: black bitboard
        # Inputs 64-127: white bitboard
        # Inputs 128-191: empty bitboard
        # Input 192: turn

        black_weights = self.neuron_weights[0:64]
        white_weights = self.neuron_weights[64:128]
        empty_weights = self.neuron_weights[128:192]
        turn_weight = self.neuron_weights[192]

        def weights_to_mask(w):
            """Convert weight vector to (positive_mask, negative_mask, weights)"""
            pos_mask = sum(1 << i for i in range(64) if w[i] > 0)
            neg_mask = sum(1 << i for i in range(64) if w[i] < 0)
            return pos_mask, neg_mask

        black_pos, black_neg = weights_to_mask(black_weights)
        white_pos, white_neg = weights_to_mask(white_weights)
        empty_pos, empty_neg = weights_to_mask(empty_weights)

        code = f"""
// Neuron {self.neuron_idx}
int16_t neuron_{self.neuron_idx}(uint64_t black, uint64_t white, uint64_t empty, int turn) {{
    int32_t sum = {self.neuron_bias};

    // Black contributions
    sum += {int(black_weights[black_weights > 0].sum())} * popcount(black & 0x{black_pos:016x}ULL);
    sum -= {int(-black_weights[black_weights < 0].sum())} * popcount(black & 0x{black_neg:016x}ULL);

    // White contributions
    sum += {int(white_weights[white_weights > 0].sum())} * popcount(white & 0x{white_pos:016x}ULL);
    sum -= {int(-white_weights[white_weights < 0].sum())} * popcount(white & 0x{white_neg:016x}ULL);

    // Empty contributions
    sum += {int(empty_weights[empty_weights > 0].sum())} * popcount(empty & 0x{empty_pos:016x}ULL);
    sum -= {int(-empty_weights[empty_weights < 0].sum())} * popcount(empty & 0x{empty_neg:016x}ULL);

    // Turn contribution
    sum += turn * {turn_weight};

    return sum > 0 ? 1 : 0;
}}
"""
        return code


def analyze_weight_patterns(weights):
    """Analyze patterns in weight matrix that could enable compression."""
    W = weights['input_to_h1']

    print("\n=== Weight Matrix Analysis ===")
    print(f"Shape: {W.shape}")
    print(f"Sparsity: {np.sum(W == 0) / W.size * 100:.1f}%")
    print(f"Unique values: {len(np.unique(W))}")

    # Check for repeated columns (neurons with same weights)
    unique_cols = len(set(tuple(W[:, i]) for i in range(W.shape[1])))
    print(f"Unique neurons: {unique_cols}/{W.shape[1]}")

    # Check for repeated rows (inputs with same effect)
    unique_rows = len(set(tuple(W[i, :]) for i in range(W.shape[0])))
    print(f"Unique input patterns: {unique_rows}/{W.shape[0]}")

    # Analyze weight distribution per neuron
    weights_per_neuron = np.sum(W != 0, axis=0)
    print(f"Non-zero weights per neuron: min={weights_per_neuron.min()}, "
          f"max={weights_per_neuron.max()}, mean={weights_per_neuron.mean():.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Synthesize logic for NNUE neurons')
    parser.add_argument('--weights', type=str, default='nnue_quantized.bin',
                        help='Path to quantized weights file')
    parser.add_argument('--neuron', type=int, default=0,
                        help='Neuron index to analyze')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of random samples')
    args = parser.parse_args()

    # Check if weights file exists
    if not Path(args.weights).exists():
        print(f"Weights file {args.weights} not found.")
        print("Creating synthetic weights for demonstration...")
        # Create mock weights for demo
        weights = {
            'h1_bias': np.random.randint(-100, 100, H1_SIZE, dtype=np.int16),
            'input_to_h1': np.random.randint(-10, 10, (INPUT_SIZE, H1_SIZE), dtype=np.int8)
        }
        # Make it sparse
        mask = np.random.random((INPUT_SIZE, H1_SIZE)) > 0.7
        weights['input_to_h1'][mask] = 0
    else:
        weights = load_nnue_weights(args.weights)

    # Analyze overall weight patterns
    analyze_weight_patterns(weights)

    # Generate samples
    print(f"\nGenerating {args.samples} random positions...")
    samples = generate_random_positions(args.samples)

    # Compute activations
    activations = compute_h1_activations(samples, weights)
    labels = (activations[:, args.neuron] > 0).astype(np.int8)

    print(f"\nNeuron {args.neuron} output distribution:")
    print(f"  0: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    print(f"  1: {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")

    # Synthesize
    synth = NeuronSynthesizer(args.neuron, weights, samples, labels)

    print("\n=== Threshold Function Form ===")
    print(synth.synthesize_threshold())

    print("\n=== Bitwise C Code ===")
    print(synth.to_bitwise_ops())

    print("\n=== DNF Synthesis (simple) ===")
    print(synth.synthesize_dnf())

    if PYEDA_AVAILABLE:
        print("\n=== ESPRESSO Synthesis ===")
        print(synth.synthesize_espresso(max_inputs=15, max_samples=500))

        print("\n=== ESPRESSO Bitwise Code ===")
        print(synth.synthesize_espresso_bitwise())
    else:
        print("\n[!] Install PyEDA for ESPRESSO synthesis: pip install pyeda")


if __name__ == '__main__':
    main()
