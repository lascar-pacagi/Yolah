#!/usr/bin/env python3
"""
Visualize Yolah bitboards in UTF-8/ASCII format or LaTeX format.

This tool displays bitboards with various filtering options and multiple representations:
- Board view: Visual board with pieces
- Bitboard view: 64-bit representation with indices and square names
- Filter options: Show only black pieces, white pieces, empty squares, or all
- LaTeX output: Generate LaTeX code for direct inclusion in documents

Usage:
    # Show initial position (all pieces)
    python visualize_bitboard.py

    # Show only black pieces
    python visualize_bitboard.py --show black

    # Show only white pieces
    python visualize_bitboard.py --show white

    # Show only empty squares
    python visualize_bitboard.py --show empty

    # Show board after moves
    python visualize_bitboard.py --moves "a1:b1 b1:b5 c8:c7"

    # Show bitboard array representation
    python visualize_bitboard.py --array

    # Generate LaTeX output
    python visualize_bitboard.py --latex

    # Combine options
    python visualize_bitboard.py --moves "a1:b1" --show black --array --latex
"""

import sys
import argparse

# Add the server directory to the path to import yolah
sys.path.insert(0, '../server')
from yolah import Yolah, Move, Cell

# Board configuration
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']

# UTF-8 characters for board display
CHAR_BLACK = '●'  # Black circle
CHAR_WHITE = '○'  # White circle
CHAR_EMPTY = '×'  # Empty cell (not playable)
CHAR_VACANT = '·'  # Vacant square (playable but no piece)

# Box drawing characters
BOX_TL = '┌'  # Top-left
BOX_TR = '┐'  # Top-right
BOX_BL = '└'  # Bottom-left
BOX_BR = '┘'  # Bottom-right
BOX_H = '─'   # Horizontal
BOX_V = '│'   # Vertical
BOX_VR = '├'  # Vertical-right
BOX_VL = '┤'  # Vertical-left
BOX_HU = '┴'  # Horizontal-up
BOX_HD = '┬'  # Horizontal-down
BOX_CROSS = '┼'  # Cross


def bitboard_to_string(bitboard):
    """Convert a 64-bit integer to binary string (64 bits)."""
    return format(bitboard, '064b')


def get_bit(bitboard, index):
    """Get bit at index (0-63) from bitboard."""
    return (bitboard >> index) & 1


def square_to_index(rank, file):
    """Convert rank (0-7) and file (0-7) to bitboard index (0-63)."""
    return rank * 8 + file


def index_to_square(index):
    """Convert bitboard index (0-63) to square name (a1-h8)."""
    rank = index // 8
    file = index % 8
    return f"{FILES[file]}{RANKS[rank]}"


def print_board_utf8(yolah, show='all'):
    """
    Print the board using UTF-8 characters.

    Args:
        yolah: Yolah game instance
        show: What to display - 'all', 'black', 'white', 'empty', or 'vacant'
    """
    grid = yolah.grid()

    # Print top border with file labels
    print("\n  " + BOX_TL + (BOX_H * 3 + BOX_HD) * 7 + BOX_H * 3 + BOX_TR)

    # Print board from rank 8 to rank 1 (top to bottom)
    for i in range(Yolah.DIM - 1, -1, -1):
        # Print rank label and left border
        print(f"{RANKS[i]} {BOX_V}", end='')

        for j in range(Yolah.DIM):
            cell = grid[i][j]

            # Determine what character to display based on filter
            char = ' '
            if show == 'all':
                if cell == Cell.BLACK:
                    char = CHAR_BLACK
                elif cell == Cell.WHITE:
                    char = CHAR_WHITE
                elif cell == Cell.EMPTY:
                    char = CHAR_EMPTY
                else:  # Vacant
                    char = CHAR_VACANT
            elif show == 'black':
                if cell == Cell.BLACK:
                    char = CHAR_BLACK
                else:
                    char = CHAR_VACANT
            elif show == 'white':
                if cell == Cell.WHITE:
                    char = CHAR_WHITE
                else:
                    char = CHAR_VACANT
            elif show == 'empty':
                if cell == Cell.EMPTY:
                    char = CHAR_EMPTY
                else:
                    char = CHAR_VACANT
            elif show == 'vacant':
                if cell == Cell.NONE:
                    char = CHAR_VACANT
                else:
                    char = ' '

            print(f" {char} ", end='')
            if j < Yolah.DIM - 1:
                print(BOX_V, end='')

        print(BOX_V)

        # Print horizontal separator (except after last rank)
        if i > 0:
            print("  " + BOX_VR + (BOX_H * 3 + BOX_CROSS) * 7 + BOX_H * 3 + BOX_VL)

    # Print bottom border with file labels
    print("  " + BOX_BL + (BOX_H * 3 + BOX_HU) * 7 + BOX_H * 3 + BOX_BR)
    print("    " + "   ".join(FILES))

    # Print legend
    print(f"\nLegend: {CHAR_BLACK} = Black  {CHAR_WHITE} = White  {CHAR_EMPTY} = Empty  {CHAR_VACANT} = Vacant")


def print_single_bitboard_latex(bitboard, name=""):
    """
    Generate LaTeX code showing a single bitboard as a grid with 1s and 0s.

    Args:
        bitboard: 64-bit integer representing the bitboard
        name: Name of the bitboard (e.g., "Noir", "Blanc")
    """
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    print("\\hline")

    # Print board from rank 8 to rank 1 (top to bottom)
    for i in range(7, -1, -1):
        # Print rank label
        print(f"{RANKS[i]}", end='')

        for j in range(8):
            # Calculate bit index for this square
            bit_index = square_to_index(i, j)
            # Get bit value (1 or 0)
            bit_value = (bitboard >> bit_index) & 1
            print(f" & {bit_value}", end='')

        print(" \\\\ \\hline")

    # Print file labels with vertical line on left cell
    print("\\multicolumn{1}{c|}{} & " + " & ".join(FILES) + " \\\\")
    print("\\cline{2-9}")
    print("\\end{tabular}")
    print("\\end{table}")

    # Print binary representation
    print()
    print("\\noindent")
    print(f"Représentation binaire: \\texttt{{{bitboard:064b}}}")


def print_bitboard_indices_latex(caption='', label=''):
    """
    Generate LaTeX code showing bitboard bit indices for each square.

    Args:
        caption: Optional caption for the table
        label: Optional label for referencing
    """
    print("\\begin{table}[H]")
    print("\\centering")
    if caption:
        print(f"\\caption{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    print("\\hline")

    # Print board from rank 8 to rank 1 (top to bottom)
    for i in range(7, -1, -1):
        # Print rank label
        print(f"{RANKS[i]}", end='')

        for j in range(8):
            # Calculate bit index for this square
            bit_index = square_to_index(i, j)
            print(f" & bit$_{{{bit_index}}}$", end='')

        print(" \\\\ \\hline")

    # Print file labels with vertical line on left cell
    print("\\multicolumn{1}{c|}{} & " + " & ".join(FILES) + " \\\\")
    print("\\cline{2-9}")
    print("\\end{tabular}")
    print("\\end{table}")


def print_board_latex(yolah, show='all', caption='', label=''):
    """
    Generate LaTeX code for the board.

    Args:
        yolah: Yolah game instance
        show: What to display - 'all', 'black', 'white', 'empty', 'vacant', or comma-separated list like 'black,white'
        caption: Optional caption for the table
        label: Optional label for referencing
    """
    grid = yolah.grid()

    print("\\begin{table}[H]")
    print("\\centering")
    if caption:
        print(f"\\caption{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    print("\\hline")

    # Parse show parameter - can be comma-separated list
    show_list = [s.strip() for s in show.split(',')]

    # Print board from rank 8 to rank 1 (top to bottom)
    for i in range(Yolah.DIM - 1, -1, -1):
        # Print rank label
        print(f"{RANKS[i]}", end='')

        for j in range(Yolah.DIM):
            cell = grid[i][j]

            # Determine what character to display based on filter
            char = '$\\cdot$'  # Default to vacant

            if show == 'all':
                if cell == Cell.BLACK:
                    char = '$\\bullet$'
                elif cell == Cell.WHITE:
                    char = '$\\circ$'
                elif cell == Cell.EMPTY:
                    char = '$\\times$'
                else:  # Vacant
                    char = '$\\cdot$'
            elif len(show_list) > 1:
                # Multiple filters specified
                if 'black' in show_list and cell == Cell.BLACK:
                    char = '$\\bullet$'
                elif 'white' in show_list and cell == Cell.WHITE:
                    char = '$\\circ$'
                elif 'empty' in show_list and cell == Cell.EMPTY:
                    char = '$\\times$'
                elif 'vacant' in show_list and cell == Cell.NONE:
                    char = '$\\cdot$'
            else:
                # Single filter
                if show == 'black':
                    if cell == Cell.BLACK:
                        char = '$\\bullet$'
                elif show == 'white':
                    if cell == Cell.WHITE:
                        char = '$\\circ$'
                elif show == 'empty':
                    if cell == Cell.EMPTY:
                        char = '$\\times$'
                elif show == 'vacant':
                    if cell == Cell.NONE:
                        char = '$\\cdot$'
                    else:
                        char = ' '

            print(f" & {char}", end='')

        # Use \hline for all ranks
        print(" \\\\ \\hline")

    # Print file labels with vertical line on left cell
    print("\\multicolumn{1}{c|}{} & " + " & ".join(FILES) + " \\\\")
    print("\\cline{2-9}")
    print("\\end{tabular}")
    print("\\end{table}")

    # Print bitboard representations
    print()
    print("\\noindent")
    print("Représentations binaires des bitboards:")
    print("\\begin{itemize}")
    print(f"\\item Black: \\texttt{{{yolah.black:064b}}}")
    print(f"\\item White: \\texttt{{{yolah.white:064b}}}")
    print(f"\\item Empty: \\texttt{{{yolah.empty:064b}}}")
    print("\\end{itemize}")

    # Print individual bitboards for black and white
    print()
    print("\\noindent")
    print("Bitboard des pièces noires:")
    print()
    print_single_bitboard_latex(yolah.black, "Noir")

    print()
    print("\\noindent")
    print("Bitboard des pièces blanches:")
    print()
    print_single_bitboard_latex(yolah.white, "Blanc")

    # Print combined bitboard (black | white)
    print()
    print("\\noindent")
    print("Bitboard des pièces (noir | blanc):")
    print()
    print_single_bitboard_latex(yolah.black | yolah.white, "Noir | Blanc")


def print_bitboard_latex(yolah, show='all', caption='', label=''):
    """
    Generate LaTeX code for horizontal bitboard array with index, square, and bit value.

    Args:
        yolah: Yolah game instance
        show: What to display - 'all', 'black', 'white', or 'empty'
        caption: Optional caption
        label: Optional label
    """
    black_bb = yolah.black
    white_bb = yolah.white
    empty_bb = yolah.empty

    print("\\begin{table}[H]")
    print("\\centering")
    if caption:
        print(f"\\caption{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")
    print("\\tiny")

    # Create a single horizontal table with index on top, square in middle, bit on bottom
    print("\\begin{tabular}{|c|" + "|c" * 64 + "|}")
    print("\\hline")

    # Row 1: Bit indices (0-63)
    print("\\textbf{Index}", end='')
    for rank in range(7, -1, -1):
        for file in range(8):
            index = square_to_index(rank, file)
            print(f" & {index}", end='')
    print(" \\\\ \\hline")

    # Row 2: Square names (a1-h8)
    print("\\textbf{Square}", end='')
    for rank in range(7, -1, -1):
        for file in range(8):
            square = index_to_square(square_to_index(rank, file))
            print(f" & \\texttt{{{square}}}", end='')
    print(" \\\\ \\hline")

    # Row 3: Bit values (0 or 1)
    print("\\textbf{Bit}", end='')
    for rank in range(7, -1, -1):
        for file in range(8):
            index = square_to_index(rank, file)

            # Get bit values
            black_bit = get_bit(black_bb, index)
            white_bit = get_bit(white_bb, index)
            empty_bit = get_bit(empty_bb, index)

            # Determine what to show
            bit_val = '0'

            if show == 'all':
                if black_bit or white_bit or empty_bit:
                    bit_val = '1'
            elif show == 'black':
                bit_val = str(black_bit)
            elif show == 'white':
                bit_val = str(white_bit)
            elif show == 'empty':
                bit_val = str(empty_bit)

            print(f" & {bit_val}", end='')
    print(" \\\\ \\hline \\hline")

    print("\\end{tabular}")
    print("\\end{table}")

    # Print bitboard hex values
    print("\n\\noindent")
    print("Bitboard values (hexadecimal):")
    print("\\begin{itemize}")
    print(f"\\item Black: \\texttt{{0x{black_bb:016X}}}")
    print(f"\\item White: \\texttt{{0x{white_bb:016X}}}")
    print(f"\\item Empty: \\texttt{{0x{empty_bb:016X}}}")
    print("\\end{itemize}")


def print_bitboard_array(yolah, show='all'):
    """
    Print bitboard as an array of 64 squares with indices and square names.

    Args:
        yolah: Yolah game instance
        show: What to display - 'all', 'black', 'white', or 'empty'
    """
    # Get internal bitboards
    black_bb = yolah.black
    white_bb = yolah.white
    empty_bb = yolah.empty

    print("\n" + "=" * 70)
    print(f"BITBOARD ARRAY VIEW (showing: {show})")
    print("=" * 70)

    # Print bitboard values in hex
    print(f"\nBitboard values (hex):")
    print(f"  Black:  0x{black_bb:016X}")
    print(f"  White:  0x{white_bb:016X}")
    print(f"  Empty:  0x{empty_bb:016X}")

    # Print array representation
    print(f"\nArray representation (index | square | bit):")
    print("-" * 70)

    # Print from rank 7 to rank 0 (64-bit index order)
    for rank in range(7, -1, -1):
        print(f"\nRank {rank + 1}:")
        for file in range(8):
            index = square_to_index(rank, file)
            square = index_to_square(index)

            # Get bit values
            black_bit = get_bit(black_bb, index)
            white_bit = get_bit(white_bb, index)
            empty_bit = get_bit(empty_bb, index)

            # Determine what to show based on filter
            bit_char = '0'
            piece_char = CHAR_VACANT

            if show == 'all':
                if black_bit:
                    bit_char = '1'
                    piece_char = CHAR_BLACK
                elif white_bit:
                    bit_char = '1'
                    piece_char = CHAR_WHITE
                elif empty_bit:
                    bit_char = '1'
                    piece_char = CHAR_EMPTY
            elif show == 'black':
                bit_char = str(black_bit)
                piece_char = CHAR_BLACK if black_bit else CHAR_VACANT
            elif show == 'white':
                bit_char = str(white_bit)
                piece_char = CHAR_WHITE if white_bit else CHAR_VACANT
            elif show == 'empty':
                bit_char = str(empty_bit)
                piece_char = CHAR_EMPTY if empty_bit else CHAR_VACANT

            print(f"  [{index:2d}] {square} = {bit_char}  {piece_char}")

    print("-" * 70)


def print_compact_bitboard(yolah, show='all'):
    """
    Print a compact bitboard view showing all 64 bits in 8x8 grid.

    Args:
        yolah: Yolah game instance
        show: What to display - 'all', 'black', 'white', or 'empty'
    """
    black_bb = yolah.black
    white_bb = yolah.white
    empty_bb = yolah.empty

    print("\n" + "=" * 50)
    print(f"COMPACT BITBOARD (showing: {show})")
    print("=" * 50)

    # Print header with file labels
    print("\n     " + "  ".join(FILES))
    print("   +" + "---+" * 8)

    # Print from rank 7 to rank 0 (top to bottom for display)
    for rank in range(7, -1, -1):
        print(f" {rank + 1} |", end='')

        for file in range(8):
            index = square_to_index(rank, file)

            # Get bit values
            black_bit = get_bit(black_bb, index)
            white_bit = get_bit(white_bb, index)
            empty_bit = get_bit(empty_bb, index)

            # Determine display
            if show == 'all':
                if black_bit:
                    char = CHAR_BLACK
                elif white_bit:
                    char = CHAR_WHITE
                elif empty_bit:
                    char = CHAR_EMPTY
                else:
                    char = '0'
            elif show == 'black':
                char = CHAR_BLACK if black_bit else '0'
            elif show == 'white':
                char = CHAR_WHITE if white_bit else '0'
            elif show == 'empty':
                char = CHAR_EMPTY if empty_bit else '0'
            else:
                char = '0'

            print(f" {char} |", end='')

        print(f" {rank + 1}")
        print("   +" + "---+" * 8)

    # Print footer with file labels
    print("     " + "  ".join(FILES))

    # Print bit indices
    print("\nBit indices (0-63):")
    print("   +" + "---+" * 8)
    for rank in range(7, -1, -1):
        print(f" {rank + 1} |", end='')
        for file in range(8):
            index = square_to_index(rank, file)
            print(f"{index:2d} |", end='')
        print(f" {rank + 1}")
        print("   +" + "---+" * 8)
    print("     " + "  ".join(FILES))


def parse_moves(moves_str):
    """
    Parse a string of moves in format "a1:b1 b1:b5 ..."

    Returns:
        List of Move objects
    """
    if not moves_str or moves_str.strip() == "":
        return []

    move_strings = moves_str.strip().split()
    moves = []
    for m_str in move_strings:
        try:
            moves.append(Move.from_str(m_str))
        except Exception as e:
            print(f"Warning: Could not parse move '{m_str}': {e}")
    return moves


def visualize_bitboard(moves_str=None, show='all', show_array=False, compact=False, latex=False, caption='', label='', show_indices=False):
    """
    Visualize bitboard in various formats.

    Args:
        moves_str: String of moves (e.g., "a1:b1 b1:b5"), or None for initial position
        show: What to display - 'all', 'black', 'white', 'empty', or 'vacant'
        show_array: Show detailed array representation with indices
        compact: Show compact bitboard view
        latex: Generate LaTeX output
        caption: Caption for LaTeX tables
        label: Label for LaTeX tables
        show_indices: Show bitboard bit indices table (LaTeX only)
    """
    # Create Yolah game
    yolah = Yolah()
    yolah.reset()

    # Parse and play moves
    moves = parse_moves(moves_str) if moves_str else []

    for move in moves:
        try:
            yolah.play(move)
        except Exception as e:
            print(f"Warning: Could not play move {move}: {e}")
            break

    if latex:
        # LaTeX output mode
        print("% Yolah Bitboard Visualization - LaTeX Output")
        print(f"% Moves: {' '.join(str(m) for m in moves) if moves else 'Initial position'}")
        print(f"% Showing: {show}")
        print()

        # Generate bitboard indices table if requested
        if show_indices:
            indices_caption = caption if caption else "Bitboard bit indices"
            indices_label = label if label else "tab:bitboard_indices"
            print_bitboard_indices_latex(indices_caption, indices_label)
            print()

        # Generate board table
        board_caption = caption if caption else f"Yolah board (showing: {show})"
        board_label = label if label else f"tab:yolah_{show}"
        print_board_latex(yolah, show, board_caption, board_label)
    else:
        # Standard terminal output
        # Print header
        print("\n" + "=" * 70)
        print("YOLAH BITBOARD VISUALIZATION")
        print("=" * 70)

        # Print game state info
        black_score, white_score = yolah.black_score, yolah.white_score
        print(f"\nMoves played: {len(moves)}")
        if moves:
            print(f"Moves: {' '.join(str(m) for m in moves)}")
        print(f"Current turn: {'WHITE' if yolah.current_player() == Yolah.WHITE_PLAYER else 'BLACK'}")
        print(f"Black score: {black_score}, White score: {white_score}")
        if yolah.game_over():
            print("Game is over!")

        # Print board view
        print_board_utf8(yolah, show)

        # Print array view if requested
        if show_array:
            print_bitboard_array(yolah, show)

        # Print compact view if requested
        if compact:
            print_compact_bitboard(yolah, show)

        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Yolah bitboards in UTF-8/ASCII format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show initial position (all pieces)
  python visualize_bitboard.py

  # Show only black pieces
  python visualize_bitboard.py --show black

  # Show only white pieces
  python visualize_bitboard.py --show white

  # Show only empty squares
  python visualize_bitboard.py --show empty

  # Show board after moves
  python visualize_bitboard.py --moves "a1:b1 b1:b5 c8:c7"

  # Show with array representation
  python visualize_bitboard.py --array

  # Show compact bitboard view
  python visualize_bitboard.py --compact

  # Combine options
  python visualize_bitboard.py --moves "a1:b1" --show black --array --compact
        """
    )

    parser.add_argument('--moves', type=str, default=None,
                       help='Space-separated moves in format "a1:b1 b1:b5 ..."')
    parser.add_argument('--show', type=str, default='all',
                       help='What to display: all, black, white, empty, vacant, or comma-separated (e.g., black,white) (default: all)')
    parser.add_argument('--array', action='store_true',
                       help='Show detailed array representation with indices and square names')
    parser.add_argument('--compact', action='store_true',
                       help='Show compact bitboard view')
    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX output for direct inclusion in documents')
    parser.add_argument('--caption', type=str, default='',
                       help='Caption for LaTeX table (only with --latex)')
    parser.add_argument('--label', type=str, default='',
                       help='Label for LaTeX table (only with --latex)')
    parser.add_argument('--indices', action='store_true',
                       help='Show bitboard bit indices table (only with --latex)')

    args = parser.parse_args()

    visualize_bitboard(
        moves_str=args.moves,
        show=args.show,
        show_array=args.array,
        compact=args.compact,
        latex=args.latex,
        caption=args.caption,
        label=args.label,
        show_indices=args.indices
    )


if __name__ == '__main__':
    main()
