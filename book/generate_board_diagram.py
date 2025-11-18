#!/usr/bin/env python3
"""
Generate Yolah board diagrams from initial position or move sequences.

Usage:
    # Generate initial board
    python generate_board_diagram.py --output initial_board.png

    # Generate board from moves
    python generate_board_diagram.py --moves "a1:b1 b1:b5 c8:c7" --output board.png

    # Custom size
    python generate_board_diagram.py --moves "a1:b1" --output board.png --size 800
"""

import sys
import argparse
from PIL import Image, ImageDraw, ImageFont

# Add the server directory to the path to import yolah
sys.path.insert(0, '../server')
from yolah import Yolah, Move, Cell

# Board configuration
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']

# Color scheme
COLOR_LIGHT_SQUARE = (240, 217, 181)  # Light brown
COLOR_DARK_SQUARE = (181, 136, 99)    # Dark brown
COLOR_EMPTY_SQUARE = (50, 50, 50)     # Dark grey for empty cells
COLOR_BLACK_PIECE = (30, 30, 30)      # Almost black
COLOR_WHITE_PIECE = (240, 240, 240)   # Almost white
COLOR_PIECE_OUTLINE_BLACK = (240, 240, 240)  # White outline for black pieces
COLOR_PIECE_OUTLINE_WHITE = (30, 30, 30)     # Black outline for white pieces
COLOR_ARROW = (255, 140, 0)           # Orange for last move arrow
COLOR_TEXT = (0, 0, 0)                # Black for labels
COLOR_BORDER = (0, 0, 0)              # Black border


def draw_board(yolah, last_move=None, board_size=800, show_labels=True):
    """
    Draw the Yolah board with pieces and optional labels.

    Args:
        yolah: Yolah game instance
        last_move: Optional Move to highlight with an arrow
        board_size: Size of the board in pixels
        show_labels: Whether to show file/rank labels

    Returns:
        PIL Image object
    """
    # Calculate dimensions
    label_margin = 40 if show_labels else 0
    total_size = board_size + 2 * label_margin
    square_size = board_size // Yolah.DIM
    piece_margin = square_size // 8

    # Create image
    img = Image.new('RGB', (total_size, total_size), 'white')
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   label_margin // 2)
    except:
        font = ImageFont.load_default()

    # Get grid
    grid = yolah.grid()

    # Draw squares and pieces
    for i in range(Yolah.DIM):
        for j in range(Yolah.DIM):
            # Calculate position (flip vertically for display)
            # Board coordinates: rank 1 at bottom, rank 8 at top
            display_row = Yolah.DIM - 1 - i
            x = label_margin + j * square_size
            y = label_margin + display_row * square_size

            # Draw square
            color = COLOR_LIGHT_SQUARE if (i + j) % 2 == 0 else COLOR_DARK_SQUARE
            draw.rectangle([x, y, x + square_size, y + square_size],
                          fill=color, outline=COLOR_BORDER)

            # Draw piece or empty cell
            cell = grid[i][j]
            if cell == Cell.BLACK:
                # Black piece
                draw.ellipse([x + piece_margin, y + piece_margin,
                            x + square_size - piece_margin,
                            y + square_size - piece_margin],
                           fill=COLOR_BLACK_PIECE,
                           outline=COLOR_PIECE_OUTLINE_BLACK,
                           width=2)
            elif cell == Cell.WHITE:
                # White piece
                draw.ellipse([x + piece_margin, y + piece_margin,
                            x + square_size - piece_margin,
                            y + square_size - piece_margin],
                           fill=COLOR_WHITE_PIECE,
                           outline=COLOR_PIECE_OUTLINE_WHITE,
                           width=2)
            elif cell == Cell.EMPTY:
                # Empty cell (no piece can be placed here)
                draw.rectangle([x, y, x + square_size, y + square_size],
                             fill=COLOR_EMPTY_SQUARE, outline=COLOR_BORDER)

    # Draw labels if requested
    if show_labels:
        for i, file_label in enumerate(FILES):
            # File labels (a-h) at bottom
            x = label_margin + i * square_size + square_size // 2
            y = total_size - label_margin // 2
            draw.text((x, y), file_label, fill=COLOR_TEXT, font=font, anchor="mm")

        for i, rank_label in enumerate(RANKS):
            # Rank labels (1-8) on left
            x = label_margin // 2
            y = label_margin + (Yolah.DIM - 1 - i) * square_size + square_size // 2
            draw.text((x, y), rank_label, fill=COLOR_TEXT, font=font, anchor="mm")

    # Draw arrow for last move if provided
    if last_move is not None:
        from_i, from_j = last_move.from_sq.to_coordinates()
        to_i, to_j = last_move.to_sq.to_coordinates()

        # Calculate arrow positions (flip vertically)
        from_display_row = Yolah.DIM - 1 - from_i
        to_display_row = Yolah.DIM - 1 - to_i

        x1 = label_margin + from_j * square_size + square_size // 2
        y1 = label_margin + from_display_row * square_size + square_size // 2
        x2 = label_margin + to_j * square_size + square_size // 2
        y2 = label_margin + to_display_row * square_size + square_size // 2

        # Draw arrow
        arrow_width = max(3, square_size // 15)
        draw.line([x1, y1, x2, y2], fill=COLOR_ARROW, width=arrow_width)

        # Draw arrowhead
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_size = square_size // 3

        # Calculate arrowhead points
        left_angle = angle + math.pi * 5 / 6
        right_angle = angle - math.pi * 5 / 6

        left_x = x2 + arrow_size * math.cos(left_angle)
        left_y = y2 + arrow_size * math.sin(left_angle)
        right_x = x2 + arrow_size * math.cos(right_angle)
        right_y = y2 + arrow_size * math.sin(right_angle)

        draw.polygon([(x2, y2), (left_x, left_y), (right_x, right_y)],
                    fill=COLOR_ARROW)

    return img


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


def generate_board_diagram(moves_str=None, output_file="board.png",
                          board_size=800, show_labels=True, show_last_move=True):
    """
    Generate a board diagram and save it to a file.

    Args:
        moves_str: String of moves (e.g., "a1:b1 b1:b5"), or None for initial position
        output_file: Path to save the PNG file
        board_size: Size of the board in pixels
        show_labels: Whether to show file/rank labels
        show_last_move: Whether to show an arrow for the last move
    """
    # Create Yolah game
    yolah = Yolah()
    yolah.reset()

    # Parse and play moves
    moves = parse_moves(moves_str) if moves_str else []
    last_move = None

    for move in moves:
        try:
            yolah.play(move)
            last_move = move
        except Exception as e:
            print(f"Warning: Could not play move {move}: {e}")
            break

    # Draw board
    last_move_to_show = last_move if (show_last_move and last_move) else None
    img = draw_board(yolah, last_move_to_show, board_size, show_labels)

    # Save image
    img.save(output_file)
    print(f"Board diagram saved to {output_file}")

    # Print game state info
    black_score, white_score = yolah.black_score, yolah.white_score
    print(f"Moves played: {len(moves)}")
    print(f"Current turn: {'WHITE' if yolah.current_player() == Yolah.WHITE_PLAYER else 'BLACK'}")
    print(f"Black score: {black_score}, White score: {white_score}")
    if yolah.game_over():
        print("Game is over!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Yolah board diagrams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate initial board
  python generate_board_diagram.py --output initial_board.png

  # Generate board from moves
  python generate_board_diagram.py --moves "a1:b1 b1:b5 c8:c7" --output board.png

  # Without labels
  python generate_board_diagram.py --moves "a1:b1" --no-labels --output board.png

  # Custom size
  python generate_board_diagram.py --size 1200 --output large_board.png
        """
    )

    parser.add_argument('--moves', type=str, default=None,
                       help='Space-separated moves in format "a1:b1 b1:b5 ..."')
    parser.add_argument('--output', '-o', type=str, default='board.png',
                       help='Output PNG file path (default: board.png)')
    parser.add_argument('--size', type=int, default=800,
                       help='Board size in pixels (default: 800)')
    parser.add_argument('--no-labels', action='store_true',
                       help='Hide file/rank labels')
    parser.add_argument('--no-arrow', action='store_true',
                       help='Hide last move arrow')

    args = parser.parse_args()

    generate_board_diagram(
        moves_str=args.moves,
        output_file=args.output,
        board_size=args.size,
        show_labels=not args.no_labels,
        show_last_move=not args.no_arrow
    )


if __name__ == '__main__':
    main()
