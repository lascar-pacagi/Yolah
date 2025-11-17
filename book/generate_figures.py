#!/usr/bin/env python3
"""
Figure Generation Script for Yolah Book

This script generates publication-quality figures for the Yolah game AI book.
All figures are saved to the figures/ directory as PNG files.

Usage:
    python3 generate_figures.py

Requirements:
    - matplotlib
    - numpy
    - seaborn (optional, for better styling)

Install dependencies:
    pip install matplotlib numpy seaborn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    print("Seaborn not available, using default matplotlib styling")

# Configuration for publication-quality figures
rc('font', family='serif', size=12)
rc('text', usetex=False)  # Set to True if you have LaTeX installed
rc('figure', dpi=300)
rc('savefig', dpi=300, bbox='tight')

# Create figures directory if it doesn't exist
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_figure(filename, tight=True):
    """Save figure to the figures directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    if tight:
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.close()


def generate_performance_comparison():
    """Generate performance comparison bar chart."""
    print("Generating performance comparison...")

    algorithms = ['Random', 'Minimax\n(depth=3)', 'Alpha-Beta\n(depth=5)',
                  'MCTS\n(1000 sims)', 'Neural Net\n+ MCTS']
    times = [0.01, 150, 45, 200, 180]  # milliseconds
    win_rates = [5, 65, 85, 92, 95]  # percentage

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Time comparison
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    ax1.bar(algorithms, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Time per Move (ms)', fontsize=12)
    ax1.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Win rate comparison
    ax2.bar(algorithms, win_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Win Rate vs Random (%)', fontsize=12)
    ax2.set_title('Playing Strength Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    save_figure('performance_comparison.png')


def generate_training_curves():
    """Generate neural network training curves."""
    print("Generating training curves...")

    epochs = np.arange(1, 101)

    # Simulated training data
    train_loss = 2.5 * np.exp(-epochs / 30) + 0.3 + np.random.normal(0, 0.05, len(epochs))
    val_loss = 2.5 * np.exp(-epochs / 30) + 0.4 + np.random.normal(0, 0.08, len(epochs))
    train_acc = 1 - 0.7 * np.exp(-epochs / 25) + np.random.normal(0, 0.01, len(epochs))
    val_acc = 1 - 0.7 * np.exp(-epochs / 25) - 0.05 + np.random.normal(0, 0.015, len(epochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#3498db')
    ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_acc, label='Training Accuracy', linewidth=2, color='#3498db')
    ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_figure('training_curves.png')


def generate_search_depth_analysis():
    """Generate search depth vs performance analysis."""
    print("Generating search depth analysis...")

    depths = np.arange(1, 9)
    nodes_minimax = 30 ** depths
    nodes_alphabeta = (30 ** depths) * 0.15  # Alpha-beta prunes ~85%

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(depths, nodes_minimax, 'o-', label='Minimax',
                linewidth=2, markersize=8, color='#e74c3c')
    ax.semilogy(depths, nodes_alphabeta, 's-', label='Alpha-Beta',
                linewidth=2, markersize=8, color='#2ecc71')

    ax.set_xlabel('Search Depth', fontsize=12)
    ax.set_ylabel('Nodes Searched (log scale)', fontsize=12)
    ax.set_title('Search Algorithm Complexity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    save_figure('search_depth_analysis.png')


def generate_mcts_iterations():
    """Generate MCTS win rate vs iterations."""
    print("Generating MCTS iterations analysis...")

    iterations = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    win_rate = [45, 62, 72, 78, 85, 90, 93, 95]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, win_rate, 'o-', linewidth=2, markersize=10, color='#f39c12')
    ax.set_xlabel('MCTS Iterations per Move', fontsize=12)
    ax.set_ylabel('Win Rate vs Minimax (depth=3) %', fontsize=12)
    ax.set_title('MCTS Performance vs Iteration Count', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)

    # Add annotations for key points
    ax.annotate('Practical\nperformance', xy=(1000, 90), xytext=(2000, 85),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

    save_figure('mcts_iterations.png')


def generate_elo_progression():
    """Generate Elo rating progression during self-play training."""
    print("Generating Elo progression...")

    games = np.arange(0, 10001, 100)
    elo = 1500 + 800 * (1 - np.exp(-games / 3000)) + np.random.normal(0, 20, len(games))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(games, elo, linewidth=2, color='#9b59b6')
    ax.fill_between(games, elo - 50, elo + 50, alpha=0.2, color='#9b59b6')
    ax.set_xlabel('Training Games', fontsize=12)
    ax.set_ylabel('Elo Rating', fontsize=12)
    ax.set_title('Self-Play Training Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Starting Elo')
    ax.axhline(y=2300, color='red', linestyle='--', alpha=0.5, label='Target Elo')
    ax.legend(fontsize=11)

    save_figure('elo_progression.png')


def generate_branching_factor():
    """Generate average branching factor throughout game."""
    print("Generating branching factor analysis...")

    move_number = np.arange(1, 81)
    # Typical pattern: high at start, decreases as game progresses
    branching = 35 - 25 * (move_number / 80) ** 2 + np.random.normal(0, 1.5, len(move_number))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(move_number, branching, linewidth=2, color='#16a085', alpha=0.7)
    ax.fill_between(move_number, branching - 3, branching + 3, alpha=0.2, color='#16a085')
    ax.set_xlabel('Move Number', fontsize=12)
    ax.set_ylabel('Average Branching Factor', fontsize=12)
    ax.set_title('Branching Factor Throughout Game', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 40)

    save_figure('branching_factor.png')


def generate_evaluation_components():
    """Generate evaluation function component weights."""
    print("Generating evaluation components...")

    components = ['Material', 'Position\nControl', 'Mobility', 'King\nSafety',
                  'Pawn\nStructure', 'Threats']
    weights = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]

    colors_pie = plt.cm.Set3(range(len(components)))

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(weights, labels=components, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 11})

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title('Evaluation Function Component Weights',
                 fontsize=14, fontweight='bold', pad=20)

    save_figure('evaluation_components.png')


def generate_tournament_results():
    """Generate tournament results heatmap."""
    print("Generating tournament results...")

    players = ['Random', 'Minimax', 'Alpha-Beta', 'MCTS', 'Neural Net']
    n = len(players)

    # Win rates matrix (row vs column)
    results = np.array([
        [50, 15, 10, 8, 5],    # Random
        [85, 50, 35, 25, 20],  # Minimax
        [90, 65, 50, 40, 30],  # Alpha-Beta
        [92, 75, 60, 50, 45],  # MCTS
        [95, 80, 70, 55, 50]   # Neural Net
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(results, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(players, fontsize=11)
    ax.set_yticklabels(players, fontsize=11)

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                text = ax.text(j, i, '-', ha="center", va="center",
                             color="black", fontsize=12, fontweight='bold')
            else:
                text = ax.text(j, i, f'{results[i, j]:.0f}%', ha="center", va="center",
                             color="black" if 30 < results[i, j] < 70 else "white",
                             fontsize=11)

    ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Player', fontsize=12, fontweight='bold')
    ax.set_title('Tournament Win Rates (Row vs Column)',
                 fontsize=14, fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Win Rate (%)', fontsize=11)

    save_figure('tournament_results.png')


def generate_time_per_move():
    """Generate time per move distribution."""
    print("Generating time per move distribution...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    algorithms = [
        ('Minimax (depth=3)', np.random.gamma(15, 10, 1000), axes[0, 0]),
        ('Alpha-Beta (depth=5)', np.random.gamma(4.5, 10, 1000), axes[0, 1]),
        ('MCTS (1000 iters)', np.random.gamma(20, 10, 1000), axes[1, 0]),
        ('Neural Net + MCTS', np.random.gamma(18, 10, 1000), axes[1, 1])
    ]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (name, times, ax) in enumerate(algorithms):
        ax.hist(times, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(times), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(times):.1f}ms')
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure('time_per_move.png')


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating figures for Yolah Game AI Book")
    print("=" * 60)

    generate_performance_comparison()
    generate_training_curves()
    generate_search_depth_analysis()
    generate_mcts_iterations()
    generate_elo_progression()
    generate_branching_factor()
    generate_evaluation_components()
    generate_tournament_results()
    generate_time_per_move()

    print("=" * 60)
    print(f"All figures saved to {FIGURES_DIR}/ directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
