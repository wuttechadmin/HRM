import numpy as np
import os

train_path = os.path.join('data', 'sudoku-extreme-1k-aug-1000', 'train')
test_path = os.path.join('data', 'sudoku-extreme-1k-aug-1000', 'test')

# Load training data
inputs = np.load(os.path.join(train_path, 'all__inputs.npy'))
labels = np.load(os.path.join(train_path, 'all__labels.npy'))

print(f"Training data shape: {inputs.shape}")

# Check difficulty by counting non-zero entries (clues)
clue_counts = [np.count_nonzero(inputs[i]) for i in range(min(1000, len(inputs)))]
avg_clues = sum(clue_counts) / len(clue_counts)
min_clues = min(clue_counts)
max_clues = max(clue_counts)

print(f"Average clues per puzzle: {avg_clues:.2f}")
print(f"Min clues: {min_clues}, Max clues: {max_clues}")

# Print distribution of clue counts
from collections import Counter
clue_distribution = Counter(clue_counts)
print("\nClue distribution (top 10):")
for clues, count in sorted(clue_distribution.items())[:10]:
    print(f"  {clues} clues: {count} puzzles")

# Examine a few samples
print("\nSample puzzles:")
for i in range(3):
    sample = inputs[i]
    clue_count = np.count_nonzero(sample)
    print(f"\nSample {i} - {clue_count} clues:")
    grid = sample.reshape(9, 9)
    for r in range(9):
        print(' '.join(['.' if x == 0 else str(x) for x in grid[r]]))
