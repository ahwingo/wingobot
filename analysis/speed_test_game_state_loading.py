"""
This utility evaluates how fast games can be loaded from the SGF format in batches.
TODO: This is really only interesting if we compare H5 to SGF load times, or evaluate multiprocessing speedups.
"""
# Python Standard
import os
import random
import sys
import time

# Third Party
from matplotlib import pyplot as plt
import numpy as np

# Local
sys.path.append("..")
from source.gooop import Goban


def main():
    """
    Plot the speeds at which different size batches of game training data can be loaded.
    """
    # Set the batch sizes to evaluate.
    batch_sizes = [2**p for p in range(13)]

    # Get the paths to the game data.
    data_dir = "../self_play_games/sgf"
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("sgf")]

    # Run the trials.
    avg_speed_by_batch_size = []
    for batch_size in batch_sizes:
        start_time = time.time()
        for _ in range(batch_size):
            game = Goban(13)
            game.load_game_from_sgf(random.choice(all_files))
        end_time = time.time()
        avg_speed_by_batch_size.append(end_time - start_time)

    # Plot the load times.
    plt.plot(np.asarray(batch_sizes), np.asarray(avg_speed_by_batch_size), label="SGF")
    plt.xlabel("Batch Sizes")
    plt.ylabel("Load Times (Seconds)")
    plt.show()


if __name__ == "__main__":
    main()