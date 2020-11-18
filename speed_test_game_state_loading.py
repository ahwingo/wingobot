from gooop import Goban
import numpy as np
import random
import os


batch_size = 2048
num_batches = 16
last_self_play_batch = 500
data_dir = "self_play_games/self_play_batches"
num_game_batches = 4096 // 128
current_batch_range = range(last_self_play_batch - num_game_batches + 1, last_self_play_batch + 1, 1)
top_level_dirs = [os.path.join(data_dir, "batch_" + str(batch_num)) for batch_num in current_batch_range]
all_files = []
for tld in top_level_dirs:
    all_files.extend([os.path.join(tld, f) for f in os.listdir(tld) if f.endswith("sgf")])

for batch_num in range(num_batches):
    train_batch = []
    print("starting a batch")
    for _ in range(batch_size):
        game = Goban(13)
        game.load_game_from_sgf(random.choice(all_files))

print(len(all_files))
