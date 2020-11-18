"""
This script will generate stats on SGF games stored in a single directory.
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sgf_dir", default="self_play_games/sgf")
args = parser.parse_args()

all_files = [os.path.join(args.sgf_dir, f) for f in os.listdir(args.sgf_dir) if f.endswith("sgf")]

total_white_wins = 0
white_win_files = []
total_white_margin = 0

total_black_wins = 0
black_win_files = []
total_black_margin = 0

for sgf_file in all_files:
    file_num = sgf_file.split("/")[-1].split("_")[1][:-4]
    with open(sgf_file, "r") as sgf_f:
        header = sgf_f.readlines()[0].strip().split(";")[1]
        result = header.split("RE")[1].split("GN")[0]
        winner = result[1]
        margin = float(result.split("+")[1][:-1])
        if winner == "B":
            total_black_wins += 1
            black_win_files.append(file_num)
            total_black_margin += margin
        else:
            total_white_wins += 1
            white_win_files.append(file_num)
            total_white_margin += margin

print("Total White Wins: ", total_white_wins, " Average Margin of Victory: ", total_white_margin / max(1, total_white_wins))
print("Total Black Wins: ", total_black_wins, " Average Margin of Victory: ", total_black_margin / max(1, total_black_wins))

