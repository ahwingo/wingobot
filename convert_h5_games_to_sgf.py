import os
import h5py
from gooop import Goban

data_dir = "self_play_games/h5_games"
game_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("h5")]

move_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: ""}

output_dir = "self_play_games/sgf_from_h5"


header_part1 = "(;GM[1]FF[4]CA[UTF-8]RU[Chinese]SZ[13]KM[7.5]TM[600]PW[wingobot]PB[wingobot]WR[00]BR[00]DT[2020-07-30]PC[wingo-desktop]RE["

header_part2 = "]GN[007];"


for game_file in game_files:
    h5_file = h5py.File(game_file, "r")
    games = h5_file["games"]
    for game in games:
        outfile = game_file.split("/")[-1].split(".")[0] + "_" + game + ".sgf"
        outfile_path = os.path.join(output_dir, outfile)
        game_data = h5_file["games"][game]
        moves = game_data["moves"]
        #result = "W+1000" if game_data["outcome"][()] == -1 else "B+1000"
        result = game_data["outcome"][()]
        sgf_string = header_part1 + result + header_part2
        player = "B"
        for move in moves:
            row, col = move
            row_code = move_map[row]
            col_code = move_map[col]
            sgf_string = sgf_string + player + "[" + row_code + col_code + "];"
            if player == "B":
                player = "W"
            else:
                player = "B"
        sgf_string = sgf_string + ")"
        with open(outfile_path, "w") as f:
            f.write(sgf_string)
    h5_file.close()


