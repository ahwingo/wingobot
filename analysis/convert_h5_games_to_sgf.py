"""
This utility loads data stored in the custom h5 file format and converts it into the standard SGF format.
"""
# Python Standard
import argparse
import os

# Third Party
import h5py


def convert_games(h5_gamefile, output_dir):
    """
    Given an H5 file and a directory to write output to, load all games from the file and write them in SGF format.
    """
    move_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e",
                5: "f", 6: "g", 7: "h", 8: "i", 9: "j",
                10: "k", 11: "l", 12: "m", 13: ""}
    header_part1 = "(;GM[1]FF[4]CA[UTF-8]RU[Chinese]SZ[13]KM[7.5]TM[600]"
    header_part1 += "PW[wingobot]PB[wingobot]WR[00]BR[00]DT[2020-07-30]PC[wingo-desktop]RE["
    header_part2 = "]GN[007];"
    h5_file = h5py.File(h5_gamefile, "r")
    games = h5_file["games"]
    for game in games:
        outfile = os.path.basename(h5_gamefile).split(".")[0] + "_" + game + ".sgf"
        outfile_path = os.path.join(output_dir, outfile)
        game_data = h5_file["games"][game]
        moves = game_data["moves"]
        result = game_data["outcome"][()]
        sgf_string = header_part1 + str(result) + header_part2
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


def main():
    # Collect user arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="../self_play_games/h5_games")
    parser.add_argument("-f", "--game_file", default="batch_0.h5")
    parser.add_argument("-o", "--output_dir", default="self_play_games/sgf_from_h5")
    args = parser.parse_args()
    data_dir = args.data_dir
    game_file = os.path.join(data_dir, args.game_file)
    output_dir = os.path.join(args.output_dir, os.path.splitext(args.game_file)[0])
    os.makedirs(output_dir, exist_ok=True)
    convert_games(game_file, output_dir)


if __name__ == "__main__":
    main()