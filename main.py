import cv2 as cv
import os
import shutil

from src.board import get_board, get_scoring_board, get_board_tiles_positions, get_board_tile_type_initial, get_board_simple_shapes
from src.tiles import get_patches, get_different_tiles, get_tile_type, get_different_tiles, get_tile_color
from src.scorer import Scorer
from src.helpers import show_image

COL_MAP = {i: chr(ord("A") + i) for i in range(16)}
SHAPE_MAP = {"circle": 1, "flower": 2, "diamond": 3, "square": 4, "star": 5, "sun": 6}
COLOR_MAP = {"red": "R", "blue": "B", "green": "G", "yellow": "Y", "orange": "O", "white": "W"}
GAMES = [i for i in range(1, 6) if i != 2]
MOVES_PER_GAME = 20
PADDING = 50

INPUT_DIR = "./train/"
OUTPUT_DIR = "./Olaeriu_Vlad_Mihai_407/"

def interpret_game() -> None:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for game in GAMES:
        image = cv.imread(f"{INPUT_DIR}{game}_00.jpg")
        board = get_board(image)
        board_padded = get_board(image, padding=PADDING)
        patches_curr = get_patches(board)

        # compute the scoring board 
        board_scoring = get_scoring_board(patches_curr)
        # compute the board that contains the tile types
        board_tiles = get_board_tile_type_initial(board_scoring, patches_curr)
        # compute the board that contains the positions where tiles are placed
        board_tiles_pos_curr = get_board_tiles_positions(board_padded, padding=PADDING)

        scorer = Scorer(board_scoring)

        print(f"GAME {game}:")
        print("-" * 10)

        # compute the path to the boards that include the moves
        moves = [f"{INPUT_DIR}{game}_{i:02d}.jpg" for i in range(1, MOVES_PER_GAME + 1)]
        outputs = [f"{OUTPUT_DIR}{game}_{i:02d}.txt" for i in range(1, MOVES_PER_GAME + 1)]

        for move_idx, move_path in enumerate(moves):
            print(f"Processing move {move_idx + 1}")

            # read the board corresponding to the current move
            # and extract the patches with tiles
            image = cv.imread(move_path)
            board = get_board(image, blur=7)
            board_padded = get_board(image, padding=PADDING)
            board_tiles_pos_next = get_board_tiles_positions(board_padded, padding=PADDING)
            patches_next = get_patches(board)

            # search for the new tiles that were added
            new_tiles = get_different_tiles(board_tiles_pos_curr, board_tiles_pos_next)
            new_tiles_simple_shapes = get_board_simple_shapes(board_padded, padding=PADDING)

            output = ""

            for (i, j) in new_tiles:
                if (i, j) in new_tiles_simple_shapes:
                    shape, color = new_tiles_simple_shapes[(i, j)], get_tile_color(patches_next[16 * i + j])
                else:
                    shape, color = get_tile_type(patches_next[16 * i + j])  

                board_tiles[i][j] = (shape, color)
                board_tiles_pos_curr[i][j] = 1
                
                row = i + 1
                col = COL_MAP[j]
                shape = SHAPE_MAP[shape]
                color = COLOR_MAP[color]

                output = output + f"{row}{col} {shape}{color}\n"
            
            score = scorer.get_points(board_tiles, new_tiles)
            output = output + str(score) + "\n"

            with open(outputs[move_idx], "w+") as f:
                f.write(output)

            patches_curr = patches_next

if __name__ == "__main__":
    interpret_game()

    for game in GAMES:
        my_outputs = [f"{OUTPUT_DIR}{game}_{i:02d}.txt" for i in range(1, MOVES_PER_GAME + 1)]
        outputs = [f"{INPUT_DIR}{game}_{i:02d}.txt" for i in range(1, MOVES_PER_GAME + 1)]#
        for idx, _ in enumerate(outputs):
            with open(my_outputs[idx], "r+") as f:
                print(f.readlines())
            with open(outputs[idx], "r+") as g:
                print(g.readlines())
            print()