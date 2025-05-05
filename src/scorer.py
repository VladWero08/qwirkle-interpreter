import numpy as np

class Scorer:
    def __init__(self, board_scoring: np.ndarray):
        self.board_scoring = board_scoring

    def get_points(self, board: np.ndarray, new_tiles: list[tuple]) -> int:
        points = 0
        points += self.get_combination_points(board, new_tiles)
        points += self.get_bonus_points(new_tiles)

        return points

    def get_combination_points(self, board: np.ndarray, new_tiles: list[tuple]) -> int:
        # when points are computed based on a certain row / column,
        # it is added to these list to avoid repetition 
        total_points = 0

        rows_marked = []
        cols_marked = []

        for (i, j) in new_tiles:
            if i not in rows_marked:
                row_points = self.get_row_points(i, j, board)
                
                if row_points != 0:
                    total_points += row_points
                    rows_marked.append(i)

            if j not in cols_marked:
                col_points = self.get_col_points(i, j, board)

                if col_points != 0:
                    total_points += col_points
                    cols_marked.append(j)

        return total_points

    def get_row_points(
        self, 
        i: int,
        j: int,
        board: np.ndarray,     
    ) -> int:
        """
        Checks if there is a line starting from position `[i][j]`,
        and if there is, it counts how many tiles it contains.

        Return:
        -------
        no_tiles: int
            Number of tiles forming the line.
        """
        line_score = 0
        j_left = j - 1
        j_right = j + 1
        
        # go to the left part of the row
        while j_left > 0 and board[i][j_left] is not None:
            line_score += 1
            j_left -= 1

        # go to the right part of the row
        while j_right < 15 and board[i][j_right] is not None:
            line_score += 1
            j_right += 1

        # if the tile is not part of any row, 
        # no points were made with this move
        if line_score == 0:
            return 0
            
        # add the current tile to the row score
        line_score += 1

        # check for Qwirkle bonus
        if line_score == 6:
            line_score += 6
            
        return line_score
            

    def get_col_points(
        self,
        i: int,
        j: int,
        board: np.ndarray
    ) -> int:
        """
        Checks if there is a column starting from position `[i][j]`,
        and if there is, it counts how many tiles it contains.

        Return:
        -------
        no_tiles: int
            Number of tiles forming the column.
        """
        col_score = 0
        i_left = i - 1
        i_right = i + 1

        # go to the left part of the row
        while i_left > 0 and board[i_left][j] is not None:
            col_score += 1
            i_left -= 1

        # go to the right part of the row
        while i_right < 15 and board[i_right][j] is not None:
            col_score += 1
            i_right += 1

        # if the tile is not part of any column, 
        # no points were made with this move
        if col_score == 0:
            return 0
        
        # add the current tile to the column score
        col_score += 1

        # check for Qwirkle bonus
        if col_score == 6:
            col_score += 6
            
        return col_score


    def get_bonus_points(self, new_tiles: list[tuple]) -> int:
        """
        Given a list of tuples, which represent the position of
        the new tiles, count how many of the new tiles correspond
        to a tile marked with bonus points.
        """
        bonus_points = 0

        for (i, j) in new_tiles:
            if self.board_scoring[i][j] == 1 or self.board_scoring[i][j] == 2:
                bonus_points += self.board_scoring[i][j]

        return bonus_points