from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    reward: float
    done: bool


class ConnectKEnv:
    def __init__(self, nrows: int = 3, ncols: int = None, k: int = None):
        self.nrows = nrows
        self.ncols = ncols or nrows
        self.k = k or nrows
        assert self.k <= self.nrows and self.k <= self.ncols
        self.reset()

    def reset(self):
        self.board = np.zeros((self.nrows, self.ncols), dtype=np.int8)
        self.current_player = 1

    @property
    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.board.ravel() == 0).astype(np.int32, copy=False)

    @property
    def last_move_wins(self):
        for direction_row in (-1, 0, 1):
            for direction_col in (-1, 0, 1):
                if direction_col == 0 and direction_row == 0:
                    continue
                max_row_index = self.last_move[0] + (self.k - 1) * direction_row
                if max_row_index < 0 or max_row_index >= self.nrows:
                    continue

                max_col_index = self.last_move[1] + (self.k - 1) * direction_col
                if max_col_index < 0 or max_col_index >= self.ncols:
                    continue

                winning_line = all(
                    self.board[self.last_move[0] + i * direction_row, self.last_move[1] + i * direction_col] == self.board[self.last_move[0], self.last_move[1]]
                    for i in range(self.k)
                )
                if winning_line:
                    return True
        return False

    def step(self, row: int, col: int) -> StepResult:
        if self.board[row, col] != 0:
            return StepResult(reward=-1, done=True)

        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        if self.last_move_wins:
            return StepResult(reward=1, done=True)

        self.current_player *= -1
        return StepResult(reward=0, done=len(self.legal_actions) == 0)

    def render(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = list()
        for r in range(self.nrows):
            lines.append(" ".join(symbols[int(x)] for x in self.board[r]))
        print("\n".join(lines))
