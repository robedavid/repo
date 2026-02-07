from rl.env_base import RLEnv, StepResult
import numpy as np


class ConnectKEnv(RLEnv):
    def __init__(self, nrows: int = 9, ncols: int = 8, k: int = 4, use_gravity: bool = True):
        self.nrows = nrows
        self.ncols = ncols or nrows
        self.use_gravity = use_gravity
        self.k = k or nrows
        assert self.k <= self.nrows and self.k <= self.ncols
        self.reset()

    def reset(self):
        self.board = np.zeros((self.nrows, self.ncols), dtype=np.int8)
        self.current_player = 1
        self.nmoves = 0
        self.winner = None
        return self.obs

    @property
    def legal_actions(self) -> np.ndarray:
        if self.use_gravity:
            return np.flatnonzero(self.board[0, :] == 0).astype(np.int32, copy=False)
        return np.flatnonzero(self.board.ravel() == 0).astype(np.int32, copy=False)

    @property
    def obs(self) -> bytes:
        return self.board.astype("int8", copy=False).tobytes()

    @property
    def last_move_wins(self):
        for direction_row, direction_col in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for eps in [-1, 1]:
                for i in range(1, self.k):
                    nr = self.last_move[0] + eps * i * direction_row
                    nc = self.last_move[1] + eps * i * direction_col
                    if 0 <= nr < self.nrows and 0 <= nc < self.ncols and self.board[nr, nc] == self.current_player:
                        count += 1
                        if count >= self.k:
                            return True
                    else:
                        break
        return False

    def step(self, action: int) -> StepResult:
        self.nmoves += 1
        if self.use_gravity:
            empties = np.flatnonzero(self.board[:, action] == 0)
            if len(empties) == 0:
                return StepResult(reward=-1, done=True)
            row, col = int(empties[-1]), action
        else:
            row, col = divmod(action, self.ncols)
        if self.board[row, col] != 0:
            return StepResult(reward=-1, done=True)

        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        if self.last_move_wins:
            self.winner = self.current_player
            return StepResult(reward=1, done=True)

        self.current_player *= -1
        return StepResult(reward=0, done=len(self.legal_actions) == 0)

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = list()
        for r in range(self.nrows):
            lines.append(" ".join(symbols[int(x)] for x in self.board[r]))
        if self.winner is not None:
            print(f"Board played {self.nmoves} moves -- Winner is {symbols[self.winner]}")
        elif self.nmoves == self.ncols * self.nrows:
            print(f"Board played {self.nmoves} moves -- Draw ! ")
        else:
            print(f"Board played {self.nmoves} moves")
        print("\n".join(lines) + "\n")


class TicTacToe(ConnectKEnv):
    def __init__(self):
        super().__init__(nrows=3, ncols=3, k=3, use_gravity=False)
