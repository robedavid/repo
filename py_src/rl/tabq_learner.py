from collections import defaultdict
from rl.agent_base import RLAgent, ScoredAction
import numpy as np
from typing import final

from rl.env_base import RLEnv, StepResult


class TabularQLearner(RLAgent):
    def __init__(self, alpha: float = 0.2, gamma: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(float)

    @classmethod
    def make_key(cls, obs):
        try:
            res = tuple(obs.reshape(-1).tolist())
        except:
            res = tuple(obs)
        return hash(res)

    @final
    def optimal_choice(self, game: RLEnv, player: int = None) -> ScoredAction:
        res = ScoredAction()
        state_key = self.make_key(game.folded_obs(player=player))
        for action in np.random.permutation(game.legal_actions):
            sc = self.Q[(state_key, action)]
            if sc > res.score or res.action is None:
                res.action = action
                res.score = sc
        return res

    @final
    def update(self, action: int, pre_action_folded_board: [int], res: StepResult, post_update_game: RLEnv):
        state_key = self.make_key(pre_action_folded_board)
        old_val = self.Q[(state_key, action)]
        target = res.reward
        if not res.done:
            # Look at the game from the perspective of the opponent, and picks its best potential choice
            best_next_score = self.optimal_choice(game=post_update_game, player=-post_update_game.current_player).score
            target -= self.gamma * best_next_score
        self.Q[(state_key, action)] += self.alpha * (target - old_val)
