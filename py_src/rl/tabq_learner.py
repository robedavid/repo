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
    def optimal_choice(self, game: RLEnv) -> ScoredAction:
        res = ScoredAction()
        state_key = self.make_key(game.obs)
        for action in np.random.permutation(game.legal_actions):
            sc = self.Q[(state_key, action)]
            if sc > res.score or res.action is None:
                res.action = action
                res.score = sc
        return res

    @final
    def update(self, action: int, cur_obs: [int], res: StepResult, game: RLEnv):
        state_key = self.make_key(cur_obs)
        old_val = self.Q[(state_key, action)]
        target = res.reward
        if not res.done:
            target -= self.gamma * self.optimal_choice(game=game).score
        self.Q[(state_key, action)] = old_val + self.alpha * (target - old_val)
