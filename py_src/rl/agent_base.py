from abc import abstractmethod
from dataclasses import dataclass
from typing import final
import random
import numpy as np
from rl.env_base import RLEnv, StepResult


@dataclass
class ScoredAction:
    action: int = None
    score: float = 0


class RLAgent:
    def __init__(self, eps_start: float = 1, eps_end: float = 0.05, eps_decay: float = 0.9995):
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    @final
    def _decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    @classmethod
    def agent_duel(cls, lhs, rhs, game: RLEnv, n_games: int):
        out = {"lhs": 0, "rhs": 0, "draw": 0}
        for ep in range(n_games):
            obs = game.reset()
            players = [lhs, rhs] if ep % 2 == 0 else [rhs, lhs]
            is_done = False
            while not is_done:
                for idx, player in enumerate(players):
                    res = game.step(player.optimal_choice(game).action)
                    if res.done:
                        is_done = True
                        if res.reward > 0:
                            out["lhs" if idx == (ep % 2) else "rhs"] += 1
                        else:
                            out["draw"] += 1
                        break

        return {k: v / n_games for k, v in out.items()}

    @final
    def train(self, game: RLEnv, episodes: int = 50_000, eval_every: int = 0):
        from tqdm import tqdm

        for ep in tqdm(range(episodes)):
            game.reset()
            while True:
                cur_obs = game.obs
                if random.random() < self.eps:
                    action = random.choice(game.legal_actions.tolist())
                else:
                    action = self.optimal_choice(game=game).action
                res = game.step(action)
                self.update(action=action, cur_obs=cur_obs, res=res, game=game)
                if res.done:
                    break

            self._decay_eps()
            if eval_every > 0 and ep % eval_every == 0 and ep > 0:
                from rl.random_learner import RandomLearner

                winrate = self.agent_duel(self, RandomLearner(), game=game, n_games=200)
                print(f"Episode {ep} | eps={self.eps:.3f} | winrate vs random={winrate:.2%}")

    @abstractmethod
    def optimal_choice(self, game: RLEnv) -> ScoredAction:
        pass

    @abstractmethod
    def update(self, action: int, cur_obs: [int], res: StepResult, game: RLEnv):
        pass
