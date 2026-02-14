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
    def __init__(self, eps_start: float = 1, eps_end: float = 0.01, eps_decay: float = 0.9999, name: str = None):
        self.name = name or self.__class__.__name__
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    @final
    def _decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    @classmethod
    def agent_duel(cls, lhs, rhs, game: RLEnv, n_games: int = 1, verbose: bool = False):
        out = {lhs.name: 0, rhs.name: 0, "draw": 0}
        for ep in range(n_games):
            obs = game.start_new_game()
            players = [lhs, rhs] if ep % 2 == 0 else [rhs, lhs]
            is_done = False
            while not is_done:
                for idx, player in enumerate(players):
                    game.new_turn()
                    if verbose:
                        print(f"Player {player.name} playing")
                    # res plays as the current player
                    res = game.step(player.optimal_choice(game).action)
                    if verbose:
                        game.render()
                    if res.done:
                        is_done = True
                        if res.reward > 0:
                            out[player.name] += 1
                            if verbose:
                                print(f"Player {player.name} won!")
                        else:
                            out["draw"] += 1
                        break

        return {k: v / n_games for k, v in out.items()}

    @final
    def train(self, game: RLEnv, episodes: int = 30_000, eval_every: int = 0):
        from tqdm import tqdm

        def foo():
            from rl.random_learner import RandomLearner

            if eval_every <= 0:
                return
            winrate = self.agent_duel(self, RandomLearner(), game=game, n_games=200)
            winrate = winrate[self.name] / sum([v for k, v in winrate.items() if k != "draw"])
            print(f"Episode {ep} | eps={self.eps:.3f} | winrate vs random={winrate:.2%}")

        print(f"Training {self.name} over {episodes} episodes")
        for ep in tqdm(range(episodes)):
            if ep % eval_every == 0:
                foo()

            game.start_new_game()
            while True:
                game.new_turn()  # In practice, flips the player

                pre_action_folded_board = game.folded_obs(player=None)  # Defaults to the current player

                # Explore / Exploit
                if random.random() < self.eps:
                    action = random.choice(game.legal_actions.tolist())
                else:
                    action = self.optimal_choice(game=game).action

                # Apply the action -- This DOES NOT modify the player
                res = game.step(action)

                # Update the Q-function -- At this stage be careful, the game is still from the view of the player that just played
                self.update(action=action, pre_action_folded_board=pre_action_folded_board, res=res, post_update_game=game)

                if res.done:
                    # TODO: Update the opponent as well
                    break
                assert res.reward == 0  # Only at final state do we give a reward

            self._decay_eps()
        foo()

    @abstractmethod
    def optimal_choice(self, game: RLEnv, **kwargs) -> ScoredAction:
        pass

    @abstractmethod
    def update(self, action: int, pre_action_folded_board: [int], res: StepResult, post_update_game: RLEnv):
        pass
