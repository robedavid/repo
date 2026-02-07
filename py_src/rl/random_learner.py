import random
from typing import final
from rl.agent_base import RLAgent, ScoredAction
from rl.env_base import RLEnv


class RandomLearner(RLAgent):
    @final
    def optimal_choice(self, game: RLEnv) -> ScoredAction:
        return ScoredAction(action=random.choice(game.legal_actions.tolist()))

    @final
    def update(self, *args, **kwargs):
        pass
