from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StepResult:
    reward: float
    done: bool


class RLEnv(ABC):

    @abstractmethod
    def start_new_game(self):
        pass

    @property
    def legal_actions(self) -> [int]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> StepResult:
        pass

    @abstractmethod
    def folded_obs(self, player: int = None) -> bytes:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def new_turn(self):
        pass
