from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StepResult:
    reward: float
    done: bool


class RLEnv(ABC):

    @abstractmethod
    def reset(self):
        pass

    @property
    def legal_actions(self) -> [int]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> StepResult:
        pass

    @property
    def obs(self):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        pass
