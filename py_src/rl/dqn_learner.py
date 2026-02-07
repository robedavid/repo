from collections import deque
from dataclasses import dataclass
from functools import cached_property
import random
from typing import Final, final
from rl.env_base import RLEnv, StepResult
import torch
from rl.agent_base import RLAgent, ScoredAction
import numpy as np


@dataclass
class _Transition:
    s: np.array
    sp: np.array
    action: int
    reward: float
    done: bool
    legals: [int]


class _DQNMLP(torch.nn.Module):
    def __init__(self, n_actions: int, n: int = 512) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LazyLinear(n), torch.nn.ReLU(), torch.nn.Linear(n, n), torch.nn.ReLU(), torch.nn.Linear(n, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(RLAgent):
    def __init__(
        self, n_actions: int, lr: float = 2.5e-4, warmup: int = 5_000, target_sync_every: int = 2_000, batch_size: int = 128, replay_capacity: int = 200_000
    ):
        super().__init__()
        self.q: [_DQNMLP] = _DQNMLP(n_actions).to(self.device)
        self.tgt: [_DQNMLP] = _DQNMLP(n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.replay: [_Transition] = deque(maxlen=replay_capacity)
        self.warmup: Final[int] = warmup
        self.batch_size: Final[int] = batch_size
        self.target_sync_every: Final[int] = target_sync_every
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_updates = 0

    @cached_property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def obs_to_tensor(cls, obs: bytes):
        flat = np.frombuffer(obs, dtype=np.int8)
        return np.concatenate([flat == 1, flat == -1], axis=0).astype(np.float32)

    @final
    def optimal_choice(self, game: RLEnv) -> ScoredAction:
        x_np = self.obs_to_tensor(game.obs)
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q(x)[0]
            mask = torch.full_like(qvals, float("-inf"))
            mask[game.legal_actions] = 0
            a = int(torch.argmax(qvals + mask).item())
            return ScoredAction(action=a, score=float(qvals[a].item()))

    @property
    def _should_learn(self):
        return len(self.replay) > max(self.warmup, self.batch_size)

    @final
    def update(self, action: int, cur_obs: [int], res: StepResult, game: RLEnv):
        s = self.obs_to_tensor(cur_obs)
        sp = self.obs_to_tensor(game.obs)
        self.replay.append(_Transition(s=s, action=action, reward=res.reward, sp=sp, done=res.done, legals=game.legal_actions if not res.done else list()))
        if not self._should_learn:
            return
        self._learn_batch()
        self.n_updates += 1
        if self.n_updates % self.target_sync_every == 0:
            self.tgt.load_state_dict(self.q.state_dict())

    def _learn_batch(self):
        batch: [_Transition] = random.sample(self.replay, self.batch_size)
        s = torch.from_numpy(np.stack([t.s for t in batch], axis=0)).to(self.device)
        sp = torch.from_numpy(np.stack([t.sp for t in batch], axis=0)).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_sa = self.q(s).gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next_all = self.tgt(sp)
            max_next = torch.empty((self.batch_size,), device=self.device)
            for i, t in enumerate(batch):
                max_next[i] = q_next_all[i][t.legals].max()

            target = rewards + self.gamma * (1.0 - dones) * max_next
            loss = torch.nn.functional.smooth_l1_loss(q_sa, target)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()
