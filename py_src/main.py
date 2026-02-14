from connect_k_rl.env import ConnectKEnv
from rl.agent_base import RLAgent
from rl.env_base import RLEnv
from rl.random_learner import RandomLearner
from rl.tabq_learner import TabularQLearner
from rl.dqn_learner import DQNAgent

if __name__ == "__main__":
    game = ConnectKEnv(nrows=9, ncols=8, k=4, use_gravity=True)
    player = TabularQLearner()
    player.train(game=game, eval_every=5_000, episodes=30_000)
    RLAgent.agent_duel(player, RandomLearner(), game=game, verbose=True)
