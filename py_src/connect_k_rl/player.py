from connect_k_rl.env import ConnectKEnv
from rl.agent_base import RLAgent
from rl.env_base import RLEnv
from rl.random_learner import RandomLearner


def play_game(player: RLAgent, game: RLEnv):
    RLAgent.agent_duel(player, RandomLearner(), game=game, verbose=True)
    print(f"------- Game done, winner is {game.winner} -------")
    game.render()


def train_tabular(**kwargs):
    from rl.tabq_learner import TabularQLearner

    game = ConnectKEnv(**kwargs)
    player = TabularQLearner()
    player.train(game=game, eval_every=5_000)
    return player, game


def train_nn(**kwargs):
    from rl.dqn_learner import DQNAgent

    game = ConnectKEnv(**kwargs)
    player = DQNAgent(n_actions=game.nrows * game.ncols)
    player.train(game=game, eval_every=5_000)
    return player, game


if __name__ == "__main__":
    player, game = train_nn()
    play_game(player, game)
