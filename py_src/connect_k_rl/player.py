from connect_k_rl.env import ConnectKEnv


def random_game(**kwargs):
    import random

    game = ConnectKEnv(**kwargs)

    while True:
        # game.render()
        cands = game.legal_actions
        res = game.step(random.choice(cands.tolist()))
        if res.done:
            print(f"------- Game done, winner is {game.winner}, reward is {res.reward} -------")
            game.render()
            return


def train_tabular(**kwargs):
    from rl.tabq_learner import TabularQLearner

    game = ConnectKEnv(**kwargs)
    player = TabularQLearner()
    player.train(game=game, eval_every=5_000)


def train_nn(**kwargs):
    from rl.dqn_learner import DQNAgent

    game = ConnectKEnv(**kwargs)
    player = DQNAgent(n_actions=game.nrows * game.ncols)
    player.train(game=game, eval_every=5_000)


if __name__ == "__main__":
    train_nn()
