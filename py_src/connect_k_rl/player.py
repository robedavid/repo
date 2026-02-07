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
    # player.train(game=game, eval_every=1_000)
    winrate = player.agent_duel(player, RandomLearner(), game=game, n_games=1_000)
    print(f"winrate vs random={winrate}")


if __name__ == "__main__":
    train_tabular()
