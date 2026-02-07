import random
from connect_k_rl.env import ConnectKEnv


def random_game(**kwargs):
    game = ConnectKEnv(**kwargs)
    game.render()

    while True:
        cands = game.legal_actions
        a = random.choice(cands.tolist())
        a = divmod(int(a), game.ncols)
        res = game.step(*a)
        game.render()
        if res.done:
            break


if __name__ == "__main__":
    random_game()
