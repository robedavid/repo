import random
from connect_k_rl.env import ConnectKEnv


def random_game(**kwargs):
    game = ConnectKEnv(**kwargs)

    while True:
        # game.render()
        cands = game.legal_actions
        a = random.choice(cands.tolist())
        a = divmod(int(a), game.ncols)
        res = game.step(*a)
        if res.done:
            print(f"------- Game done, winner is {game.winner}, reward is {res.reward} -------")
            game.render()
            return


if __name__ == "__main__":
    random_game()
