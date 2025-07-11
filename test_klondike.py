from game.KlondikeGame import KlondikeGame
from klondike.klondikeNNet import NNet as KlondikeNNet
from MCTS import MCTS
import numpy as np

# minimal self-play loop to ensure interaction between game and network
if __name__ == "__main__":
    game = KlondikeGame()
    nnet = KlondikeNNet(game)
    mcts = MCTS(game, nnet, args={'numMCTSSims': 25, 'cpuct': 1.0})

    board = game.getInitBoard()
    pi = mcts.getActionProb(board, temp=1)

    print("Initial board:\n", board)
    print("Policy:\n", pi)
    print("Valid moves:\n", game.getValidMoves(board, 1))
