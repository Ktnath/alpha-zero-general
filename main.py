import logging

import coloredlogs

import argparse
from Coach import Coach
from utils import *
from game.KlondikeGame import KlondikeGame
from klondike.klondikeNNet import NNet as klondikeNNet
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as othelloNNet

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='othello', help='Name of the game to train')
cmd_args, _ = parser.parse_known_args()

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'game': cmd_args.game.lower(),

})


def main():
    if args.game == 'klondike':
        log.info('Loading KlondikeGame...')
        g = KlondikeGame()
        nnet = klondikeNNet(g)
    elif args.game == 'othello':
        log.info('Loading OthelloGame...')
        g = OthelloGame(6)
        nnet = othelloNNet(g)
    else:
        raise ValueError(f"Unknown game {args.game}")

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
