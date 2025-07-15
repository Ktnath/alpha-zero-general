import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import csv

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from lonelybot_py import analyze_state_py

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.prediction_values = []
        self.episode_results = []

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is the predicted value
                           adjusted by intermediate rewards.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        max_steps = 1000  # Limite pour éviter les parties infinies
        cumulative_reward = 0

        while True:
            episodeStep += 1
            if episodeStep > max_steps:
                log.warning(f"Game exceeded {max_steps} steps, forcing end with result -1")
                self.episode_results.append(-1.0)
                return [(x[0], x[2], -1 * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            try:
                _, v_pred = self.nnet.predict(canonicalBoard)
                self.prediction_values.append(float(v_pred))
                if episodeStep % 100 == 0:
                    log.info(f"Step {episodeStep}, predicted value: {v_pred:.3f}, cumulative reward: {cumulative_reward:.3f}")
            except Exception as e:
                log.error(f"Prediction error at step {episodeStep}: {str(e)}")
                pass

            temp = int(episodeStep < self.args.tempThreshold)
            try:
                pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
                if episodeStep % 100 == 0:
                    log.info(f"Step {episodeStep}, policy entropy: {-np.sum(pi * np.log(pi + 1e-8)):.3f}")
            except Exception as e:
                log.error(f"MCTS error at step {episodeStep}: {str(e)}")
                self.episode_results.append(-1.0)
                return [(x[0], x[2], -1 * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                # Ajuster la valeur prédite avec la récompense cumulée
                adjusted_value = v_pred + cumulative_reward
                trainExamples.append([b, self.curPlayer, p, adjusted_value])

            try:
                action = np.random.choice(len(pi), p=pi)
                next_board, next_player = self.game.getNextState(board, self.curPlayer, action)
                
                # Calculer la récompense intermédiaire basée sur l'analyse d'état
                unknown_cards, remaining_cards, blocked_cols, mobility, deadlock_risk = analyze_state_py(self.game.state)
                intermediate_reward = 0
                
                # Bonus pour les cartes révélées (moins de cartes inconnues)
                revealed_cards = max(0, 24 - unknown_cards)  # 24 est le nombre total de cartes cachées au début
                intermediate_reward += 0.15 * revealed_cards  # Bonus constant pour la révélation
                
                # Bonus exponentiel pour la mobilité
                mobility_bonus = 0.3 * (2 ** (mobility - 1)) if mobility > 1 else 0
                intermediate_reward += min(mobility_bonus, 1.0)  # Plafonné à 1.0
                
                # Bonus pour les colonnes non bloquées
                unblocked_cols = 7 - blocked_cols  # 7 est le nombre total de colonnes
                intermediate_reward += 0.25 * unblocked_cols  # Bonus linéaire pour chaque colonne libre
                
                # Pénalité pour le risque d'impasse avec seuil progressif
                if deadlock_risk > 0.3:
                    penalty = 0.5 * (deadlock_risk - 0.3) ** 2  # Pénalité quadratique
                    intermediate_reward -= penalty
                
                # Bonus pour les cartes restantes (progression vers la victoire)
                remaining_bonus = 0.2 * (1 - len(remaining_cards) / 52)  # 52 cartes au total
                intermediate_reward += remaining_bonus
                
                # Bonus de continuation
                intermediate_reward += 0.05  # Petit bonus constant
                
                cumulative_reward += intermediate_reward
                board, self.curPlayer = next_board, next_player
                
            except Exception as e:
                log.error(f"Action selection/execution error at step {episodeStep}: {str(e)}")
                self.episode_results.append(-1.0)
                return [(x[0], x[2], -1 * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                log.info(f"Game ended after {episodeStep} steps with result {r}, final cumulative reward: {cumulative_reward:.3f}")
                self.episode_results.append(float(r))
                # Ajuster la valeur finale avec la récompense cumulée
                final_value = r + cumulative_reward
                return [(x[0], x[2], final_value * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        # Phase 1: Apprentissage par imitation si activé et si des parties d'expert sont disponibles
        if hasattr(self.args, 'use_imitation') and self.args.use_imitation:
            try:
                from klondike.ImitationLearning import ImitationLearner
                imitation_learner = ImitationLearner(self.nnet, self.args)
                if hasattr(self.args, 'expert_games_file'):
                    success = imitation_learner.run_imitation_learning(self.args.expert_games_file)
                    if success:
                        log.info("✅ Phase d'imitation terminée avec succès")
                    else:
                        log.warning("⚠️ Phase d'imitation terminée avec des erreurs")
                else:
                    log.warning("⚠️ Apprentissage par imitation activé mais aucun fichier de parties d'expert spécifié")
            except Exception as e:
                log.error(f"❌ Erreur lors de l'apprentissage par imitation: {str(e)}")

        # Phase 2: Apprentissage par renforcement
        log.info("🎮 Démarrage de l'apprentissage par renforcement")
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print(f"\n🔁 Iteration {i}/{self.args.numIters}")
            self.prediction_values = []
            self.episode_results = []
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for eps in range(self.args.numEps):
                    print(f"  🤖 Self-play game {eps+1}/{self.args.numEps}")
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            print("  🎯 Training complete.")
            nmcts = MCTS(self.game, self.nnet, self.args)

            print("  ⚔️  Pitting new model against previous...")
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            print(f"✅ New model won {nwins}/{self.args.arenaCompare} games")

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

            win_rate = 0.0 if (pwins + nwins) == 0 else 100.0 * nwins / (pwins + nwins)
            avg_value = float(np.mean(self.prediction_values)) if self.prediction_values else 0.0
            log_file = os.path.join(os.getcwd(), 'training_log.csv')
            write_header = not os.path.isfile(log_file)
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile,
                                       fieldnames=['iteration', 'win_rate', 'new_wins', 'old_wins', 'draws', 'avg_value'])
                if write_header:
                    writer.writeheader()
                writer.writerow({'iteration': i,
                                 'win_rate': f"{win_rate:.2f}",
                                 'new_wins': nwins,
                                 'old_wins': pwins,
                                 'draws': draws,
                                 'avg_value': f"{avg_value:.4f}"})

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
