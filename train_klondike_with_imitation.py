import logging
import coloredlogs
import torch
from Coach import Coach
from game.KlondikeGame import KlondikeGame
from klondike.klondikeNNet import NNetWrapper as nn
from utils import *
from klondike.generate_expert_games import generate_expert_games

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

def main():
    # Configuration de l'entraînement
    args = dotdict({
        # Paramètres d'apprentissage par imitation
        'use_imitation': True,
        'expert_games_file': './expert_games.jsonl',
        'generate_expert_games': False,  # On utilise le fichier existant
        'num_expert_games': 1000000,    # Nombre de parties pour l'apprentissage
        
        # Paramètres d'apprentissage par renforcement
        'numIters': 5,                  # Plus d'itérations pour l'apprentissage
        'numEps': 500,                  # Plus de parties de test
        'tempThreshold': 15,
        'updateThreshold': 0.45,         # Seuil encore plus bas pour accepter les nouveaux modèles
        'maxlenOfQueue': 500000,         # Plus grande file d'exemples
        'numMCTSSims': 75,              # Encore plus de simulations MCTS
        'arenaCompare': 200,            # Plus de parties de comparaison pour une meilleure évaluation
        'cpuct': 3,                     # Exploration encore plus agressive

        # Paramètres d'entraînement du réseau
        'lr': 0.0005,                   # Taux d'apprentissage plus faible
        'dropout': 0.4,                 # Plus de régularisation
        'epochs': 20,                   # Plus d'époques d'entraînement
        'batch_size': 128,              # Taille de batch plus grande
        'cuda': torch.cuda.is_available(),
        
        # Paramètres généraux
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
    })

    # Initialisation du jeu et du réseau
    log.info('🎮 Initialisation du jeu de Klondike')
    game = KlondikeGame()

    log.info('🧠 Création du réseau neuronal')
    nnet = nn(game)

    if args.load_model:
        log.info('📥 Chargement du modèle checkpoint...')
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    
    # Génération des parties d'expert si nécessaire
    if args.generate_expert_games:
        log.info('👨‍🏫 Génération des parties d\'expert...')
        generate_expert_games(num_games=args.num_expert_games, output_file=args.expert_games_file)

    # Création du coach et lancement de l'entraînement
    log.info('🎓 Démarrage de l\'entraînement...')
    c = Coach(game, nnet, args)
    
    # Lancement de l'apprentissage
    try:
        c.learn()
    except KeyboardInterrupt:
        log.warning('⚠️ Entraînement interrompu par l\'utilisateur')
    except Exception as e:
        log.error(f'❌ Erreur pendant l\'entraînement: {str(e)}')
    finally:
        # Sauvegarde du modèle final
        log.info('💾 Sauvegarde du modèle final...')
        nnet.save_checkpoint(folder=args.checkpoint, filename='final_model.pth.tar')

if __name__ == "__main__":
    main()