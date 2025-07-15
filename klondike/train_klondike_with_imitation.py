import torch
from Coach import Coach
from game.KlondikeGame import KlondikeGame
from klondike.pytorch.NNet import NNetWrapper as nn
from utils import *
from klondike.generate_expert_games import generate_expert_games
from klondike.ImitationLearning import ImitationLearner

def main():
    # Configuration de l'apprentissage
    args = dotdict({
        'numIters': 100,
        'numEps': 100,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 50,
        'arenaCompare': 40,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('./temp/', 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        'useImitationLearning': True,
        'numExpertGames': 1000000,  # Augmenté à 1 million
        'expertGamesFile': 'expert_games.jsonl',
        'lr': 0.001,
        'epochs': 20,  # Augmenté pour mieux apprendre des données
        'batch_size': 128,  # Augmenté pour un apprentissage plus efficace
        'cuda': torch.cuda.is_available()
    })

    # Initialisation du jeu et du réseau
    g = KlondikeGame()
    nnet = nn(g)

    # Apprentissage par imitation si activé
    if args.useImitationLearning:
        print("🎓 Démarrage de l'apprentissage par imitation...")
        
        # Génération des parties d'expert
        print(f"📝 Génération de {args.numExpertGames} parties d'expert...")
        generate_expert_games(args.numExpertGames, args.expertGamesFile)
        
        # Création de l'apprenant par imitation
        imitator = ImitationLearner(nnet, args)
        
        # Lancement de l'apprentissage par imitation
        success = imitator.run_imitation_learning(args.expertGamesFile)
        
        if not success:
            print("❌ Échec de l'apprentissage par imitation. Passage à l'apprentissage par renforcement.")
    
    # Création et lancement de l'entraîneur
    c = Coach(g, nnet, args)
    
    # Lancement de l'apprentissage par renforcement
    print("🎮 Démarrage de l'apprentissage par renforcement...")
    c.learn()

if __name__ == "__main__":
    main()