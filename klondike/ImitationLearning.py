import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from tqdm import tqdm
from lonelybot_py import GameState, ranked_moves_py, encode_observation_py, get_valid_actions_py

class ImitationLearner:
    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args

    def load_expert_games(self, games_file: str) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Charge les parties d'expert depuis un fichier."""
        if not os.path.exists(games_file):
            raise FileNotFoundError(f"Fichier de parties d'expert non trouvé: {games_file}")
        
        expert_examples = []
        with open(games_file, 'r') as f:
            for line in f:
                game_record = json.loads(line)
                state = game_record['partial_state']  # Utiliser l'état partiel directement
                
                # Convertir l'état en format de tableau pour le réseau
                board = np.array(encode_observation_py(state)).reshape(1, -1)
                
                # Créer le vecteur de politique à partir du mouvement sélectionné
                policy = np.zeros(215)  # Taille de l'espace d'actions
                valid_actions = get_valid_actions_py(state)
                
                # Assigner une probabilité de 1.0 au mouvement sélectionné
                selected_move = game_record['selected_move']
                for action in valid_actions:
                    if str(action) == selected_move:
                        policy[action] = 1.0
                        break
                
                # Utiliser win comme valeur
                value = 1.0 if game_record['win'] else -1.0
                
                expert_examples.append((board, policy, value))
        
        return expert_examples

    def train_on_expert_games(self, expert_examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Entraîne le réseau sur les parties d'expert."""
        self.nnet.train()
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=self.args.lr)

        batch_count = int(len(expert_examples) / self.args.batch_size)
        for epoch in range(self.args.epochs):
            print(f'Epoch d\'imitation {epoch + 1}/{self.args.epochs}')
            t = tqdm(range(batch_count), desc='Entraînement sur parties d\'expert')
            
            for _ in t:
                sample_ids = np.random.randint(len(expert_examples), size=self.args.batch_size)
                boards, pis, vs = zip(*[expert_examples[i] for i in sample_ids])
                
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                
                # Calcul des pertes
                policy_loss = -torch.mean(torch.sum(target_pis * out_pi, dim=1))
                value_loss = F.mse_loss(out_v.squeeze(), target_vs)
                total_loss = policy_loss + value_loss

                # Mise à jour des poids
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                t.set_postfix(policy_loss=policy_loss.item(), value_loss=value_loss.item())

    def run_imitation_learning(self, expert_games_file: str):
        """Exécute le processus complet d'apprentissage par imitation."""
        print("🎓 Démarrage de l'apprentissage par imitation...")
        
        # Chargement des parties d'expert
        try:
            expert_examples = self.load_expert_games(expert_games_file)
            print(f"✅ {len(expert_examples)} parties d'expert chargées")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des parties d'expert: {str(e)}")
            return False

        # Entraînement sur les parties d'expert
        try:
            self.train_on_expert_games(expert_examples)
            print("✅ Apprentissage par imitation terminé")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'apprentissage par imitation: {str(e)}")
            return False