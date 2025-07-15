import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

class KlondikeNNet(nn.Module):
    def __init__(self, game, args):
        # Réseau de base du jeu
        super(KlondikeNNet, self).__init__()
        
        # Paramètres du réseau
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # Architecture du réseau
        # Couches de convolution pour extraire les caractéristiques
        self.conv1 = nn.Conv2d(1, args['num_channels'], 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1, padding=1)
        
        # Couches de batch normalization
        self.bn1 = nn.BatchNorm2d(args['num_channels'])
        self.bn2 = nn.BatchNorm2d(args['num_channels'])
        self.bn3 = nn.BatchNorm2d(args['num_channels'])
        
        # Calcul de la taille des features après convolution
        self.board_size = self.board_x * self.board_y
        self.conv_size = self.board_size * args['num_channels']
        
        # Couches fully connected pour la politique (pi)
        self.fc1 = nn.Linear(self.conv_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_pi = nn.Linear(512, self.action_size)
        
        # Couches fully connected pour la valeur (v)
        self.fc3 = nn.Linear(self.conv_size, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc_v = nn.Linear(512, 1)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(p=args['dropout'])

    def forward(self, s):
        # Reshape de l'entrée pour la convolution
        s = s.view(-1, 1, self.board_x, self.board_y)
        
        # Convolutions avec activation ReLU et batch normalization
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        
        # Flatten pour les couches fully connected
        s = s.view(-1, self.conv_size)
        
        # Branche de la politique (pi)
        pi = F.relu(self.fc1(s))
        pi = self.dropout(pi)
        pi = F.relu(self.fc2(pi))
        pi = self.dropout(pi)
        pi = self.fc_pi(pi)
        pi = F.log_softmax(pi, dim=1)
        
        # Branche de la valeur (v)
        v = F.relu(self.fc3(s))
        v = self.dropout(v)
        v = F.relu(self.fc4(v))
        v = self.dropout(v)
        v = torch.tanh(self.fc_v(v))
        
        return pi, v