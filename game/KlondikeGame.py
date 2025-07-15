# alpha-zero-general/game/KlondikeGame.py

import numpy as np
import logging
from lonelybot_py import (
    reset,
    step_action,
    get_valid_actions,
    get_game_result,
    get_board_size,
    get_action_size,
    get_canonical_board,
    analyze_state_py,
    HeuristicConfigPy,
    ranked_moves_py,
    legal_actions_py,
)

log = logging.getLogger(__name__)

class KlondikeGame:
    def __init__(self):
        self.board_x, self.board_y = get_board_size()
        self.action_size = get_action_size()
        self.heuristic_config = HeuristicConfigPy(
            reveal_bonus=100,
            empty_column_bonus=50,
            early_foundation_penalty=-30,
            keep_king_bonus=20,
            deadlock_penalty=-100,
            long_column_bonus=10,
            chain_bonus=15,
            aggressive_coef=2,
            conservative_coef=1,
            neutral_coef=1
        )
        self.last_moves = []
        self.repetition_count = 0
        self.max_repetitions = 3

    def getInitBoard(self):
        self.state, board = reset()
        self.last_moves = []
        self.repetition_count = 0
        return board

    def getBoardSize(self):
        return (self.board_x, self.board_y)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board, player, action):
        try:
            # Get ranked moves to evaluate the quality of the action
            moves = ranked_moves_py(self.state, "neutral", self.heuristic_config)
            legal_moves = legal_actions_py(self.state)
            move_scores = {move: info for move, info in zip(legal_moves, moves)}
            
            # Apply the move
            next_state, next_board, reward, done = step_action(self.state, action)
            self.state = next_state
            
            # Adjust reward based on move quality from lonelybot analysis
            for move_str, move_info in move_scores.items():
                if str(action) in move_str:  # Simple string matching since we can't convert action to move string
                    # Bonus for revealing cards
                    if len(move_info["revealed_cards"]) > 0:
                        reward += 2
                    # Bonus for freeing columns
                    if move_info["columns_freed"] > 0:
                        reward += 3
                    # Penalty for moves that might block progress
                    if move_info["will_block"]:
                        reward -= 2
                    # Scale reward by move's win rate
                    reward = int(reward * (1 + move_info["win_rate"]))
                    break
            
            return next_board, -player
        except ValueError as e:
            log.error(f"Invalid move {action}: {str(e)}")
            # En cas d'action invalide, retourner l'état actuel sans changement
            return board, -player

    def getValidMoves(self, board, player):
        # Get base valid actions
        valid_actions = get_valid_actions(self.state)
        valids = np.zeros(self.action_size)
        
        # Si aucune action valide, retourner un tableau vide
        if len(valid_actions) == 0:
            return valids
        
        # Get move rankings from lonelybot
        moves = ranked_moves_py(self.state, "neutral", self.heuristic_config)
        move_scores = {}
        
        # Convert moves to action indices and store their scores
        legal_moves = legal_actions_py(self.state)
        for move_info, move_str in zip(moves, legal_moves):
            try:
                # Get the action index for this move
                for action in valid_actions:
                    if str(action) in move_str:  # Simple string matching since we can't convert action to move string
                        move_scores[action] = move_info
                        break
            except:
                continue
        
        # Filter moves based on lonelybot analysis but keep at least one valid move
        best_move = None
        best_score = -float('inf')
        
        for action in valid_actions:
            if action in move_scores:
                move_info = move_scores[action]
                score = move_info["win_rate"]
                if len(move_info["revealed_cards"]) > 0:
                    score += 0.5  # Bonus pour les coups qui révèlent des cartes
                
                if score > best_score:
                    best_score = score
                    best_move = action
                
                # Accept move if it reveals cards or has decent win rate
                if len(move_info["revealed_cards"]) > 0 or move_info["win_rate"] >= 0.05:
                    valids[action] = 1
            else:
                # Allow moves not analyzed by lonelybot with a low priority
                if best_move is None:
                    best_move = action
                valids[action] = 1
        
        # Si aucun coup n'a été accepté, autoriser le meilleur coup disponible
        if np.sum(valids) == 0 and best_move is not None:
            valids[best_move] = 1
            log.warning(f"Forcing best available move {best_move} as no moves met criteria")
        
        return valids

    def getGameEnded(self, board, player):
        result = get_game_result(self.state)
        if result == 0:  # Game not ended
            # Analyze current state for potential deadlocks or progress
            unknown_cards, remaining_cards, blocked_cols, mobility, deadlock_risk = analyze_state_py(self.state)
            
            # Get ranked moves to evaluate position quality
            moves = ranked_moves_py(self.state, "neutral", self.heuristic_config)
            
            log.info(f"Game state analysis - Unknown cards: {unknown_cards}, Remaining: {remaining_cards}, "
                     f"Blocked cols: {blocked_cols}, Mobility: {mobility}, Deadlock risk: {deadlock_risk:.2f}")
            log.info(f"Available moves: {len(moves)}")
            
            # Détection des mouvements répétitifs
            if len(moves) == 1:
                current_move = str(moves[0])
                if len(self.last_moves) >= 2 and current_move == self.last_moves[-1] == self.last_moves[-2]:
                    self.repetition_count += 1
                    if self.repetition_count >= self.max_repetitions:
                        log.warning(f"Game ended due to move repetition: {current_move} repeated {self.repetition_count} times")
                        return -1
                else:
                    self.repetition_count = 0
                self.last_moves.append(current_move)
                if len(self.last_moves) > 10:  # Garder un historique limité
                    self.last_moves.pop(0)
            else:
                self.last_moves = []
                self.repetition_count = 0
            
            # Considérer la partie comme perdue uniquement si mobilité nulle ou critique
            if mobility == 0 or (mobility == 1 and deadlock_risk > 0.85):
                log.warning(f"Game ended due to no mobility or critical deadlock risk (mobility: {mobility}, risk: {deadlock_risk:.2f})")
                return -1
            
            # If no good moves available and high deadlock risk, consider game lost
            if len(moves) == 0 and deadlock_risk > 0.8:
                log.warning("Game ended due to no moves and high deadlock risk")
                return -1
            
            # If very low mobility and many blocked columns, consider game lost
            if mobility < 2 and blocked_cols > 4:
                log.warning("Game ended due to low mobility and blocked columns")
                return -1
                
            return 0
        
        log.info(f"Game ended with result: {result}")
        return result

    def getCanonicalForm(self, board, player):
        return get_canonical_board(board, player)

    def getSymmetries(self, board, pi):
        return [(board, pi)]  # no symmetries for Klondike

    def stringRepresentation(self, board):
        return board.tobytes()
