from typing import Any
from alphagomoku.game import GomokuGame
from alphagomoku.players.mcts_player import MCTS, get_valid_moves, Player
import torch.nn as nn
import torch
import numpy as np
from scipy.special import softmax
import itertools, random

class PolicyNet(nn.Module):
    """policy-value network module"""
    def __init__(self):
        super().__init__()
        
        self.fun = nn.Sequential(
            nn.Unflatten(1, (1, 15)),
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.LogSoftmax(1),
            nn.Unflatten(1, (15, 15))
        )

    def forward(self, state_input):
        return self.fun(state_input)


# customized MCTS
class PolicyMCTS(MCTS):
    def use(self, model):
        self.model = model
    
    def get_expand_dict(self, game: GomokuGame, side):
        board_ts = torch.tensor((1-2*side)*game.board, dtype=torch.float32)
        board_ts = torch.unsqueeze(board_ts, 0)
        logodds = self.model(board_ts).detach().numpy()[0]
        moves = get_valid_moves(game.board)
        random.shuffle(moves)
        logodds = [logodds[move] for move in moves]
        action_p_dict = {move: p for (move, p) in zip(moves, softmax(logodds))}
        return action_p_dict
    
    def evaluate_node(self, game: GomokuGame, side):
        return 0.5
    
    def __getitem__(self, index):
        game = self.game.copy()
        game.play(index, self.side)
        tree = PolicyMCTS(game, 1-self.side, self.max_depth)
        tree.root = self.root.children[index]
        return tree

class PolicyMCTSPlayer(Player):
    def __init__(self, max_depth, n_playouts, model, save_tree=False, save_xy=True):
        super().__init__()
        self.max_depth, self.n_playouts = max_depth, n_playouts
        self.model = model
        self.save_tree, self.save_xy = save_tree, save_xy
        if self.save_tree:
            self.trees = []
        if self.save_xy:
            self.xys = []
    
    def play(self, game: GomokuGame, side):
        super().play(game, side)
        tree = PolicyMCTS(game, side, self.max_depth)
        tree.use(self.model)
        tree.playout(self.n_playouts)
        move, _ = tree.root.select(mode='n_visits')
        game.play(move, side)
        if self.save_tree:
            self.trees.append(tree)
        self.xys.append(np.stack([tree.game.board*(1-2*tree.side), tree.probmat]))