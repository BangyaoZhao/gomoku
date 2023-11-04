import numpy as np
from abc import ABC, abstractclassmethod
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from .basic_player import Player
from ..game import GomokuGame
from ..helper_funcs import get_normal_moves, get_valid_moves

class Node:
    def __init__(self, parent, p, sign=-1):
        self.parent = parent
        self.children = {} 
        self.sign = sign
        self.n_visits = 0
        self.Q = 0.5
        self.u = 0
        self.p = p
    
    def __getitem__(self, key):
        return self.children[key]

    def expand(self, action_p_dict):
        for action, p in action_p_dict.items():
            self.children[action] = Node(self, p, -self.sign)

    def select(self, mode = 'value'):
        return max(self.children.items(), key = lambda x: x[1].__getattribute__(mode))

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if not self.parent is None:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)
        
    @property
    def value(self):
        return -self.sign+self.sign*self.action_value+self.bonus
    
    @property
    def action_value(self):
        n0 = 100; factor = 100
        a = factor*self.Q*self.n_visits+n0
        b = factor*(1-self.Q)*self.n_visits+n0
        return np.random.beta(a, b)
    
    @property
    def bonus(self):
        return (self.p/(1+self.n_visits))**(1/3)
    
    # @property
    # def action_value(self):
    #     return self.Q
    
    # @property
    # def bonus(self):
    #     return self.p/(1+self.n_visits)

class MCTS(ABC):
    def __init__(self, game: GomokuGame, side, max_depth):
        self.root = Node(None, 0.5)
        self.game, self.side = game.copy(), side
        self.max_depth = max_depth

    @abstractclassmethod
    def get_expand_dict(self, game: GomokuGame, side):
        pass
    
    @abstractclassmethod
    # assuming game not end
    def evaluate_node(self, game: GomokuGame, side):
        pass

    def one_playout(self):
        node = self.root
        game = self.game.copy()
        if not game.winner is None:
            raise 'Game is already over'
        # MTS
        depth = 0
        side = self.side
        
        while True:
            depth += 1
            if len(node.children) == 0:
                action_p_dict = self.get_expand_dict(game, side)
                node.expand(action_p_dict)
            move, node = node.select()
            game.play(move, side)
            if not game.winner is None:
                if game.winner == 9:
                    node.update_recursive(0.5)
                else:
                    leaf_value = float(game.winner == self.side)
                    node.update_recursive(leaf_value)
                break
            elif depth == self.max_depth:
                leaf_value = self.evaluate_node(game, side)
                node.update_recursive(leaf_value)
                break
            side = 1 - side

    def playout(self, N):
        for _ in range(N):
            self.one_playout()
    
    def visualize(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2)
            show = True
        else:
            show = False
        for ax in axes:
            self.game.plot_game(None, ax)

        ax = axes[0]
        ax.set_title('Prior Probs')
        for (j, i), node in self.root.children.items():
            p = node.p
            circle = Circle((i, j), 0.45*np.sqrt(p), facecolor='black' if self.side==0 else 'white')
            ax.add_patch(circle)
            if p>0.01:
                ax.text(i, j, f'{p:.1%}', ha='center', va='center', color='blue')
        
        ax = axes[1]
        ax.set_title(f'MCTS visits: {self.root.n_visits} WinRate: {self.root.Q :.1%}')
        for (j, i), node in self.root.children.items():
            p = node.n_visits/self.root.n_visits
            circle = Circle((i, j), 0.45*np.sqrt(p), facecolor='black' if self.side==0 else 'white')
            ax.add_patch(circle)
            if p>0.01:
                ax.text(i, j-0.3, f'{p:.1%}', ha='center', va='center', color='blue')
                ax.text(i, j+0.3, f'{node.Q:.1%}', ha='center', va='center', color='red')
        
        if show:
            fig.show()
    
    @property
    def probmat(self):
        mat= np.zeros((15, 15))
        for move, node in self.root.children.items():
            mat[move] = node.n_visits
        mat /= self.root.n_visits
        return mat


# customized MCTS
class NaiveMCTS(MCTS):
    def get_expand_dict(self, game: GomokuGame, side):
        moves = get_normal_moves(game.board, self.dist)
        random.shuffle(moves)
        n = len(moves)
        action_p_dict = {move: 1/n for move in moves}
        return action_p_dict
    
    def evaluate_node(self, game: GomokuGame, side):
        return 0.5+np.random.normal(scale=self.noise_level)

class NaiveMCTSPlayer(Player):
    def __init__(self, max_depth, n_playouts, noise_level=0.01, dist=1):
        super().__init__()
        self.max_depth, self.n_playouts, self.noise_level, self.dist = max_depth, n_playouts, noise_level, dist
    
    def play(self, game: GomokuGame, side):
        super().play(game, side)
        tree = NaiveMCTS(game, side, self.max_depth)
        tree.noise_level, tree.dist = self.noise_level, self.dist
        tree.playout(self.n_playouts)
        move, _ = tree.root.select(mode='n_visits')
        game.play(move, side)
        
        
        
    
    