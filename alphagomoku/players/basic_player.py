import numpy as np
from abc import ABC, abstractclassmethod
from ..game import GomokuGame
import random, itertools

class Player(ABC):
    def __init__(self):
        pass
    
    @abstractclassmethod
    def play(self, game: GomokuGame, side):
        if not game.winner is None:
            raise 'Game already over'

class NaivePlayer(Player):
    def play(self, game: GomokuGame, side):
        super().play(game, side)
        moves = [move for move in itertools.product(range(15), range(15)) if game.board[move] == 0]
        move = random.choice(moves)
        game.play(move, side)