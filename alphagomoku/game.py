import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

def check_one_dir(x: np.ndarray):
    return np.any(np.convolve(np.ones(5), x) == 5)

def check_win(board, loc):
    i, j = loc
    if check_one_dir(board[i]):
        return True
    elif check_one_dir(board[:, j]):
        return True
    elif check_one_dir(board.diagonal(j-i)):
        return True
    return check_one_dir(board[:, ::-1].diagonal(14-j-i))

class GomokuGame:
    def __init__(self) -> None:
        self.board = np.zeros((15, 15))
        self.hist = []
        self.winner = None
        pass
    
    def copy(self, n=None):
        if n is None:
            n = len(self.hist)
        game = GomokuGame()
        for i, j, side in self.hist[:n]:
            game.play((i, j), side)
        return game
        
    def play(self, loc, side) -> None:
        if not self.board[loc] == 0:
            raise 'Location already taken.'
        if not self.winner is None:
            raise 'Game already finished.'
        # place the move on board
        self.hist.append(loc + (side,))
        self.board[loc] = 1-2*side
        # check winner
        if check_win((1-2*side)*self.board, loc):
            self.winner = side
        if len(self.hist) == 15*15 and self.winner is None:
            self.winner = 9
    
    @property
    def board_hist(self):
        board_hist = [np.zeros((15, 15))]
        for i, j, side in self.hist:
            board_hist.append(board_hist[-1].copy())
            board_hist[-1][i, j] = 1-2*side
        return board_hist
    
    def plot_game(self, i, ax):
        if i is None:
            i = len(self.hist)
        # define color of stones and the board
        colors = ["white", "#DEB887", "black"]
        cmap = ListedColormap(colors)
        # clear the plot
        ax.cla()
        board = self.board_hist[i]
        ax.imshow(board, cmap=cmap, vmin=-1, vmax=1)
        # minor changes for the board
        for i in range(15):
            ax.axhline(i - 0.5, color='grey', linewidth=1.5)
            ax.axvline(i - 0.5, color='grey', linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def visualize(self):
        fig, ax = plt.subplots()
        i = 0
        def click_action(di, event):
            nonlocal i
            i += di
            i %= (len(self.hist)+1)
            self.plot_game(i, ax)
            ax.set_title(f'Move {i}')
            fig.canvas.draw()
        
        click_action(0, None)
        axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(lambda event: click_action(1, event))
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(lambda event: click_action(-1, event))
        
        plt.show()


        
            