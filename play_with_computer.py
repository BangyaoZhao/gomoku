import pygame, torch
import numpy as np

from alphagomoku.game import GomokuGame
from alphagomoku.players.basic_player import NaivePlayer
from alphagomoku.players.mcts_player import MCTS, NaiveMCTSPlayer, NaiveMCTS, get_valid_moves, Player
from alphagomoku.players.mcts_policy_player import PolicyMCTSPlayer, PolicyNet
#####################
pygame.init()
computer_side = 0

width, height = 800, 800
font = pygame.font.SysFont('monospace', 20)

# constant vals
xy_min = 100; xy_max = 700
l = (xy_max-xy_min)/15

# define computer player
model = PolicyNet()
model.load_state_dict(torch.load('server/models/model11.pth', map_location=torch.device('cpu')))
model.eval()
player = PolicyMCTSPlayer(20, 1000, model, True)

# draw the board
def draw_board(window):
    for x in np.linspace(xy_min, xy_max, 16):
        pygame.draw.line(window, '#000000', (x, xy_min), (x, xy_max), width = 2)
    for y in np.linspace(xy_min, xy_max, 16):
        pygame.draw.line(window, '#000000', (xy_min, y), (xy_max, y), width = 2)

# draw a move
def draw_move(window, move, side):
    color = 'black' if side==0 else 'white'
    pygame.draw.circle(window, color, (xy_min+(move[0]+0.5)*l+1, xy_min+(move[1]+0.5)*l+1), 0.5*l-2)

# get row/column number with mouse click
def get_index(x):
    locs = np.linspace(xy_min+0.5*l, xy_max-0.5*l, 15)
    distance = np.abs(locs-x)
    return np.argmin(distance)

def main():
    window = pygame.display.set_mode((width, height))
    window.fill('#996600')
    game = GomokuGame()
    pygame.display.set_caption('Gomoku')
    clock = pygame.time.Clock()
    
    draw_board(window)
    
    run = True
    side = 0
    while run:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif not game.winner is None:
                pass
            elif computer_side==side:
                player.play(game, side)
                move = game.hist[-1][:2]
                draw_move(window, move, side)
                # draw win rate
                window.fill('#996600', rect = (0, 0, width, xy_min))
                text_surface = font.render(f'Computer: {player.trees[-1].root.Q :.1%}', 0, (0, 0, 0))
                window.blit(text_surface, (xy_min, xy_min/2))
                side = 1-side
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if xy_min<x<xy_max and xy_min<y<xy_max:
                    move = (get_index(x), get_index(y))
                    game.play(move, side)
                    draw_move(window, move, side)
                    side = 1-side
                pygame.display.update()
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    main()

