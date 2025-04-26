import random

class RandomChessBot:

    def __init__(self):
        random.seed(42)

    def make_move(self, board):
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        # Select a random move
        random_move = random.choice(legal_moves)
        board.push(random_move)
        return board, 0