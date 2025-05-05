import random
import torch
import torch.nn as nn
from utils.board import tokenize, compute_all_possible_moves
from pathlib import Path
import config.NeuralChessBot as CONFIG


class NeuralChessModel(nn.Module):
    def __init__(self, num_vocab=CONFIG.INPUT_SIZE, emb_size=CONFIG.EMB_SIZE, hidden_size=CONFIG.HIDDEN_SIZE, output_size=CONFIG.OUTPUT_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(num_vocab * emb_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        return self.mlp(x)
    
class NeuralChessBot:

    def __init__(self):

        self.mta, self.atm = compute_all_possible_moves()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu' # No need for GPU in this case
        self.model = NeuralChessModel(CONFIG.INPUT_SIZE, CONFIG.EMB_SIZE, CONFIG.HIDDEN_SIZE, CONFIG.OUTPUT_SIZE).to(self.device)

        # Load the model parameters
        model_path = Path(__file__).resolve().parents[2]/'checkpoints/mlp_best_model.pth'
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def make_move(self, board):
        # Get all legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves = set(legal_moves)

        board_status = tokenize(board.fen())
        board_vector = torch.from_numpy(board_status).long().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(board_vector)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_moves = torch.topk(probs.flatten(), 100).indices

        indices = top_moves.cpu().numpy()

        move_strs = [self.atm[idx] for idx in indices]
        for move_str in move_strs:
            if move_str in legal_moves:
                board.push(board.parse_uci(move_str))
                return board, 0
        
        # Fallback to random move if no top move is legal
        board.push(board.parse_uci(random.choice(list(legal_moves))))
        return board, 1