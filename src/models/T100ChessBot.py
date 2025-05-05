import torch
import torch.nn as nn
from torch.nn import functional as F
import config.T100ChessBot as CONFIG
import random
from utils.board import tokenize, compute_all_possible_moves
from pathlib import Path

class FeedForward(nn.Module):
    """
    A feed-forward neural network with GeLU activation.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG.EMB_SIZE, CONFIG.EMB_SIZE * 4),
            nn.GELU(),
            nn.Linear(CONFIG.EMB_SIZE * 4, CONFIG.EMB_SIZE),
        )

    def forward(self, x):
        return self.net(x)

    
class MultiHeadAttention(nn.Module):
    """
    A multi-head self-attention
    """
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(CONFIG.EMB_SIZE, CONFIG.EMB_SIZE * 3) # Compute Q, K, V in one go for efficiency
        self.ln = nn.Linear(CONFIG.HEAD_SIZE * CONFIG.NUM_HEADS, CONFIG.EMB_SIZE)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(CONFIG.EMB_SIZE, dim=2)             # [B, T, 3 * C] -> [B, T, C], [B, T, C], [B, T, C]
        q = q.view(B, T, CONFIG.NUM_HEADS, CONFIG.HEAD_SIZE).transpose(1, 2) # [B, T, H, C] -> [B, H, T, C]
        k = k.view(B, T, CONFIG.NUM_HEADS, CONFIG.HEAD_SIZE).transpose(1, 2) # [B, T, H, C] -> [B, H, T, C]
        v = v.view(B, T, CONFIG.NUM_HEADS, CONFIG.HEAD_SIZE).transpose(1, 2) # [B, T, H, C] -> [B, H, T, C]
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False) # If flash attention is available, use it for efficiency
        else:
            att = q @ k.transpose(-2, -1) * CONFIG.HEAD_SIZE ** -0.5 # Transpose becase k is [B, H, T, C] and q is [B, H, T, C]. We then get k [B, H, C, T]
            att = att.softmax(dim=-1)
            out = att @ v                                            # [B, H, T, C] @ [B, H, C, T] -> [B, H, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, CONFIG.NUM_HEADS * CONFIG.HEAD_SIZE)
        out = self.ln(out)
        return out
    
class Block(nn.Module):
    """
    A single transformer block. 
    It consists of a multi-head attention layer and a feed-forward layer, each followed by a residual connection and layer normalization.
    """
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(CONFIG.EMB_SIZE)
        self.ln2 = nn.LayerNorm(CONFIG.EMB_SIZE)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AttentionPooling(nn.Module):
    """
    Attention pooling layer that computes a weighted sum of the input sequences using attention scores.
    """
    def __init__(self, emb_size):
        super().__init__()
        self.attn = nn.Linear(emb_size, 1)

    def forward(self, x):
        attn_scores = self.attn(x) 
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled
    
class T100ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(CONFIG.VOCAB_SIZE, CONFIG.EMB_SIZE)
        self.position_embedding_table = nn.Embedding(CONFIG.SEQ_LENGTH, CONFIG.EMB_SIZE)
        self.blocks = nn.Sequential(*[Block() for _ in range(CONFIG.NUM_LAYERS)])
        self.ln = nn.LayerNorm(CONFIG.EMB_SIZE)
        self.attention_pooling = AttentionPooling(CONFIG.EMB_SIZE)
        self.ln_head = nn.Linear(CONFIG.EMB_SIZE, CONFIG.OUTPUT_DIM)

    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.token_embedding_table(x)
        positions = torch.arange(T, device=x.device)
        position_embeddings = self.position_embedding_table(positions) 
        position_embeddings = position_embeddings.unsqueeze(0)  
        out = token_embeddings + position_embeddings          

        out = self.blocks(out)                
        out = self.ln(out)                               
        pooled = self.attention_pooling(out)       
        logits = self.ln_head(pooled)           
        return logits


class T100ChessBot:
    def __init__(self):

        self.mta, self.atm = compute_all_possible_moves()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T100ChessModel().to(self.device)

        # Load the model parameters
        model_path = Path(__file__).resolve().parents[2]/'checkpoints/t100_best_model.pth'
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def make_move(self, board):
        """
        Make a move on the chess board using the model's predictions.
        The function first checks the legal moves available and then selects the best move based on the model's output.
        If no legal move is found among the top predictions, a random legal move is made.
        """
        # Get all legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves = set(legal_moves)

        board_status = tokenize(board.fen())
        board_vector = torch.from_numpy(board_status).long().unsqueeze(0).to(self.device)

        with torch.no_grad():
            with CONFIG.ctx:
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