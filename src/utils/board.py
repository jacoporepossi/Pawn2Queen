import numpy as np
import chess

# This source of this code is https://github.com/google-deepmind/searchless_chess/blob/main/src/utils.py


# The lists of the strings of the row and columns of a chess board,
# traditionally named rank and file.
_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

_CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'p', 'n', 'r', 'k', 'q',
               'P', 'B', 'N', 'R', 'Q', 'K', 'w', '.'
               ]

_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77

def tokenize(fen: str):
  """Returns an array of tokens from a fen string.

  Compute a tokenized representation of the board, from the FEN string.
  The final array of tokens is a mapping from this string to numbers, which
  are defined in the dictionary `_CHARACTERS_INDEX`.
  For the 'en passant' information, we convert the '-' (which means there is
  no en passant relevant square) to '..', to always have two characters, and
  a fixed length output.

  Args:
    fen: The board position in Forsyth-Edwards Notation.
  """
  # Extracting the relevant information from the FEN.
  board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  board = board.replace('/', '')
  board = side + board

  indices = list()

  for char in board:
    if char in _SPACES_CHARACTERS:
      indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
    else:
      indices.append(_CHARACTERS_INDEX[char])

  if castling == '-':
    indices.extend(4 * [_CHARACTERS_INDEX['.']])
  else:
    for char in castling:
      indices.append(_CHARACTERS_INDEX[char])
    # Padding castling to have exactly 4 characters.
    if len(castling) < 4:
      indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  if en_passant == '-':
    indices.extend(2 * [_CHARACTERS_INDEX['.']])
  else:
    # En passant is a square like 'e3'.
    for char in en_passant:
      indices.append(_CHARACTERS_INDEX[char])

  # Three digits for halfmoves (since last capture) is enough since the game
  # ends at 50.
  halfmoves_last += '.' * (3 - len(halfmoves_last))
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  # Three digits for full moves is enough (no game lasts longer than 999
  # moves).
  fullmoves += '.' * (3 - len(fullmoves))
  indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

  assert len(indices) == SEQUENCE_LENGTH

  return np.asarray(indices, dtype=np.uint8)

def detokenize(tokens):
    """Inverse of tokenize: reconstructs FEN from token array."""
    # Build inverse mapping
    index_to_char = {v: k for k, v in _CHARACTERS_INDEX.items()}
    # Ensure tokens is a list or 1D array of ints
    tokens = list(tokens)
    chars = [index_to_char[int(t)] for t in tokens]

    # Side to move
    side = chars[0]

    # Board (64 squares)
    board_chars = chars[1:65]
    fen_board = ''
    empty_count = 0
    for i, c in enumerate(board_chars):
        if c == '.':
            empty_count += 1
        else:
            if empty_count > 0:
                fen_board += str(empty_count)
                empty_count = 0
            fen_board += c
        if (i + 1) % 8 == 0:
            if empty_count > 0:
                fen_board += str(empty_count)
                empty_count = 0
            if i != 63:
                fen_board += '/'

    # Castling rights
    castling = ''.join(chars[65:69]).replace('.', '')
    if not castling:
        castling = '-'

    # En passant
    en_passant = ''.join(chars[69:71])
    if en_passant == '..':
        en_passant = '-'

    # Halfmove clock
    halfmove = ''.join(chars[71:74]).replace('.', '')
    if not halfmove:
        halfmove = '0'

    # Fullmove number
    fullmove = ''.join(chars[74:77]).replace('.', '')
    if not fullmove:
        fullmove = '1'

    # Compose FEN
    fen = f"{fen_board} {side} {castling} {en_passant} {halfmove} {fullmove}"
    return fen

def compute_all_possible_moves():
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol('Q'))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol('N'))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [('2', '1'), ('7', '8')]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f'{file}{rank}{file}{next_rank}'
      promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

      # Capture promotions.
      # Left side.
      if file > 'a':
        next_file = _CHESS_FILE[index_file - 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
      # Right side.
      if file < 'h':
        next_file = _CHESS_FILE[index_file + 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move