import chess
import chess.engine
from IPython.display import display
import chess
from chess.pgn import Game
from importlib import import_module
import random

LICHESS_LEVELS = {
    1: {"SKILL": -9, "DEPTH": 5, "TIME_CONSTRAINT": 0.050},
    2: {"SKILL": -5, "DEPTH": 5, "TIME_CONSTRAINT": 0.100},
    3: {"SKILL": -1, "DEPTH": 5, "TIME_CONSTRAINT": 0.150},
    4: {"SKILL": 3, "DEPTH": 5, "TIME_CONSTRAINT": 0.200},
    5: {"SKILL": 7, "DEPTH": 5, "TIME_CONSTRAINT": 0.300},
    6: {"SKILL": 11, "DEPTH": 8, "TIME_CONSTRAINT": 0.400},
    7: {"SKILL": 16, "DEPTH": 13, "TIME_CONSTRAINT": 0.500},
    8: {"SKILL": 20, "DEPTH": 22, "TIME_CONSTRAINT": 1.000},
}

def get_game_pgn(board, w_name, b_name, result=None):
    """
    Create a PGN game object from the current board state.
    This function sets up the game with the provided player names and result.

    Args:
        board (chess.Board): The current board state.
        w_name (str): Name of the white player.
        b_name (str): Name of the black player.
        result (str, optional): Result of the game. Defaults to None.

    Returns:
        chess.pgn.Game: A PGN game object representing the game.
    """
    # Create a PGN game object
    game = Game()
    game.setup(board.starting_fen)
    game.headers["Event"] = f"{w_name} vs. {b_name}"
    game.headers["White"] = w_name
    game.headers["Black"] = b_name
    game.headers["Result"] = board.result() if not result else result
    node = game

    # Add all moves to the PGN game
    for move in board.move_stack:
        node = node.add_variation(move)
    
    return game


def load_engine(path):
    """
    Load a chess engine.

    Args:
        path (str): The path to the engine file (an executable).

    Returns:
        chess.engine.SimpleEngine: The chess engine.
    """
    engine = chess.engine.SimpleEngine.popen_uci(path)
    return engine

def engine_move(board, engine, engine_config):
    """
    Make a move using the chess engine.

    Args:
        board (chess.Board): The current board state.
        engine (chess.engine.SimpleEngine): The chess engine instance.
        engine_level (int): The skill level of the engine.
    
    Returns:
        chess.Board: The updated board state after the engine's move.
    """
    if engine_config:
        depth = engine_config.get("DEPTH")
        time_constraint = engine_config.get("TIME_CONSTRAINT")
        limit = chess.engine.Limit(time=time_constraint, depth=depth)
        result = engine.play(board, limit=limit)
        board.push(result.move)
        return board
    else:
        raise ValueError("Engine level not specified.")

def load_model(model_name):
    """
    Load a chess model from the specified module.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        object: An instance of the model class.

    Raises:
        ImportError: If the model module or class cannot be found.
    """
    try:
        module = import_module(
            "models.{}".format(model_name)
        )
        model_class = getattr(module, model_name)
        return model_class()
    except ModuleNotFoundError:
        raise ImportError(f"Module {model_name} not found. Please check the model name.")
    except AttributeError:
        raise ImportError(f"Module {model_name} does not have a class named {model_name}.")

def model_move(model, board):
    """
    Make a move using the chess model.

    Args:
        model (object): The chess model instance.
        board (chess.Board): The current board state.

    Returns:
        chess.Board: The updated board state after the model's move.
    """
    board = model.make_move(board) 
    return board

def human_move(board, move):
    """
    Handle human move input.

    Args:
        board (chess.Board): The current board state.
        move (str): The move input by the user. Can be in UCI format (e.g., e2e4) or 'r' for resign, 'd' for draw.
    
    Returns:
        chess.Board: The updated board state after the move.
    """
    try:
        player_move = board.parse_uci(move)
        board.push(player_move)
        if board.is_checkmate():
            display("Checkmate!")
        return board
    except ValueError as e:
        display(f"Error parsing move: {e}")
        return None

def model_vs_machine(stockfish_path, engine_config, white_player_name, black_player_name, model_color, model, rounds=1):
    """
    Function to handle model vs machine chess game. This function will play a game between the model and the engine, alternating moves.
    The game will be played for a specified number of rounds, and the results will be recorded.

    Args:
        stockfish_path (str): Path to the Stockfish engine executable.
        engine_config (dict): Configuration for the engine (e.g., skill level).
        white_player_name (str): Name of the white player.
        black_player_name (str): Name of the black player.
        model_color (str): Color played by the model ('w' or 'b').
        model: The chess model instance.
        rounds (int): Number of rounds to play.
    
    Returns:
        tuple: A tuple containing the number of wins, losses, draws, and the PGNs of the games played.
    """
    random.seed(42)  # For reproducibility
    engine = load_engine(stockfish_path)
    engine.configure({'Skill Level': engine_config['SKILL'], 'Threads': 4, 'Hash': 4000})

    try:
        # Model color
        model_color = model_color.lower()
        assert model_color in ["w", "b"], "Color played by model must be 'w' or 'b'!"

        # Play
        pgns = list()
        wins, draws, losses = 0, 0, 0

        for _ in range(rounds):
            board = chess.Board()
            
            if model_color == "w":
                try:
                    board = model_move(model, board)
                except Exception as e:
                    print(f"Error in model move: {e}")
                    continue

            while not board.is_game_over():
                try:
                    # Engine move
                    board = engine_move(board, engine, engine_config=engine_config)
                except Exception as e:
                    print(f"Error in engine move: {e}")
                
                if not board.is_game_over():
                    try:
                        board = model_move(model, board)
                    except Exception as e:
                        print(f"Error in model move: {e}")
                        continue

            # Record the result
            result = board.result()
            game = get_game_pgn(board=board, w_name=white_player_name, b_name=black_player_name, result=result)
            pgns.append(game)

            wins += (int(result == '1-0') if model_color == "w" else int(result == '0-1'))
            losses += (int(result == '0-1') if model_color == "w" else int(result == '1-0'))
            draws += int(result == '1/2-1/2')
        return wins, losses, draws, pgns

    finally:
        engine.close()
