from utils.utils import load_model, model_vs_machine, LICHESS_LEVELS
from pathlib import Path
import yaml
import argparse
import random
import math

def evaluate_model(levels, rounds, model):
    """
    Evaluate the chess model against Stockfish at different levels.
    This function will play games against the engine at various levels and record the results.
    Stats based on https://www.chessprogramming.org/Match_Statistics

    Args:
        levels (list): List of Stockfish levels to evaluate against.
        rounds (int): Number of rounds to play per level and color.
        model: The chess model instance.

    Returns:
        dict: A dictionary containing the results of the evaluation.
    """
    results = {}
    # Evaluate
    for LL in levels:
        level_results = {"wins": 0, "losses": 0, "draws": 0, "pgns": []}
        print(f"Evaluating against Stockfish level {LL}...")
        for model_color in ["w", "b"]:
            print(f"Model playing as {model_color.upper()}...")
            # Play
            w, l, d, pgns = model_vs_machine(
                stockfish_path=config['stockfish_path'],
                engine_config=LICHESS_LEVELS[LL],
                white_player_name="White Player",
                black_player_name="Black Player",
                model_color=model_color,
                model=model,
                rounds=rounds)
            level_results["wins"] += w
            level_results["losses"] += l
            level_results["draws"] += d
            level_results["pgns"].extend(pgns)
    
        results[f"level_{LL}"] = level_results
    return results


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    parser = argparse.ArgumentParser(description="Evaluate a chess model against Stockfish.")
    parser.add_argument("--levels", type=int, nargs='+', default=[1, 2],
                        help="Stockfish levels to evaluate against (e.g., 1 2 3).")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of rounds to play per level and color.")
    parser.add_argument("--model_name", type=str, default='RandomChessBot', help="Name of model to use.")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[2]/'config.yaml'

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")
    
    model = load_model(args.model_name)
    results = evaluate_model(
                levels=args.levels,
                rounds=args.rounds,
                model=model
                )

    for lvl, res in results.items():
        wins = res['wins']
        losses = res['losses']
        draws = res['draws']
        total_games = wins + losses + draws

        print(f"Results for {lvl}:")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Win ratio: {(wins + draws/2) / (total_games) * 100:.2f}%")
        print(f"Draw ratio: {draws / total_games * 100:.2f}%")
        print(f"Likelihood of superiority: {round(0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * (wins + losses)))), 2) * 100:.2f}%")
        print()
    print("Evaluation completed.")