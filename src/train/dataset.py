from torch.utils.data import Dataset
from utils.board import tokenize, compute_all_possible_moves
import json
import numpy as np
import argparse
from pathlib import Path
import hashlib
import re
import glob
import os


class ChessNpyDataset(Dataset):
    def __init__(self, shards_dir):
        self.x_paths = sorted(glob.glob(os.path.join(shards_dir, 'x_*.npy')))
        self.y_paths = sorted(glob.glob(os.path.join(shards_dir, 'y_*.npy')))
        assert len(self.x_paths) == len(self.y_paths)
        self.shard_lengths = []
        self.cum_lengths = []
        total = 0
        for x_path in self.x_paths:
            length = np.load(x_path, mmap_mode='r+').shape[0]
            self.shard_lengths.append(length)
            total += length
            self.cum_lengths.append(total)
        self.total = total

        # Defer actual array opening to worker init
        self._x_arrays = None
        self._y_arrays = None

    def _ensure_arrays(self):
        # Lazily open mmap arrays per worker
        if self._x_arrays is None:
            self._x_arrays = [np.load(p, mmap_mode='r+') for p in self.x_paths]
            self._y_arrays = [np.load(p, mmap_mode='r+') for p in self.y_paths]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        self._ensure_arrays()
        shard_idx = np.searchsorted(self.cum_lengths, idx, side='right')
        if shard_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_lengths[shard_idx-1]
        x = self._x_arrays[shard_idx][local_idx]
        y = self._y_arrays[shard_idx][local_idx]
        return x, y

    
def parse_games_with_fen(pgn_path):
    """
    Parse a PGN file and yield game lines with FEN strings.
    This function reads the PGN file line by line and yields a list of lines for each game.

    Args:
        pgn_path (str): Path to the PGN file.
    
    Yields:
        list: A list of lines representing a game.
    """
    with open(pgn_path, 'r', encoding='utf-8') as f:
        buffer = []
        for line in f:
            if line.strip().startswith('[Event') and buffer:  # game separator
                yield buffer
                buffer = []
                buffer.append(line)
            else:
                buffer.append(line)
        if buffer:
            yield buffer

def fen_label_hash(tok, label):
    """
    Hash the tokenized board and move label to a unique identifier.
    This function uses the blake2b hash function to create a unique hash for the given token and label.

    Args:
        tok (list): Tokenized representation of the board.
        label (int): Move label.

    Returns:
        bytes: A unique hash for the token and label.
    """
    # Concatenate bytes of tok and lab
    b = bytes(tok) + label.to_bytes(4, byteorder='little')
    h = hashlib.blake2b(b, digest_size=8).digest()
    return h

def generate_dataset(pgn_path, npy_path, max_games=None, chunk_size=1000):
    """
    Generate a dataset from a PGN file and save it as numpy arrays.
    This function reads the PGN file, tokenizes the board positions, and saves the data in numpy shards after chunk_size (and shuffles the data).
    The dataset is made up of pairs of (tokenized board, UCI move that follows).

    Args:
        pgn_path (str): Path to the PGN file.
        npy_path (str): Path to save the numpy arrays.
        max_games (int, optional): Maximum number of games to process. Defaults to None.
        chunk_size (int, optional): Size of each chunk to save. Defaults to 1000.
    """
    os.makedirs(npy_path, exist_ok=True)
    seen = set()
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    tokens = []
    labels = []
    num_positions = 0
    games_processed = 0
    shard_idx = 0

    total_games = 1293126
    # for i, game_lines in enumerate(parse_games_with_fen(pgn_path)):
    #     total_games += 1

    # Store ECO codes seen in the dataset
    eco_seen = {}
    print(f"Total games in PGN: {total_games}")
    for i, game_lines in enumerate(parse_games_with_fen(pgn_path)):
        if i == max_games and max_games is not None:
            break
        for line in game_lines:
            if line.startswith('[ECO'):
                eco = line.split('"')[1].strip()
                if eco in eco_seen:
                    eco_seen[eco] += 1
                else:
                    eco_seen[eco] = 1
                continue
            if line.startswith('[') or line.strip() == '':
                continue
            else:
                fen_matches = re.findall(r'\{\s([^}]*)\}', line.strip())
                uci_text = re.sub(r'\{[^}]*\}', '', line.strip())
                uci_matches = re.findall(r'([a-h][1-8][a-h][1-8][qrbnQBRN]?)', uci_text)
                fen_list = [i.strip() for i in fen_matches[:-1]]
                uci_list = [i.strip().lower() for i in uci_matches]
                fen_list.insert(0, starting_fen)
                for j in range(len(uci_list)):
                    try:
                        tok = tokenize(fen_list[j])
                        lab = mta[uci_list[j]]
                        hval = fen_label_hash(tok, lab)
                        # Check if the hash is already seen so we don't have duplicates
                        if hval not in seen:
                            seen.add(hval)
                            tokens.append(tok)       
                            labels.append(lab)  
                            num_positions += 1
                    except Exception:
                        continue

                    # Save in chunks
                    if len(tokens) >= chunk_size:
                        idx = np.random.permutation(len(tokens)) # shuffle the data
                        x_arr = np.stack([tokens[i] for i in idx])
                        y_arr = np.array([labels[i] for i in idx], dtype='int32')
                        np.save(npy_path / f'x_{shard_idx:05d}.npy', x_arr)
                        np.save(npy_path / f'y_{shard_idx:05d}.npy', y_arr)
                        tokens.clear()
                        labels.clear()
                        shard_idx += 1

        games_processed += 1
        if games_processed % 50000 == 0:
            print(f"Processed {games_processed} games {round(games_processed*100/total_games, 2)}%, {num_positions} positions so far.")
            with open(os.path.join(npy_path.resolve().parent, 'eco_seen.json'), 'w') as f:
                json.dump(eco_seen, f, indent=4)

    # Write any leftovers
    if tokens:
        idx = np.random.permutation(len(tokens))
        x_arr = np.stack([tokens[i] for i in idx])
        y_arr = np.array([labels[i] for i in idx], dtype='int32')
        np.save(os.path.join(npy_path, f'x_{shard_idx:05d}.npy'), x_arr)
        np.save(os.path.join(npy_path, f'y_{shard_idx:05d}.npy'), y_arr)
        print(f"Final shard {shard_idx:05d} written ({len(x_arr)} positions).")

    print(f"All done. Saved {num_positions} positions from {games_processed} games in {shard_idx+1} shards to {npy_path}")


if __name__ == "__main__":
    lichess_pgn_path = Path(__file__).resolve().parents[2] / 'data/all_elite_2021_fen.pgn'
    npy_path = Path(__file__).resolve().parents[2] / 'data/npy_shards'

    parser = argparse.ArgumentParser(description="Convert PGN to HDF5")
    parser.add_argument("pgn_path", type=str, nargs='?', default=lichess_pgn_path, help="Path to the PGN file")
    parser.add_argument("npy_path", type=str, nargs='?', default=npy_path, help="Path to the output HDF5 file")
    parser.add_argument("--max_games", type=str, nargs='?', default=50000, help="Maximum number of games to process")
    parser.add_argument("--chunk_size", type=int, nargs='?', default=15_000_000, help="Chunk size for writing to HDF5")

    args = parser.parse_args()

    if isinstance(args.max_games, str) and args.max_games.lower() == 'none':
        args.max_games = None
    else:
        args.max_games = int(args.max_games)

    mta, atm = compute_all_possible_moves()

    generate_dataset(args.pgn_path, args.npy_path, args.max_games, args.chunk_size)