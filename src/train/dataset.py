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
import torch
from collections import defaultdict

def get_batch(shards_dir, batch_size, device, split, split_ratio=0.9):
    """
    Get a batch of data from the dataset, using memory mapping to load the data efficiently.

    Args:
        shards_dir (str): Directory containing the dataset shards.
        batch_size (int): Size of the batch to load.
        device (torch.device): Device to load the data onto (CPU or GPU).
        split (str): Split type ('train' or 'val').
        split_ratio (float, optional): Ratio for splitting the data into training and validation sets. Defaults to 0.9.

    Returns:
        tuple: A tuple containing the input data (x) and labels (y) as PyTorch tensors.
    """
    
    x_array = np.memmap(os.path.join(shards_dir, 'x.bin'), dtype=np.uint8, mode='r').reshape(-1, 77)
    y_array = np.memmap(os.path.join(shards_dir, 'y.bin'), dtype=np.uint16, mode='r') 
    
    total = x_array.shape[0]
    # Compute split indices
    n_train = int(split_ratio * total)
    if split == 'train':
        start = 0
        end = n_train
    else:
        start = n_train
        end = total

    # Sample random indices in the split
    idx = np.random.randint(start, end, size=batch_size)
    x = torch.from_numpy((x_array[idx]).astype(np.int64))
    y = torch.from_numpy((y_array[idx]).astype(np.int64))

    # Move to device, pin memory if CUDA
    if device.type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
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

def unify_shards(npy_path):
    """
    Combine all numpy shards into a single memmap file.
    This function reads all numpy files in the specified directory and combines them into a single memmap file.
    """

    x_paths = sorted(glob.glob(os.path.join(npy_path, 'x_*.npy')))
    y_paths = sorted(glob.glob(os.path.join(npy_path, 'y_*.npy')))
    x_shapes = [np.load(fn, mmap_mode='r').shape for fn in x_paths]
    y_shapes = [np.load(fn, mmap_mode='r').shape for fn in y_paths]

    total_rows = sum(shape[0] for shape in x_shapes)
    D = x_shapes[0][1]  # assuming all have the same number of columns

    big_shard = np.memmap(
        '../data/npy_shards_v2/x.bin',
        mode='w+',
        dtype=np.uint8,
        shape=(total_rows, D)
    )

    start = 0
    BATCH_SIZE = 10000  # Adjust this based on your memory constraints
    for fn, shape in zip(x_paths, x_shapes):
        nrows = shape[0]
        shard = np.load(fn, mmap_mode='r').astype(np.uint8)
        for i in range(0, nrows, BATCH_SIZE):
            end = min(i + BATCH_SIZE, nrows)
            big_shard[start + i:start + end] = shard[i:end]
        start += nrows

    # Optional: flush changes to disk
    big_shard.flush()

    total_rows = sum(shape[0] for shape in y_shapes)

    big_shard = np.memmap(
        '../data/npy_shards_v2/y.bin',
        mode='w+',
        dtype=np.uint16,
        shape=(total_rows, )
    )

    start = 0
    BATCH_SIZE = 10000  # Adjust this based on your memory constraints
    for fn, shape in zip(y_paths, y_shapes):
        nrows = shape[0]
        shard = np.load(fn, mmap_mode='r').astype(np.uint16)
        for i in range(0, nrows, BATCH_SIZE):
            end = min(i + BATCH_SIZE, nrows)
            big_shard[start + i:start + end] = shard[i:end]
        start += nrows

    # Optional: flush changes to disk
    big_shard.flush()

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
                        x_arr = np.stack([tokens[i] for i in idx], dtype='uint8')
                        y_arr = np.array([labels[i] for i in idx], dtype='uint16')
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
        x_arr = np.stack([tokens[i] for i in idx], dtype='uint8')
        y_arr = np.array([labels[i] for i in idx], dtype='uint16')
        np.save(os.path.join(npy_path, f'x_{shard_idx:05d}.npy'), x_arr)
        np.save(os.path.join(npy_path, f'y_{shard_idx:05d}.npy'), y_arr)
        print(f"Final shard {shard_idx:05d} written ({len(x_arr)} positions).")

    print(f"All done. Saved {num_positions} positions from {games_processed} games in {shard_idx+1} shards to {npy_path}")


if __name__ == "__main__":
    lichess_pgn_path = Path(__file__).resolve().parents[2] / 'data/all_elite_2021_fen.pgn'
    npy_path = Path(__file__).resolve().parents[2] / 'data/npy_shards_v2'

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