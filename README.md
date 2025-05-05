<p align="center">
  <img width="200" src="img/logo.png"/>
</p>

<h1 align="center"><i>Pawn2Queen</i></h1>
<p align="center"><i>Transformer-powered Chess AI</i></p>
<br>

*Pawn2Queen* is a personal project that combines my passion for chess with Machine Learning. It leverages Transformer architectures, Reinforcement Learning and Knowledge Distillation to develop a chess AI.\
The name reflects the journey of a humble pawn becoming a queen in chess. It also symbolizes my learning process to master state-of-the-art AI techniques.

## 🚀 Project goals
The project is focused on creating a chess AI using modern techniques while serving as a space for personal growth in fields such as:
- **Transformer architectures**: GPT-style models for chess positions or move generation
- **Reinforcement Learning**: Training an agent to play chess through self-play and fine-tuning with RLHF
- **Knowledge Distillation**: Compressing large, complex models into smaller, efficient ones

Although primarily an educational project, my goal is to eventually **deploy the bot on Lichess**, where it can play against real opponents.\
By doing so, I hope to gain insights into the strengths and weaknesses of the model, which can be used to further improve it through additional training and fine-tuning (and learning for me!).


## 🚧 Project status

This project is a **work in progress**.
For a detailed history of changes, refer to the [CHANGELOG.md](CHANGELOG.md) file.

## 🛠️ Installation

This project uses Python and is managed with `pyproject.toml`. To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jacoporepossi/Pawn2Queen.git
   cd Pawn2Queen
   ```

2. Install dependencies:
    ```bash
    pip install -e .
    ```

3. Set up the configuration file to use Stockfish as the chess engine:
    - Download the Stockfish binary from the [official website](https://stockfishchess.org/download/).
    - Rename `config.template.yaml` to `config.yaml`
    - Update the `config.yaml` file in the root directory with the path to the Stockfish binary. For example:
    ```yaml
    stockfish_path: "C:/path/to/stockfish.exe"
    ```


## 📂 Project structure

```bash
Pawn2Queen/
├── README.md
├── pyproject.toml
├── config.yaml            # Configurations
├── src/                   # Source code
│   ├── config/            # Configuration files
│   ├── evaluate/          # Evaluation scripts
│   ├── models/            # Model architectures and implementations
│   ├── train/             # Training scripts
│   └── utils/             # Utility functions and helpers
├── data/                  # Data files and resources
└── notebooks/             # Jupyter notebooks for human play, experimentation and analysis
```

## 📊 Dataset

The project uses the [Lichess Elite Database](https://database.nikonoel.fr/) for training and the [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/help.html) tool to extract the data. \
Current training data includes games played in **2021 ended in a checkmate**, filtered and combined into a single PGN file.
After downloading the pgn-extract executable, the command I used to combine the data is as follows:

```bash
pgn-extract -f 2021_elite_games_files.txt -M --quiet -o all_elite_2021.pgn
```
where `2021_elite_games_files.txt` is a txt file with the paths to the PGN files of the games in the folder, `-M` option is used to extract only the games that ended in a checkmate and `-o` option is used to specify the output file name.

The extracted data is further processed to create a dataset of chess positions and moves to train the AI. The data is extracted in a format suitable for training, where each position is represented by a FEN string and each move is represented in UCI format. The data is structured as follows:

- `position`: the FEN string representing the position of the chessboard before the move (starting from the initial position)
- `move`: the move made by the player in UCI format (e.g., e2e4, g1f3)

For instance, given the following PGN:
```
1. d4 Nf6
```
the extracted data will look like this:
```
Board (initial) --> "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
Move made       --> "d2d4"                                                       

Board           --> "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"  
Move made       --> "g8f6"                                                           
```

This is only the initial step, as the strings will be converted into a tensor format suitable for training the model thanks to the `src/train/dataset.py` script.
To prepare the data, I once again used the `pgn-extract` tool to process the data, going from the list of moves to a FEN annotated PGN file. The command I used is as follows:

```bash
pgn-extract --fencomments -Wlalg -C -N -V -w40000 --nochecks --noresults --quiet all.pgn -o all_parsed.pgn
```

where `--fencomments` option is used to add the FEN string as a comment to the moves, `-Wlalg` option is used to add the move in UCI format, `-C, -V, -N` are used to suppress comments, NAGs (Numeric Annotation Glyphs) and variations, `-w40000` option controls the line lengths and `--nochecks --noresults` options are used to exclude checks and game results.

The script `src/train/dataset.py` is then used to prepare the data for training, which will save **98.013.396 positions from 1.293.126 games in 7 numpy shards (1GB each).**

## 🔬 Evaluation

The project includes an evaluation pipeline to test the AI against Stockfish at different levels. Use the following command to evaluate a model:

```bash
python src/evaluate/evaluate.py --levels 1 2 3 --rounds 50 --model_name RandomChessBot
```

- `--levels`: Stockfish levels to evaluate against (e.g., 1, 2, 3)
- `--rounds`: Number of games to play per level and color
- `--model_name`: Name of the model to evaluate

## 🧠 Models

Here are the models currently implemented in the project and their match statistics against Stockfish at different levels. The evaluation was performed using 500 games for each color, for a total of 1000 games per level.
The statistics, taken from [this reference](https://www.chessprogramming.org/Match_Statistics), are as follows:
- **Wins**: Number of games won by the model, $w$
- **Losses**: Number of games lost by the model, $l$
- **Draws**: Number of games drawn by the model, $d$
- **Win Ratio**: Ratio of wins to total games played, calculated as $(w + \frac{d}{2})/n$, where $n$ is the total number of games played
- **Draw Ratio**: Ratio of draws to total games played
- **Likelihood of Superiority (LOS)**: Calculated using the formula $0.5*[1 + erf((w - l)/√(w + 2l))]$, it refers to the statistical chance that one player (or engine) has a higher probability of winning against another, based on their rating difference or match results


**RandomChessBot**: a simple bot that makes random moves. It serves as initial baseline for testing.

| Stockfish strength | Wins  | Losses | Draws | Win Ratio |Draw Ratio | LOS  |
| :------------:     | :---: | :----: | :---: | :------:  | :------:  |:---: |
| 1                  | 2     | 735    | 263   | 13.4%     | 26.3%     | 0%   |
| 2                  | 0     | 796    | 204   | 10.2%     | 20.4%     | 0%   |
| 3                  | 0     | 1000   | 0     |    0%     |    0%     | 0%   |
| 4                  | 0     | 1000   | 0     |    0%     |    0%     | 0%   |
| 5                  | 0     | 1000   | 0     |    0%     |    0%     | 0%   |
| 6                  | 0     | 1000   | 0     |    0%     |    0%     | 0%   |
| 7                  | 0     | 1000   | 0     |    0%     |    0%     | 0%   |

**NeuralChessBot**: a simple neural network-based bot that evaluates positions and select moves.

| Stockfish strength | Wins  | Losses | Draws | Win Ratio |Draw Ratio | LOS  |
| :------------:     | :---: | :----: | :---: | :------:  | :------:  |:---: |
| 1                  | 678   | 39     | 283   | 81.9%     | 28.3%     | 100% |
| 2                  | 472   | 188    | 340   | 64.2%     | 34.0%     | 100% |
| 3                  | 17    | 969    | 14    |  2.4%     |  1.4%     |   0% |
| 4                  | 1     | 998    | 1     |  0.2%     |  0.1%     |   0% |
| 5                  | 0     | 1000   | 0     |    0%     |    0%     |   0% |
| 6                  | 0     | 1000   | 0     |    0%     |    0%     |   0% |
| 7                  | 0     | 1000   | 0     |    0%     |    0%     |   0% |

**T22ChessBot**: a transformer-based bot that uses a transformer architecture (22M params, encoder only) to evaluate positions and select moves.

| Stockfish strength | Wins  | Losses | Draws | Win Ratio |Draw Ratio | LOS  |
| :------------:     | :---: | :----: | :---: | :------:  | :------:  |:---: |
| 1                  | 910   | 1      | 89    |  95.5%    |   8.9%    | 100% |
| 2                  | 895   | 9      | 96    |  94.3%    |   9.6%    | 100% |
| 3                  | 189   | 712    | 99    |  23.9%    |   9.9%    |   0% |
| 4                  | 13    | 967    | 20    |   2.3%    |   2.0%    |   0% |
| 5                  | 1     | 989    | 10    |   0.6%    |   1.0%    |   0% |
| 6                  | 0     | 1000   | 0     |     0%    |     0%    |   0% |
| 7                  | 0     | 1000   | 0     |     0%    |     0%    |   0% |

## 🧪 Playground

It is possible to play against the AI using the `play_vs_computer.ipynb` notebook in the `notebooks/` directory, allowing you to either select the trained models or Stockfish as the opponent.
