# 📝 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Encoder-decoder transformer chess bot for move prediction and follow-up move generation
- Reinforcement learning training loop
- Knowledge distillation from Stockfish to the neural network model


## [0.3.0] - 2025-05-05
### Added
- Add encoder only 22M transformer chess bot
- Improve training scripts, adding cosine annealing, gradient accumulation, and mixed precision training

## [0.2.0] - 2025-04-26
### Added
- Initial implementation of a MLP Neural Network
- Dataset script for creating a dataset and storing in shards
- Training scripts for custom models
- Improved evaluation scripts for testing model performance against Stockfish

## [0.1.0] - 2025-04-11
### Added
- Project structure with `src/`, `tests/`, and `notebooks/` directories
- Stockfish integration for move generation and evaluation
- Evaluation scripts for testing model performance against Stockfish
- Jupyter notebooks for human play and experimentation
- RandomChessBot as a baseline model for testing