from pathlib import Path

# Training parameters
NPY_DATASET = Path(__file__).resolve().parents[2] / 'data/npy_shards'
CKPT_PATH = Path(__file__).resolve().parents[2] / 'checkpoints/mlp_best_model.pth'
EVAL_INTERVAL = 50000
TRAIN_NUM_WORKERS = 12
VAL_NUM_WORKERS = 12
TEST_NUM_WORKERS = 12
LEARNING_RATE = 0.001
PATIENCE_DECAY = 3
REDUCE_FACTOR = 0.8
BATCH_SIZE = 512
NUM_EPOCHS = 10
VERBOSE_INTERVAL = 1000

# Model parameters 
EMB_SIZE = 64
INPUT_SIZE = 77
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1968