from pathlib import Path
from contextlib import nullcontext
import torch

# Training parameters
NPY_DATASET = Path(__file__).resolve().parents[2] / 'data/npy_shards_v2'
CKPT_PATH = Path(__file__).resolve().parents[2] / 'checkpoints/t22_best_model.pth'

EVAL_ITERS = 1000
MAX_ITERS = 190_000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 5e-5 # learning rate / 10
LR_DECAY_ITERS = MAX_ITERS
BATCH_SIZE = 120
WARMUP_ITERS = 1000
GRADIENT_ACCUMULATION_STEPS = 2
VERBOSE_INTERVAL = 50

# Model parameters 
VOCAB_SIZE = 77
SEQ_LENGTH = 77 # 77 is the max length of the input sequence
EMB_SIZE = 384
NUM_HEADS = 16
NUM_LAYERS = 12
assert EMB_SIZE % NUM_HEADS == 0, "Embedding size must be divisible by number of heads"
HEAD_SIZE = EMB_SIZE//NUM_HEADS
OUTPUT_DIM = 1968

# Creating ctx for mixed precision training
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)