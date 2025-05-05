import torch
import torch.nn as nn

from train.dataset import get_batch
from utils.training import init_weights, init_transformer_weights, topk_acc, get_lr, seed_everything, inflate_weights
import config.NeuralChessBot as NNCONFIG
import config.T22ChessBot as T22CONFIG
import config.T100ChessBot as T100CONFIG
from models.NeuralChessBot import NeuralChessModel
from models.T22ChessBot import T22ChessModel
from models.T100ChessBot import T100ChessModel

from pathlib import Path
import time
import os
import argparse
from collections import defaultdict
from pprint import pprint


@torch.no_grad()
def estimate_loss(criterion, eval_iters, top_k=5):
    """
    Validate the model on the validation set.
    This function computes the average loss and accuracy on the validation set.

    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.

    Returns:
        avg_val_loss: Average validation loss.
        val_top1: Top-1 accuracy on the validation set.
        val_topk: Top-k accuracy on the validation set.
    """
    out = defaultdict(dict)
    model.eval()
    losses = torch.zeros(eval_iters)
    top1s = torch.zeros(eval_iters)
    topks = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(CONFIG.NPY_DATASET, CONFIG.BATCH_SIZE, device=device, split='val')
        with CONFIG.ctx:
            logits = model(x)
            loss = criterion(logits, y)
        losses[k] = loss.item()
        top1s[k] = topk_acc(logits, y, k=1)
        topks[k] = topk_acc(logits, y, k=top_k)

        out['loss'] = losses.mean().item()
        out['top1'] = top1s.mean().item()
        out['topk'] = topks.mean().item()

    model.train()
    return out


if __name__ == "__main__":
    seed_everything()
        
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Train a model on chess data")
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_name == 'NeuralChessBot':
        CONFIG = NNCONFIG
        model = NeuralChessModel().to(device)
        model.apply(init_weights) 
    elif args.model_name == 'T22ChessBot':
        CONFIG = T22CONFIG
        model = T22ChessModel().to(device)
        model.apply(init_transformer_weights) 
    elif args.model_name == 'T100ChessBot':
        CONFIG = T100CONFIG
        model = T100ChessModel().to(device)
        model.apply(init_transformer_weights)

        if True:  # TODO: remove this block when T100 model is ready
            CONFIG22 = T22CONFIG
            model22 = T22ChessModel().to(device)
            CKPT_PATH = Path(__file__).resolve().parents[2] / 'checkpoints/t22_best_model.pth'
            checkpoint = torch.load(CKPT_PATH, map_location=device)
            model22.load_state_dict(checkpoint['model_state_dict'])
            model = inflate_weights(small_model=model22, large_model=model)
            del model22, checkpoint
            torch.cuda.empty_cache()

    # Load checkpoint if it exists
    if os.path.exists(CONFIG.CKPT_PATH):
        print("=> Found checkpoint, loading...")
        checkpoint = torch.load(CONFIG.CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
    elif not os.path.exists(CONFIG.CKPT_PATH.parent):
        os.makedirs(CONFIG.CKPT_PATH.parent, exist_ok=True)
        print(f"=> Checkpoint path {CONFIG.CKPT_PATH.parent} created.")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Running {args.model_name} model with CONFIG:")
    pprint({k: v for k, v in vars(CONFIG).items() if not k.startswith("__") and k not in ["CKPT_PATH", "NPY_DATASET", "Path"]})
    print(f"\n> Total model parameters: {params:,}".replace(",", "."))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    start_iter = 0

    if os.path.exists(CONFIG.CKPT_PATH):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        start_iter = checkpoint['iter']
        print(f"=> Resuming from iter {start_iter} with best_val_loss = {best_val_loss:.4f}")

    checkpoint = None # Free up memory and avoid a very weird bug that made the time/iter to take 10x longer
    torch.cuda.empty_cache()

    session_iter = 0
    x, y = get_batch(CONFIG.NPY_DATASET, CONFIG.BATCH_SIZE, device=device, split='train')

    for it in range(start_iter, CONFIG.MAX_ITERS):
        iter_start = time.time()
        lr = get_lr(it, CONFIG.WARMUP_ITERS, CONFIG.LR_DECAY_ITERS, CONFIG.LEARNING_RATE, CONFIG.MIN_LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(CONFIG.GRADIENT_ACCUMULATION_STEPS):
            with CONFIG.ctx:
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / CONFIG.GRADIENT_ACCUMULATION_STEPS
            x, y = get_batch(CONFIG.NPY_DATASET, CONFIG.BATCH_SIZE, device=device, split='train')
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        iter_end = time.time()

        if it % CONFIG.VERBOSE_INTERVAL == 0 and session_iter > 0:

            print(f'Iter {it:8d} - time/iter: {(iter_end - iter_start):5.2f}s '
                  f'- Loss: {loss.detach().item() * CONFIG.GRADIENT_ACCUMULATION_STEPS:.4f} '
                  f'- LR: {lr:.6f}'
                  )

        if it % CONFIG.EVAL_INTERVAL == 0 and session_iter > 0:
            print(f"Intermediate evaluation at iter {it}...")
            outs = estimate_loss(criterion, CONFIG.EVAL_ITERS)
            
            print(f"[Iter {it:8d}]"
                f" Train Loss: {loss.detach().item()* CONFIG.GRADIENT_ACCUMULATION_STEPS:.4f} |"
                f" Val Loss: {outs['loss']:.4f} - Top-1: {outs['top1']:.3f} - Top-5: {outs['topk']:.3f} |"
                f" LR: {lr:.6f}")
            
            # ----------- SAVE ONLY THE BEST MODEL -----------
            if outs['loss'] < best_val_loss:
                best_val_loss = outs['loss']
                torch.save({
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': outs['loss'],
                }, CONFIG.CKPT_PATH)
                print(f"=> Saved new best model at iter {it} with val_loss {outs['loss']:.4f}")
        session_iter += 1
    print(f"Training complete")
