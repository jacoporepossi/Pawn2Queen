import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

from train.dataset import ChessNpyDataset
from utils.training import init_weights, topk_acc
import config.NeuralChessBot as CONFIG
from models.NeuralChessBot import NeuralChessModel

import time
import os
import argparse
import numpy as np


def chess_collate(batch):
    """
    Custom collate function for the chess dataset.
    This function stacks the input and target tensors from the batch
    and converts them to PyTorch tensors.

    Args:
        batch (list): A list of tuples containing the input and target tensors.

    Returns:
        tuple: A tuple containing the stacked input and target tensors.
    """
    xs, ys = zip(*batch)
    return torch.from_numpy(np.stack(xs)).long(), torch.from_numpy(np.stack(ys)).long()

def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader.
    This function is called for each worker process to initialize the dataset by ensuring
    that the data is not loaded in the main process.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._x_arrays = None
    dataset._y_arrays = None

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    This function computes the average loss and accuracy on the validation set.

    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run the model on (CPU or GPU).

    Returns:
        avg_val_loss: Average validation loss.
        val_top1: Top-1 accuracy on the validation set.
        val_topk: Top-k accuracy on the validation set.
    """
    model.eval()
    val_loss = 0.0
    val_top1 = 0.0
    val_topk = 0.0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            val_top1 += topk_acc(logits, y, k=1) * x.size(0)
            val_topk += topk_acc(logits, y, k=CONFIG.TOPK) * x.size(0)
            val_total += x.size(0)

    avg_val_loss = val_loss / val_total
    val_top1 = val_top1 / val_total
    val_topk = val_topk / val_total

    return avg_val_loss, val_top1, val_topk


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model on chess data")
    parser.add_argument("model_name", type=str, nargs='?', default='mlp', help="Name of the model to train")
    parser.add_argument("num_epochs", type=int, nargs='?', default=CONFIG.NUM_EPOCHS, help="Number of epochs to train")
    parser.add_argument("batch_size", type=int, nargs='?', default=CONFIG.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("learning_rate", type=float, nargs='?', default=CONFIG.LEARNING_RATE, help="Learning rate for the optimizer")

    args = parser.parse_args()


    dataset = ChessNpyDataset(CONFIG.NPY_DATASET)

    n = len(dataset)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    train_ds = Subset(dataset, range(0, n1))
    val_ds   = Subset(dataset, range(n1, n2))
    test_ds  = Subset(dataset, range(n2, n))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=CONFIG.TRAIN_NUM_WORKERS, persistent_workers=True, worker_init_fn=worker_init_fn, collate_fn=chess_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=CONFIG.VAL_NUM_WORKERS, worker_init_fn=worker_init_fn, collate_fn=chess_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=CONFIG.TEST_NUM_WORKERS, worker_init_fn=worker_init_fn, collate_fn=chess_collate)

    size_train = len(train_loader)
    size_val = len(val_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.model_name == 'mlp':
        model = NeuralChessModel().to(device)
        model.apply(init_weights)  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3) 

    best_val_loss = float('inf')
    start_epoch = 1

    # Load checkpoint if it exists
    if os.path.exists(CONFIG.CKPT_PATH):
        print("=> Found checkpoint, loading...")
        checkpoint = torch.load(CONFIG.CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['val_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"=> Resuming from epoch {start_epoch} with best_val_loss = {best_val_loss:.4f}")

    # Set the cudnn benchmark for faster training if using GPU
    if torch.cuda.is_available():
        cudnn.benchmark = True

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        running_topk = 0.0
        total = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item() * x.size(0)
            running_top1 += topk_acc(logits, y, k=1) * x.size(0)
            running_topk += topk_acc(logits, y, k=CONFIG.TOPK) * x.size(0)
            total += x.size(0)

            if batch_idx % CONFIG.VERBOSE_INTERVAL == 0 or batch_idx == size_train:
                end = time.time()
                print(f'Epoch {epoch:2d} [{batch_idx:6d}/{size_train}] - time: {end-start:5.2f}s '
                      f'- Batch Loss: {loss.detach().item():.4f} '
                      f'- Top-1: {topk_acc(logits, y, k=1):.3f} '
                      f'- Top-{CONFIG.TOPK}: {topk_acc(logits, y, k=CONFIG.TOPK):.3f}')
                
                start = time.time()
            
            if batch_idx % CONFIG.EVAL_INTERVAL == 0 and batch_idx > 0:
                train_loss = running_loss / total
                train_top1 = running_top1 / total
                train_topk = running_topk / total
                print(f"Intermediate evaluation at batch {batch_idx}...")
                avg_val_loss, val_top1, val_topk = validate_model(model, val_loader, criterion, device)
                
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                print(f"[Epoch {epoch:2d}]"
                    f" Train Loss: {train_loss:.4f} - Top-1: {train_top1:.3f} - Top-{CONFIG.TOPK}: {train_topk:.3f} |"
                    f" Val Loss: {avg_val_loss:.4f} - Top-1: {val_top1:.3f} - Top-{CONFIG.TOPK}: {val_topk:.3f} |"
                    f" LR: {current_lr:.6f}")
                
                # ----------- SAVE ONLY THE BEST MODEL -----------
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': avg_val_loss,
                    }, CONFIG.CKPT_PATH)
                    print(f"=> Saved new best model at epoch {epoch} with val_loss {avg_val_loss:.4f}")
                
                model.train()

        train_loss = running_loss / total
        train_top1 = running_top1 / total
        train_topk = running_topk / total

        print(f"Evaluating model at epoch {epoch}, batch {batch_idx}...")
        avg_val_loss, val_top1, val_topk = validate_model(model, val_loader, criterion, device)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch:2d}]"
              f" Train Loss: {train_loss:.4f} - Top-1: {train_top1:.3f} - Top-{CONFIG.TOPK}: {train_topk:.3f} |"
              f" Val Loss: {avg_val_loss:.4f} - Top-1: {val_top1:.3f} - Top-{CONFIG.TOPK}: {val_topk:.3f} |"
              f" LR: {current_lr:.6f}")

        # Save the model if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, CONFIG.CKPT_PATH)
            print(f"=> Saved new best model at epoch {epoch} with val_loss {avg_val_loss:.4f}")
    
    print("Training complete.")

    # Test set (TODO: Redundant since we'll evaluate the best model against Stockfish, better have more training data)
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(CONFIG.CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    test_top1 = 0.0
    test_topk = 0.0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item() * x.size(0)
            test_top1 += topk_acc(logits, y, k=1) * x.size(0)
            test_topk += topk_acc(logits, y, k=CONFIG.TOPK) * x.size(0)
            test_total += x.size(0)

    avg_test_loss = test_loss / test_total
    test_acc1 = test_top1 / test_total
    test_acck = test_topk / test_total

    print(f"Test Loss: {avg_test_loss:.4f} | Test Top-1: {test_acc1:.3f} | Test Top-{CONFIG.TOPK}: {test_acck:.3f}")

