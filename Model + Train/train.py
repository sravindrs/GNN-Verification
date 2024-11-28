from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_hdl_data, evaluate_model, save_model, EarlyStopping
from models import HDLGraphAttention

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
parser.add_argument('--data_path', type=str, default='processed_data_attention.pt', help='Path to processed data file.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
train_loader, val_loader, test_loader = load_hdl_data(
    args.data_path,
    batch_size=args.batch_size
)

# Get input dimension from first batch
sample_batch = next(iter(train_loader))
nfeat = sample_batch.x.size(1)

# Initialize model
model = HDLGraphAttention(
    nfeat=nfeat,
    nhid=args.hidden,
    dropout=args.dropout,
    alpha=args.alpha,
    nheads=args.nb_heads
).to(device)

optimizer = optim.Adam(model.parameters(), 
                      lr=args.lr, 
                      weight_decay=args.weight_decay)

# Initialize early stopping
early_stopping = EarlyStopping(patience=args.patience)

def train(epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        
        # Compute loss (binary cross entropy for binary classification)
        loss = F.binary_cross_entropy(output.squeeze(), batch.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        pred = (output.squeeze() > 0.5).float()
        correct = pred.eq(batch.y.float()).sum().item()
        
        total_loss += loss.item() * batch.num_graphs
        total_correct += correct
        total_samples += batch.num_graphs
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc

def validate():
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Compute loss
            loss = F.binary_cross_entropy(output.squeeze(), batch.y.float())
            
            # Compute accuracy
            pred = (output.squeeze() > 0.5).float()
            correct = pred.eq(batch.y.float()).sum().item()
            
            total_loss += loss.item() * batch.num_graphs
            total_correct += correct
            total_samples += batch.num_graphs
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc

# Training loop
best_val_acc = 0
t_total = time.time()

for epoch in range(args.epochs):
    t = time.time()
    
    # Training phase
    train_loss, train_acc = train(epoch)
    
    # Validation phase
    val_loss, val_acc = validate()
    
    # Print epoch results
    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {train_loss:.4f}',
          f'acc_train: {train_acc:.4f}',
          f'loss_val: {val_loss:.4f}',
          f'acc_val: {val_acc:.4f}',
          f'time: {time.time() - t:.4f}s')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model, 'best_model.pt', epoch, optimizer, best_val_acc)
    
    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Final testing
print("Testing on test set...")
test_metrics = evaluate_model(model, test_loader, device)
print("Test set results:",
      "accuracy= {:.4f}".format(test_metrics['accuracy']),
      "precision= {:.4f}".format(test_metrics['precision']),
      "recall= {:.4f}".format(test_metrics['recall']),
      "f1= {:.4f}".format(test_metrics['f1']))