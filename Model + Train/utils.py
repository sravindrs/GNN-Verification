import numpy as np
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_hdl_data(pt_file_path, batch_size=32, split_ratio=[0.7, 0.15, 0.15]):
    """Load HDL graph dataset from processed .pt file"""
    print('Loading HDL dataset...')
    
    # Load the saved data
    data = torch.load(pt_file_path)
    graphs = data['graphs']
    labels = data['labels']
    
    # Calculate split indices
    num_graphs = len(graphs)
    num_train = int(num_graphs * split_ratio[0])
    num_val = int(num_graphs * split_ratio[1])
    
    # Randomly shuffle the data
    indices = torch.randperm(num_graphs)
    
    # Split into train, validation, and test
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    # Create data loaders
    train_loader = DataLoader([graphs[i] for i in train_indices], 
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_indices], 
                          batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([graphs[i] for i in test_indices], 
                           batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def normalize_features(x):
    """Normalize node features"""
    row_sum = x.sum(1, keepdim=True)
    row_sum[row_sum == 0] = 1  # Avoid division by zero
    return x / row_sum

def evaluate_model(model, loader, device):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = (output > 0.5).float().cpu()
            predictions.extend(pred.numpy())
            labels.extend(data.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(model, path, epoch, optimizer, best_val_acc):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }, path)

def load_model(model, path, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_acc']

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0