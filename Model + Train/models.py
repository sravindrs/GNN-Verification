import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

class HDLGraphAttention(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.6, alpha=0.2, nheads=8):
        """
        HDL-specific version of GAT for assertion classification.
        
        Args:
            nfeat: Number of input features (768 for GraphCodeBERT embeddings)
            nhid: Number of hidden features
            dropout: Dropout rate
            alpha: LeakyReLU angle of negative slope
            nheads: Number of attention heads
        """
        super(HDLGraphAttention, self).__init__()
        self.dropout = dropout

        # Multiple attention heads for initial layer
        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True
        ) for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        # Second attention layer
        self.attention_2 = GraphAttentionLayer(
            nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True
        )

        # Final classification layers
        self.fc1 = nn.Linear(nhid, nhid // 2)
        self.fc2 = nn.Linear(nhid // 2, 1)  # Binary classification
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid // 2)

    def forward(self, data):
        """
        Forward pass for HDL graph classification.
        
        Args:
            data: PyG Data object containing:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge weights/types
                - batch: Batch indices for multiple graphs
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Convert edge_index to dense adjacency matrix
        N = x.size(0)
        adj = torch.zeros((N, N), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1

        # Apply edge weights if available
        if edge_attr is not None:
            adj[edge_index[0], edge_index[1]] = edge_attr

        # First GAT layer with multiple attention heads
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        
        # Second GAT layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention_2(x, adj)
        x = self.bn1(x)
        
        # Global mean pooling
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)

        # Final classification layers
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)  # Binary classification

    def get_attention_weights(self, data):
        """
        Extract attention weights for visualization/analysis.
        """
        attention_weights = []
        x, edge_index = data.x, data.edge_index
        
        N = x.size(0)
        adj = torch.zeros((N, N), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1

        # Get attention weights from first layer
        for att in self.attentions:
            weights = att._prepare_attentional_mechanism_input(
                torch.mm(x, att.W)
            )
            attention_weights.append(weights.detach())

        return attention_weights