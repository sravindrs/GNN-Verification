# ast_to_pyg.py
"""
This script defines a function, sv_to_data(file_path), that converts a SystemVerilog (.sv) file 
into a PyTorch Geometric Data object for GNN training. It parses the file using pyslang, traverses
the resulting AST, and extracts features for each node, including:
    - type (encoded as an integer)
    - depth (node depth in the AST)
    - num_children (number of child nodes)
    - parent_type (type of the parent node)
    - subtree_size (total number of descendant nodes)
    - start_pos and end_pos (start and end positions in the source file)
    - port_count (for module declarations only)
The function returns a Data object with node features and edge indices, suitable for GNN models.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from pyslang import SyntaxTree, SyntaxKind

def sv_to_data(file_path):
    tree = SyntaxTree.fromFile(file_path)
    root_node = tree.root
    nodes = []
    edges = []

    def traverse_ast(node, parent_id=None, depth=0):
        node_id = len(nodes)
        node_data = {
            "id": node_id,
            "type": str(node.kind),
            "depth": depth,
            "num_children": len(getattr(node, 'members', [])),
            "parent_type": str(node.parent.kind) if hasattr(node, 'parent') and node.parent else "None",
            "subtree_size": len(list(getattr(node, 'descendants', []))),
            "start_pos": getattr(node, 'span', None).start if hasattr(node, 'span') else 0,
            "end_pos": getattr(node, 'span', None).end if hasattr(node, 'span') else 0,
            "port_count": len(getattr(node, 'ports', [])) if node.kind == SyntaxKind.ModuleDeclaration else 0
        }
        nodes.append(node_data)
        if parent_id is not None:
            edges.append((parent_id, node_id))

        if hasattr(node, 'members'):
            for child in node.members:
                traverse_ast(child, node_id, depth + 1)
        elif hasattr(node, '__iter__'):
            for child in node:
                traverse_ast(child, node_id, depth + 1)

    traverse_ast(root_node)
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges, columns=['source', 'target'])

    node_type_to_index = {node_type: idx for idx, node_type in enumerate(nodes_df['type'].unique())}
    nodes_df['type_idx'] = nodes_df['type'].map(node_type_to_index)

    node_features = torch.tensor(
        nodes_df[['type_idx', 'depth', 'num_children', 'subtree_size', 'start_pos', 'end_pos', 'port_count']].fillna(0).values,
        dtype=torch.float
    )

    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)
