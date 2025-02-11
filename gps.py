import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    CrossEntropyLoss,
    Embedding
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention

from gnn_collapse.data.sbm import SBM
from torch_geometric.loader import DataLoader



# # Load Cora dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0].to(device)

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--attn_type', default='multihead',
#     help="Global attention type such as 'multihead' or 'performer'.")
# args = parser.parse_args()

class GPS(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int, attn_type: str = 'multihead', attn_kwargs: dict = None):
        super().__init__()


        if attn_kwargs is None:
            attn_kwargs = {'dropout': 0.3}

        
        self.lin1 = Linear(1, hidden_channels)
        
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels, GINConv(nn), heads=8,
                          attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index):
        x = self.lin1(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
        
        return x


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1



def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = SBM(
        args=args,
        N=args["N_train"],
        C=args["C"],
        Pr=args["Pr"],
        p=args["p_train"],
        q=args["q_train"],
        num_graphs=args["num_train_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["input_feature_dim"],
        is_training=True
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=1)


    attn_kwargs = {'dropout': 0.0}


    model = GPS(
        in_channels= args['input_feature_dim'],
        hidden_channels = args['input_feature_dim'],
        num_classes = args['C'],
        num_layers = 4,
        attn_type= 'multihead',
        attn_kwargs= attn_kwargs
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    criterion = CrossEntropyLoss()


    for epoch in range(1, 501):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            labels = data.y.long()
            loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.5f}%')

    
if __name__ == "__main__":
    # Example usage with args from your configuration

    args = {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "N_train": 100,
        "C": 2,
        "Pr": [0.5, 0.5],
        "p_train": 0.5,
        "q_train": 0.1,
        "num_train_graphs": 1,  # Set to 1 to see single instance
        "feature_strategy": "degree",  # Using degree features
        "feature_dim": 64,
        "input_feature_dim": 64,
        "num_layers": 4,
        "lr" : 0.01,
        "weight_decay": 5e-4
    }

    dataset = SBM(
        args=args,
        N=args["N_train"],
        C=args["C"],
        Pr=args["Pr"],
        p=args["p_train"],
        q=args["q_train"],
        num_graphs=args["num_train_graphs"],
        feature_strategy=args["feature_strategy"],
        feature_dim=args["feature_dim"],
        is_training=True
    )

    data = dataset[0]
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Labels shape: {data.y.shape}")
    print(f"\nClass distribution: {torch.bincount(data.y)}")
    print(f"\nFirst few edges:\n{data.edge_index[:, :5]}")
    print(f"\nFirst few node features:\n{data.x[:5]}")
    print(f"\nFirst few labels:\n{data.y[:5]}")
    train_model(args)


