import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_collapse.models.common import Normalize


class GCNSubModule(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, batch_norm, non_linearity, use_bias):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.non_linearity = non_linearity
        self.batch_norm=batch_norm
        self.conv = GCNConv(self.in_feature_dim, self.out_feature_dim, bias=use_bias)
        self.norm = Normalize(self.out_feature_dim, norm="batch")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=True):
        super().__init__()
        self.name = "gcn"
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.norm = Normalize(hidden_feature_dim, norm="batch")
        self.loss_type = loss_type
        self.proj_layer = GCNConv(input_feature_dim, hidden_feature_dim, bias=use_bias)
        self.conv_layers = [
            GCNSubModule(
                in_feature_dim=hidden_feature_dim,
                out_feature_dim=hidden_feature_dim,
                batch_norm=batch_norm,
                non_linearity=non_linearity,
                use_bias=use_bias
            )
            for _ in range(L)
        ]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.final_layer = GCNConv(hidden_feature_dim, num_classes, bias=use_bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
        x = self.final_layer(x, edge_index)
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
