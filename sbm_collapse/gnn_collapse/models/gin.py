import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from gnn_collapse.models.common import Normalize

class GINModel(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                 loss_type, num_classes, L=3, batch_norm=True,
                 non_linearity="relu", use_bias=False, use_W1=False):
        # use_w1 is a dummy parameter, not used at all by this model but included for easy compatibility with online.py
        super().__init__()
        self.name = "gin_model" # String name assignment
        self.L = L # Number of Layers
        self.non_linearity = non_linearity # Type of nonlinearity
        self.loss_type = loss_type # Loss function
        self.batch_norm = batch_norm # Boolean - yes/no to using batch norm
        self.norm = Normalize(hidden_feature_dim, norm="batch") # Batch norm itself
        nn = torch.nn.Linear(input_feature_dim, hidden_feature_dim)
        self.proj_layer = GINConv(nn, train_eps=False, eps=0.0)
                                                    
        self.layers = torch.nn.ModuleList()
        # Simplified MLP in GINConv - single layer instead of two
        for i in range(self.L):
            if i == 0:
                nn = torch.nn.Linear(input_feature_dim, hidden_feature_dim)
            else:
                nn = torch.nn.Linear(hidden_feature_dim, hidden_feature_dim)
            # Disable eps training to remove adaptivity
            conv = GINConv(nn, train_eps=False, eps=0.0)
            self.layers.append(conv)

        if self.non_linearity == "relu":
            self.non_linear_layers = [torch.nn.ReLU()  for _ in range(L)] # All nonlinearities
        else:
            self.non_linear_layers = []
        if self.batch_norm:
            self.normalize_layers = [Normalize(hidden_feature_dim, norm="batch")  for _ in range(L)] # All batch norms
        else:
            self.normalize_layers = []

        # Making graph "convolutions", relus, batch norms, packaged and accessible
        self.conv_layers = torch.nn.ModuleList(self.layers)
        self.non_linear_layers = torch.nn.ModuleList(self.non_linear_layers)
        self.normalize_layers = torch.nn.ModuleList(self.normalize_layers) 
        # Final layer
        nn = torch.nn.Linear(hidden_feature_dim, num_classes)
        self.final_layer = GINConv(nn, train_eps=False, eps=0.0)
                                                                    
                                                                        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.proj_layer(x, edge_index)
        if self.non_linearity == "relu":
            x = F.relu(x)
        if self.batch_norm:
            x = self.norm(x)
        
        for l in range(self.L):
            x = self.conv_layers[l](x, edge_index)
            if self.non_linearity == "relu":
                x = self.non_linear_layers[l](x)
            if self.batch_norm:
                x = self.normalize_layers[l](x)

        x = self.final_layer(x, edge_index) 
        if self.loss_type == "mse":
            return x
        return F.log_softmax(x, dim=1)
