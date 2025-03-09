import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from gnn_collapse.models.common import Normalize

class SMPNN(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim,
                loss_type, num_classes, L=3, batch_norm=True,
                non_linearity="silu", use_bias=False, use_W1=False):
        super().__init__()
        self.name = "smpnn_model" # String name assignment
        self.L = L # Number of Layers
        self.non_linearity = non_linearity # Type of nonlinearity
        self.loss_type = loss_type # Loss function
        self.batch_norm = batch_norm # Boolean - yes/no to using batch norm
        self.norm = Normalize(hidden_feature_dim, norm="batch") # Batch norm itself

        # Initial Map to Latenet Space
        self.linear_start = torch.nn.Linear(input_feature_dim, hidden_feature_dim)

        # GCN Layers and Norms
        self.layernorms_gcn = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_feature_dim) for _ in range(L)])
        self.conv_layers = torch.nn.ModuleList([GCNConv(hidden_feature_dim, hidden_feature_dim) for _ in range(L)])
        self.alphas_gcn = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1e-6)) for _ in range(L)])

        if self.non_linearity == "relu":
            self.non_linear = torch.nn.ReLU() # SiLU activation function
            self.non_linear_layers = torch.nn.ModuleList([torch.nn.ReLU()  for _ in range(L)]) # All nonlinearities
            self.non_linear_layers_gcn = torch.nn.ModuleList([torch.nn.ReLU()  for _ in range(L)]) # All nonlinearities
        if self.non_linearity == "silu":
            self.non_linear = torch.nn.SiLU() # SiLU activation function
            self.non_linear_layers_gcn = torch.nn.ModuleList([torch.nn.SiLU()  for _ in range(L)]) # All nonlinearities
            self.non_linear_layers = torch.nn.ModuleList([torch.nn.SiLU()  for _ in range(L)]) # All nonlinearities
        else:
            self.non_linear = torch.nn.Identity() # Identity activation function
            self.non_linear_layers = []
            self.non_linear_layers_gcn = torch.nn.ModuleList(self.non_linear_layers)
            self.non_linear_layers = torch.nn.ModuleList(self.non_linear_layers)
        
        # Feedforward Layers
        self.normalize_layers = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_feature_dim) for _ in range(L)])
        self.ffw = torch.nn.ModuleList([torch.nn.Linear(hidden_feature_dim, hidden_feature_dim) for _ in range(L)])
        self.alphas_ff = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1e-6)) for _ in range(L)])

        # Final Layer
        self.final_layer = torch.nn.Linear(hidden_feature_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = self.non_linear(self.linear_start(x))
        for i in range(self.L):
            x = self.forward_gcn(x, edge_index, i)
            x = self.pointwise_ff(x, i)
        x = F.log_softmax(self.final_layer(x), dim=1)
        return x

    def forward_gcn(self, x, edge_index, layer_idx):
        conv_x = self.layernorms_gcn[layer_idx](x)
        conv_x = self.conv_layers[layer_idx](conv_x, edge_index)
        conv_x = self.non_linear_layers_gcn[layer_idx](conv_x)
        x = (self.alphas_gcn[layer_idx] * conv_x) + x
        return x

    def pointwise_ff(self, x, layer_idx):
        norm_x = self.normalize_layers[layer_idx](x)
        x = self.alphas_ff[layer_idx] * self.non_linear_layers[layer_idx](self.ffw[layer_idx](norm_x)) + x
        return x
    