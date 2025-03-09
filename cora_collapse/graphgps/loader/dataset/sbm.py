from torch_geometric.graphgym.register import register_loader

import os
import shutil
from enum import Enum
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from typing import Optional, Callable, List
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureStrategy(Enum):
    EMPTY = "empty"
    DEGREE = "degree"
    RANDOM = "random"
    RANDOM_NORMAL = "random_normal"
    DEGREE_RANDOM = "degree_random"
    DEGREE_RANDOM_NORMAL = "degree_random_normal"
    @classmethod
    def present(cls, val):
        return val in [mem.value for mem in cls.__members__.values()]

class Fixed_SBM(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.args = {'device':device}
        self.N = 1000
        self.C = 2
        self.Pr = np.array([0.5, 0.5])
        self.W = self.prepare_W(p=0.017, q=0.0025)
        self.num_graphs = 1000
        self.feature_strategy = "random_normal"
        self.feature_dim = 5
        self.permute_nodes = True
        self.dataset_dir = ""
        self.is_training = True
        self.graphs_list = []
        # placeholders for pyg/python version compatibility
        self._indices = None
        self.transform = None
        self.validate()
        self.prepare_paths()
        self.load_data()

    def prepare_W(self, p, q):
        W = []
        for i in range(self.C):
            row = []
            for j in range(self.C):
                val = p if i==j else q
                row.append(val)
            W.append(row)
        return np.array(W)

    def validate(self):
        """Validate the parameters of the model"""
        if len(self.Pr) != self.C:
            raise ValueError("length of {} should be equal to {}".format(self.Pr, self.C))
        if np.sum(self.Pr) != 1.0:
            raise ValueError("Values of {} should sum to 1".format(self.Pr))
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError("{} should be symmetric".format(self.W))
        if not np.all(self.W == self.W.transpose()):
            raise ValueError("{} should be symmetric".format(self.W))
        if self.W.shape[0] != self.C:
            raise ValueError("Shape of {} should be ({}, {})".format(self.W, self.C, self.C))
        if not FeatureStrategy.present(self.feature_strategy):
            raise ValueError("Invalid feature_strategy={}. \
                Should be one of 'empty', 'degree', 'random', 'degree_random'".format(self.feature_strategy))
        if self.feature_strategy in ["random", "degree_random"] and self.feature_dim == 0:
            raise ValueError("feature_dim = 0 when random features were desired.")

    def prepare_paths(self):
        data_dir = "N_{}_C_{}_Pr_{}_p_{}_q_{}_num_graphs_{}_feat_strat_{}_feat_dim_{}_permute_{}".format(
            self.N, self.C, self.Pr, self.W[0,0], self.W[0,1], self.num_graphs, self.feature_strategy, self.feature_dim, self.permute_nodes
        )
        if self.is_training:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm/train")
        else:
            dataset_dir = os.path.join(self.dataset_dir, "data/sbm/test")
        self.dataset_path = os.path.join(dataset_dir, data_dir)

    def save_data(self):
        print("Saving data")
        os.makedirs(self.dataset_path)
        torch.save(self.graphs_list, self.dataset_path+"/data.pt")

    def load_data(self):
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
            # print("Loading data from filesystem")
            # self.graphs_list = torch.load(self.dataset_path+"/data.pt")
        # else:
        print("Generating data")
        self.generate_data()
        self.save_data()

    def generate_single_graph(self):
        """Generate a single SBM graph"""
        M = torch.ones((self.N, self.N))
        comm_num_nodes = [math.floor(self.N * p_c) for p_c in self.Pr[:-1]]
        comm_num_nodes.append(self.N - np.sum(comm_num_nodes))
        row_offset = 0
        labels = []
        for comm_idx in range(self.C):
            curr_comm_num_nodes = comm_num_nodes[comm_idx]
            labels.extend([comm_idx for _ in range(curr_comm_num_nodes)])
            col_offset = 0
            for iter_idx, _num_nodes in enumerate(comm_num_nodes):
                M[row_offset:row_offset+curr_comm_num_nodes, col_offset:col_offset+_num_nodes] = self.W[comm_idx, iter_idx]
                col_offset += _num_nodes
            row_offset += curr_comm_num_nodes

        if not torch.allclose(M, M.t()):
            raise ValueError("Error in preparing X matrix", M)

        labels = torch.Tensor(labels).type(torch.int)
        if torch.cuda.is_available():
            labels = labels.cuda()
        Adj = torch.rand((self.N, self.N)) < M
        Adj = Adj.type(torch.int)
        # Although the effect is minor, comment the following line to
        # experiment with self-loop graphs.
        Adj = Adj * (torch.ones(self.N) - torch.eye(self.N))
        Adj = torch.maximum(Adj, Adj.t())
        X = self.get_features(Adj=Adj)

        # permute nodes and corresponding features, labels
        if self.permute_nodes:
            perm = torch.randperm(self.N)
            labels = labels[perm]
            Adj = Adj[perm]
            Adj = Adj[:, perm]
            if self.feature_strategy != "empty":
                X = X[perm]

        indices = torch.nonzero(Adj)
        edge_index = indices.to(torch.long)
        data = Data(x=X, y=labels, edge_index=edge_index.t().contiguous())
        previous_embedding = data['x']
        # if(self.transform is not None):
        #     data = self.transform(data)
        #     print(data['x'].shape)
        return data

    def get_features(self, Adj):
        """Prepare the features for the nodes based on
        feature_strategy and adjacency matrix

        Args:
            Adj: Adjacency matrix of the graph

        Returns:
            Feature tensor X of shape (n, d) if d > 0 else
            returns an empty tensor
        """
        if self.feature_strategy == "empty":
            return torch.Tensor(())
        elif self.feature_strategy == "degree":
            X = torch.sum(Adj, 1).unsqueeze(1)
            return X
        elif self.feature_strategy == "random":
            X = torch.rand((self.N, self.feature_dim))
            return X
        elif self.feature_strategy == "random_normal":
            X = torch.randn((self.N, self.feature_dim))
            return X
        elif self.feature_strategy == "degree_random":
            X = torch.zeros((self.N, self.feature_dim))
            X[:, 0] = torch.sum(Adj, 1)
            X[:, 1:] = torch.rand((self.N, self.feature_dim-1))
            return X
        elif self.feature_strategy == "degree_random_normal":
            X = torch.zeros((self.N, self.feature_dim))
            X[:, 0] = torch.sum(Adj, 1)
            X[:, 1:] = torch.randn((self.N, self.feature_dim-1))
            return X

    def generate_data(self):
        print("Generating {} graphs".format(self.num_graphs))
        for _ in tqdm(range(self.num_graphs)):
            res = self.generate_single_graph()
            self.graphs_list.append(res)

    def len(self):
        """Return the number of graphs to be sampled"""
        return self.num_graphs

    def get(self, index):
        """Return a single sbm graph"""
        return self.graphs_list[index].to(self.args["device"])