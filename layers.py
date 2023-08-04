from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.utils import get_laplacian
import logging
logger = logging.getLogger(__file__)
import pickle,os

from torch_geometric.nn.conv.gcn_conv import *

class HOConv(GCNConv):
    """
    A class that add high-order random walk
    to conventional GCN
    """
    def __init__(self,input_dim,output_dim,edge_index,
                 num_nodes,alpha):
        """
        input_dim: dimension of node features
        output_dim: output dimension 
        edge_index: a Tensor with size 2*number_of_edges
        num_nodes: number of nodes/proteins
        alpha: the probability of restart in random walk
        """
        super().__init__(input_dim,output_dim)
        self.cuda_device = torch.device("cuda:0")
        I_matrix = torch.eye(num_nodes,device=self.cuda_device)
        norm_edge_index,norm_edge_weight = get_laplacian(edge_index,normalization="sym")
        laplacian = torch.sparse_coo_tensor(norm_edge_index,norm_edge_weight).cuda()
        self.A_norm = I_matrix - laplacian
        self.Y = I_matrix - (1-alpha)*self.A_norm
        self.Y = alpha*torch.linalg.inv(self.Y)
        self.Y = self.Y.to(self.cuda_device)

    def forward(self,x,edge_index,edge_weight=None):
        """
        Overwrite forward method of forward in
        original GCNConv 
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
        #                     size=None)
        #
        #
        # ####### Dingquan YU 02/06/2021 below overwrite propagate_method ######## #
        out = torch.matmul(self.Y,x)

        if self.bias is not None:
            out = out + self.bias

        return out