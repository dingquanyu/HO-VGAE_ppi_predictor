import torch
from layers import HOConv
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import logging
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import ExponentialLR
logger = logging.getLogger(__file__)
from torch_geometric.nn import GCNConv

class HOVariationalGraphEncoder(torch.nn.Module):
    """
    High-order variational graph encoder
    Compared to original variational encoder,HOVE 
    added an extra step of random walk with a probability of 
    restart (alpha) in the GCN layer 
    """
    def __init__(self, input_dim,edge_index,num_nodes,alpha) -> None:
        super().__init__()
        self.cuda_device = torch.device("cuda:0")
        self.hidden1 = HOConv(input_dim,256,edge_index,num_nodes,alpha)
        self.hidden_2_conv_mean = HOConv(256,128,edge_index,num_nodes,alpha)
        self.hidden_2_conv_logstd = HOConv(256,128,edge_index,num_nodes,alpha)
        self.conv_mean = HOConv(128,64,edge_index,num_nodes,alpha)
        self.conv_logstd = HOConv(128,64,edge_index,num_nodes,alpha)
    
    def forward(self,x,edge_index):
        x = x.float()
        x =self.hidden1(x,edge_index)
        hidden_2_conv_mean = self.hidden_2_conv_mean(x,edge_index)
        hidden_2_conv_logstd = self.hidden_2_conv_logstd(x,edge_index)
        return self.conv_mean(hidden_2_conv_mean,edge_index),self.conv_logstd(hidden_2_conv_logstd,edge_index)
    
class VariationalGraphEncoder(torch.nn.Module):
    """
    Original variational graph encoder 
    Implemented according to Kipf and Welling (2016) https://arxiv.org/abs/1611.07308
    """
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.cuda_device = torch.device("cuda:0")
        self.hidden1 = GCNConv(input_dim,256)
        self.conv_mean = GCNConv(256,128)
        self.conv_logstd = GCNConv(256,128)
    
    def forward(self,x,edge_index):
        x = x.float()
        x =self.hidden1(x,edge_index)
        return self.conv_mean(x,edge_index),self.conv_logstd(x,edge_index)