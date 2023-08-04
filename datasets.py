"""
Create pytorch dataset objects
"""
from torch.utils.data import Dataset
import networkx as nx
from utils import create_adj_mtx
import numpy as np
from torch_geometric.data import Dataset as TorchGeometricDataset
from torch_geometric.data import Data

class InputData(TorchGeometricDataset):
    """
    Create a TorchGeometricDataset object that 
    can split train,test, validation edges
    """
    def __init__(self,root,transform,x,edge_index):
        super().__init__(root,transform)
        self.x = x
        self.edge_index =edge_index

    def len(self):
        return 1
    
    def get(self,idx):
        data = Data(self.x,self.edge_index)
        return data 