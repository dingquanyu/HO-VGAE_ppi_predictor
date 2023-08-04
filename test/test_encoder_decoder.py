import unittest
from layers import Encoder,Decoder
import pickle
import numpy as np
import torch
import networkx as nx
from model import HoVGAE
from utils import * 
import nvsmi,os

class testEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.G = nx.read_gml('./input_data/influenza_human_PPN_clean.gml')
        self.n_nodes = len(list(self.G.nodes))
        self.edges = pickle.load(open("./input_data/influenza_host_edges_clean.pkl",'rb'))
        testEncoder.original_adj_mtx = create_adj_mtx(self.G)
        testEncoder.test_edges, testEncoder.val_edges, testEncoder.train_edges, testEncoder.edges=train_test_edges_split(self.edges)
        testEncoder.test_edges_false,testEncoder.val_edges_false = get_equal_number_of_false_edges(testEncoder.test_edges, testEncoder.val_edges, testEncoder.train_edges, testEncoder.edges)
        testEncoder.adj_mtx = change_original_adj_mtx(testEncoder.original_adj_mtx,testEncoder.test_edges,testEncoder.val_edges)
        testEncoder.A_norm = create_A_norm(testEncoder.adj_mtx)
        testEncoder.adj_mtx = torch.from_numpy(np.asarray(testEncoder.adj_mtx))
        return super().setUp()
    
    # def test_1_encoder(self):
    #     """Right now haven't got the node feature matrix so use adj matrix as feature matrix instead"""
    #     testEncoder.encoder = Encoder(self.A_norm,alpha=0.3)
    #     testEncoder.z,testEncoder.z_mean,testEncoder.z_log_std_div = testEncoder.encoder(testEncoder.adj_mtx)
    #     self.assertListEqual(list(testEncoder.z.shape),[self.n_nodes,64])


    # def test_2_decoder(self):
    #     testEncoder.decoder = Decoder()
    #     self.reconstruction = testEncoder.decoder(testEncoder.z)
    #     self.assertEqual(self.reconstruction.shape[0],self.n_nodes*self.n_nodes)

    def test_3_model(self):
        model = HoVGAE(testEncoder.adj_mtx,testEncoder.original_adj_mtx,
                       testEncoder.A_norm,self.n_nodes,
                       )
        model.training_step()
        
if __name__ == "__main__":
    all_gpus = [i.id for i in list(nvsmi.get_available_gpus())]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(all_gpus)
    unittest.main()