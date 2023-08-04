
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from absl import logging, app,flags
from torch_geometric.nn import VGAE
from model import HOVariationalGraphEncoder,VariationalGraphEncoder
from datasets import InputData
from utils import *
import scipy as sp
import logging
import pickle as pkl
import networkx as nx
import sys
import os
from absl import app,flags
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter 

flags.DEFINE_string("path_to_graph","","Path to input graph object")
flags.DEFINE_string("path_to_node_features","","Path to node features stored in pickle")
flags.DEFINE_integer("number_of_epoch",1000,"how many epoch to run. Default 500")
flags.DEFINE_float("alpha",0.2,"The probability of restart during random walk. Must be between 0.0 and 1.0. Default is 0.2")
flags.DEFINE_enum("model","HOVGAE",
                  ["HOVGAE","VGAE"],"which model to use either HOVGAE or VGAE")
FLAGS = flags.FLAGS
logger = logging.getLogger(__file__)

def prepare_cora_data(transform):
    """
    loaded kipf original datasets as a positive control
    just to test if my codes work. not used later on 
    """
    data_dir_path = './input_data/kipf_data'
    dataset = 'cora'
    adj,features,graph = load_kipf_data(dataset)
    input_dim = features.shape[1]
    features = torch.from_numpy(features)
    row,col = adj.nonzero()
    edge_index =  torch.from_numpy(np.array([row,col]))
    input_data = InputData(root=data_dir_path,transform=transform,
                            x=features,edge_index=edge_index)
    
    return input_data,input_dim

def prepare_influenza_data(transform):
    """
    Load the influenza-human and human-human interaction network
    """
    graph_path = FLAGS.path_to_graph
    node_feature_path = FLAGS.path_to_node_features
    input_graph = nx.read_gml(graph_path)
    node_features = pkl.load(open(node_feature_path,"rb"))
    input_dim = node_features.shape[1]
    adj = create_adj_mtx(input_graph)
    node_features = torch.from_numpy(node_features)
    row,col = adj.nonzero()
    edge_index =  torch.from_numpy(np.array([row,col]))
    influenza_data = InputData(root=graph_path,transform=transform,
                            x=node_features,edge_index=edge_index)
    return influenza_data,input_dim

def train(model,optimiser,train_data):
    model.train()
    optimiser.zero_grad()
    z = model.encode(train_data.x.double(), train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimiser.step()
    return float(loss)

@torch.no_grad()
def test_and_eval(data,model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

def save_checkpoint(epoch,model,optimiser,loss,filename):
    torch.save(
        {
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimiser.state_dict(),
            'loss':loss
        },filename
    )

def main(args):
    tensorboard_writer = SummaryWriter()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0")
    transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),])
    input_data,input_dim = prepare_influenza_data(transform)
    train_data, val_data, test_data = input_data[0]

    # below have to modify the dtype of these tensors
    # otherwise report errors
    train_data.x = train_data.x.double()
    val_data.pos_edge_label_index = val_data.pos_edge_label_index.to(torch.int64)
    train_data.pos_edge_label_index = train_data.pos_edge_label_index.to(torch.int64)
    test_data.pos_edge_label_index = test_data.pos_edge_label_index.to(torch.int64)
    train_data.edge_index=train_data.edge_index.to(torch.int64)

    if FLAGS.model == "HOVGAE":
        model = VGAE(HOVariationalGraphEncoder(input_dim,edge_index=train_data.edge_index,
                                           num_nodes=train_data.x.shape[0],alpha=FLAGS.alpha)).to(device)
    else:
        model = VGAE(VariationalGraphEncoder(input_dim)).to(device)
    
    tensorboard_writer.add_graph(model,[train_data.x,train_data.edge_index])
    tensorboard_writer.close()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimiser,gamma=0.9)
    auc,ap=0,0
    for i in range(FLAGS.number_of_epoch):
        loss = train(model,optimiser,train_data)
        if i %20 ==0:
            auc,ap = test_and_eval(val_data,model)
            tensorboard_writer.add_scalar("validation_auc",auc,i+1)
            tensorboard_writer.add_scalar("validation_ap",auc,i+1)
        print(f"epoch {i+1}, loss:{loss} AUC: {auc},AP{ap}")
        # save_checkpoint(epoch=i+1,model=model,optimiser=optimiser,
        #                 loss=loss,filename=f"checkpoint_{i+1}.pt")

        scheduler.step()
        tensorboard_writer.add_scalar("training_loss",loss,i+1)
    test_auc,test_ap = test_and_eval(test_data,model)
    print(f"Training finished. Test AUC: {test_auc} and test AP: {test_ap}")
    save_checkpoint(epoch=FLAGS.number_of_epoch,model=model,optimiser=optimiser,
                        loss=loss,filename=f"checkpoint_final.pt")

if __name__ =="__main__":
    app.run(main)