import os
import sys
import torch
import argparse
import numpy as np
import dgl
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.SAN_KAN.SAN_nodeLPE import SAN_NodeLPE
from model.SAN_KAN.laplace_decomp import laplace_decomp

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dimension')
parser.add_argument('--model', type=str, default='KAA_SAN', help='the used model type')
parser.add_argument('--heads', type=int, default=1, help='the head number')
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=1000, help='the epoch number')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--dataset', type=str, default='PubMed', help='the test dataset')
parser.add_argument('--train_round', type=int, default=5, help='the train round number')
parser.add_argument('--max_freqs', type=int, default=2, help='max freqs')
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--in_feat_dropout', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--spline_order', type=int, default=0)
parser.add_argument('--grid_size', type=int, default=0)
parser.add_argument('--hidden_layers', type=int, default=0)

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

# random generate train, validate, test mask
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def pyg_to_dgl(data):
    """Converts a PyTorch Geometric (PyG) data object to a Deep Graph Library (DGL) graph object."""
    edge_index = data.edge_index
    src = edge_index[0].to('cpu')
    dst = edge_index[1].to('cpu')
    
    g = dgl.graph((src, dst))
    g.ndata['feat'] = data.x[:g.num_nodes(), :].to('cpu')
    
    num_existing_nodes = g.num_nodes()
    num_total_nodes = data.x.size(0)
    
    if num_total_nodes > num_existing_nodes:
        num_new_nodes = num_total_nodes - num_existing_nodes
        g.add_nodes(num_new_nodes)
        g.ndata['feat'][num_existing_nodes:] = data.x[num_existing_nodes:, :].to('cpu')
    return g

# collect dataset
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
dataset = Planetoid(root="./dataset/", name=args.dataset,
                    transform=transform)
train_data, val_data, test_data = dataset[0]

train_graph = pyg_to_dgl(train_data)
train_graph = laplace_decomp(train_graph, args.max_freqs)
train_graph = train_graph.to(device)

val_graph = pyg_to_dgl(val_data)
val_graph = laplace_decomp(val_graph, args.max_freqs)
val_graph = val_graph.to(device)

test_graph = pyg_to_dgl(test_data)
test_graph = laplace_decomp(test_graph, args.max_freqs)
test_graph = test_graph.to(device)

net_params = {
    'kind': args.model,
    'in_dim': dataset.num_features,
    'GT_hidden_dim': args.hidden_dim,
    'GT_out_dim': args.hidden_dim,
    'GT_n_heads': args.heads,
    'in_feat_dropout': args.in_feat_dropout,
    'dropout': args.dropout,
    'GT_layers': args.layers,
    'max_freqs': args.max_freqs,
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'full_graph': False,
    'device': device,
    'gamma': 1,
    'LPE_dim': 8,
    'LPE_n_heads': 2,
    'LPE_layers': 2,
    'spline_order': args.spline_order,
    'grid_size': args.grid_size,
    'hidden_layers': args.hidden_layers,
}

def decode(z, edge_label_index):
    return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

def train():
    """Training"""
    model.train()
    optimizer.zero_grad()
    z = model(train_graph, train_graph.ndata['feat'], train_graph.ndata['EigVecs'], train_graph.ndata['EigVals'])

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()  # Perform the backward pass
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    """Testing"""
    model.eval()
    z_train = model(train_graph, train_graph.ndata['feat'], train_graph.ndata['EigVecs'], train_graph.ndata['EigVals'])
    z_test = model(test_graph, test_graph.ndata['feat'], test_graph.ndata['EigVecs'], test_graph.ndata['EigVals'])
    z_val = model(val_graph, val_graph.ndata['feat'], val_graph.ndata['EigVecs'], val_graph.ndata['EigVals'])
    out_test = decode(z_test, test_data.edge_label_index).view(-1).sigmoid()
    out_val = decode(z_val, val_data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(val_data.edge_label.cpu().numpy(), out_val.cpu().numpy()), roc_auc_score(test_data.edge_label.cpu().numpy(), out_test.cpu().numpy())

test_auc_list = []
for round in range(args.train_round):
    print('For the {} round'.format(round))
    best_val_auc = test_auc = 0
    model = SAN_NodeLPE(net_params).to(net_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epoch_num):
        loss = train()
        val_auc, tmp_test_auc = test()
        print('---------------------------------------------------------------------------')
        print('For the {} epoch, the train loss is {}, the val auc is {}, the test auc is {}.'.format(epoch,
                                                                                                      loss,
                                                                                                      val_auc,
                                                                                                      tmp_test_auc))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
    test_auc_list.append(test_auc)
auc_avg = float(np.average(test_auc_list))
auc_std = float(np.std(test_auc_list))
print('Mission completes.')
print('The avg auc is {}, and the std is {}.'.format(auc_avg, auc_std))