import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import dgl

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.SAN_KAN.SAN_nodeLPE import SAN_NodeLPE
from model.SAN_KAN.laplace_decomp import laplace_decomp

parser = argparse.ArgumentParser()
# Dataset: MUTAG, ENZYMES, PROTEINS
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model', type=str, default='KAA_SAN', help='the used model type')
parser.add_argument('--heads', type=int, default=1, help='the head number')
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=100, help='the epoch number')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--train_round', type=int, default=5, help='the train round number')
parser.add_argument('--max_freqs', type=int, default=1, help='max freqs')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--in_feat_dropout', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--spline_order', type=int, default=0)
parser.add_argument('--grid_size', type=int, default=0)
parser.add_argument('--hidden_layers', type=int, default=0)
args = parser.parse_args()

# random seed
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

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

path = './dataset/TU/'
dataset = TUDataset(path, name=args.dataset).shuffle()

train_loader = DataLoader(dataset[:0.8], args.batch_size, shuffle=True)
val_loader = DataLoader(dataset[0.8:0.9], args.batch_size)
test_loader = DataLoader(dataset[0.9:], args.batch_size)

net_params = {
    'kind': args.model,
    'in_dim': dataset.num_features,
    'GT_hidden_dim': args.hidden_dim,
    'GT_out_dim': args.hidden_dim,
    'GT_n_heads': args.heads,
    'dropout': args.dropout,
    'in_feat_dropout': args.in_feat_dropout,
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
    'hidden_layers': args.hidden_layers
}

def train():
    """Training"""
    model.train()
    total_loss = 0
    for data in train_loader:
        graph = pyg_to_dgl(data)
        graph = laplace_decomp(graph, args.max_freqs)
        graph = graph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(graph, graph.ndata['feat'], data.batch, graph.ndata['EigVecs'], graph.ndata['EigVals'])
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    """Testing"""
    model.eval()
    total_correct = 0
    for data in loader:
        graph = pyg_to_dgl(data)
        graph = laplace_decomp(graph, args.max_freqs)
        graph = graph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(graph, graph.ndata['feat'], data.batch, graph.ndata['EigVecs'], graph.ndata['EigVals'])
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

test_acc_list = []
for round in range(args.train_round):
    best_val_acc = 0
    best_test_acc = 0
    model = SAN_NodeLPE(net_params).to(net_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('----------------------------------------------------------')
    print('For the {}-th round'.format(round))
    for epoch in range(1, args.epoch_num + 1):
        loss = train()
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        log(Epoch=epoch, Loss=loss, Test=test_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    test_acc_list.append(best_test_acc)
    print(f"Acc this round: is {best_test_acc}")
acc_avg = float(np.average(test_acc_list))
acc_std = float(np.std(test_acc_list))
print(f"Final results: the accuracy is {acc_avg}, the std is {acc_std}")

save_dir = os.path.join('..', 'result', 'graph_classification', '{}'.format(args.dataset))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}.xlsx'.format(args.model))

result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'acc', 'std'])
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': args.dataset, 'Model': args.model,
                                                'acc': acc_avg,
                                                'std': acc_std}
result_statistic.to_excel(save_path)