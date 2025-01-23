import os
import sys
import torch
import argparse
from torch import Tensor
from torch_geometric.datasets import Amazon
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv

parser = argparse.ArgumentParser(description='PyTorch implementation of downstream adaptation.')

parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dimension')
parser.add_argument('--layer_num', type=int, default=2, help='the layer number')
parser.add_argument('--model', type=str, default='KAAGAT')
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=300, help='the epoch number')
parser.add_argument('--lr', type=float, default=0.005, help='the learning rate')
parser.add_argument('--drop_rate', type=float, default=0.1, help='the dropping rate')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--dataset', type=str, default='Computers', help='the evaluation dataset')
parser.add_argument('--train_round', type=int, default=1, help='the train round number')

parser.add_argument('--file_id', type=int, default=0, help='the file id')

args = parser.parse_args()

# random seed
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Amazon(root="./dataset/", name=args.dataset)

data = dataset[0].to(device=device)

# split data
perm = torch.randperm(data.x.size(0))
train_index = perm[:int(data.x.size(0) * 0.1)]
valid_index = perm[int(data.x.size(0) * 0.1):int(data.x.size(0) * 0.2)]
test_index = perm[int(data.x.size(0) * 0.2):]


# from logger import Logger
class Model(torch.nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, output_dim, heads, num_layers, drop_rate, kan_layers, grid_size,
                 spline_order):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if kind == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'GLCN':
            self.convs.append(GLCNConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GLCNConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(GLCNConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'CFGAT':
            self.convs.append(CFGATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(CFGATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(CFGATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'KAAGAT':
            self.convs.append(KAAGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                         spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                               spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAAGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                           spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAAGLCN':
            self.convs.append(
                KAAGLCNConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGLCNConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAAGLCNConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAACFGAT':
            self.convs.append(
                KAACFGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAACFGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                 grid_size=grid_size, spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAACFGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))

        self.dropout = drop_rate

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


model = Model(kind=args.model, input_dim=dataset.num_features, hidden_dim=args.hidden_dim,
              output_dim=dataset.num_classes, heads=args.heads, num_layers=args.layer_num, drop_rate=args.drop_rate,
              kan_layers=2, grid_size=1, spline_order=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_index], data.y[train_index])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[train_index].eq(data.y[train_index]).sum().item())
    train_acc = train_correct / int(train_index.size(0))
    validate_correct = int(pred[valid_index].eq(data.y[valid_index]).sum().item())
    validate_acc = validate_correct / int(valid_index.size(0))
    test_correct = int(pred[test_index].eq(data.y[test_index]).sum().item())
    test_acc = test_correct / int(test_index.size(0))
    return train_acc, validate_acc, test_acc


test_acc_list = []
for round in range(args.train_round):
    print('For the {} round'.format(round))
    best_val_acc = test_acc = 0
    model.reset_parameters()
    for epoch in range(args.epoch_num):
        print('---------------------------------------------------------------')
        print('For the {} epoch'.format(epoch))
        train()
        train_acc, val_acc, current_test_acc = test()
        print(
            'The train acc is {}, the val acc is {}, the test acc is {}.'.format(train_acc, val_acc, current_test_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = current_test_acc
    test_acc_list.append(test_acc)
acc_avg = float(np.average(test_acc_list))
acc_std = float(np.std(test_acc_list))
print('Mission completes.')
print('The avg acc is {}, and the std is {}.'.format(acc_avg, acc_std))
