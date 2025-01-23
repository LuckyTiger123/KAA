import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.nn import MLP, global_add_pool

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv

parser = argparse.ArgumentParser()
# Dataset: MUTAG, ENZYMES, PROTEINS
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', type=str, default='KAAGLCN', help='the used model type')
parser.add_argument('--heads', type=int, default=1, help='the head number')
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=100, help='the epoch number')
parser.add_argument('--drop_rate', type=float, default=0, help='the dropping rate')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--train_round', type=int, default=2, help='the train round number')
args = parser.parse_args()

# random seed
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

path = './dataset/TU'
dataset = TUDataset(path, name=args.dataset).shuffle()

train_loader = DataLoader(dataset[:0.9], args.batch_size, shuffle=True)
test_loader = DataLoader(dataset[0.9:], args.batch_size)


# Model
class Model(torch.nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, output_dim, heads, num_layers, drop_rate, kan_layers, grid_size,
                 spline_order):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        if kind == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=drop_rate))
        elif kind == 'GLCN':
            self.convs.append(GLCNConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(GLCNConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
            self.convs.append(GLCNConv(hidden_dim * heads, hidden_dim, heads=1, dropout=drop_rate))
        elif kind == 'CFGAT':
            self.convs.append(CFGATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(CFGATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
            self.convs.append(CFGATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=drop_rate))
        elif kind == 'KAAGAT':
            self.convs.append(KAAGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                         spline_order=spline_order, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                               spline_order=spline_order, dropout=drop_rate))
            self.convs.append(
                KAAGATConv(hidden_dim * heads, hidden_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                           spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAAGLCN':
            self.convs.append(
                KAAGLCNConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGLCNConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                spline_order=spline_order, dropout=drop_rate))
            self.convs.append(
                KAAGLCNConv(hidden_dim * heads, hidden_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAACFGAT':
            self.convs.append(
                KAACFGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAACFGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                 grid_size=grid_size, spline_order=spline_order, dropout=drop_rate))
            self.convs.append(
                KAACFGATConv(hidden_dim * heads, hidden_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))

        self.mlp = MLP([hidden_dim, hidden_dim, output_dim], norm=None, dropout=[drop_rate, drop_rate])

        self.dropout = drop_rate
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


model = Model(kind=args.model, input_dim=dataset.num_features, hidden_dim=args.hidden_channels,
              output_dim=dataset.num_classes, heads=args.heads, num_layers=args.num_layers, drop_rate=args.drop_rate,
              kan_layers=2, grid_size=1, spline_order=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, args.epoch_num + 1):
    start = time.time()
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
