import os
import sys
import gc
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
import pandas as pd

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.nn import global_add_pool

from torch_geometric.nn import GCNConv, SAGEConv, GINConv, MLP

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    if hidden_layers >= 2:
        list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers - 2):
            list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Linear(hidden_dim, out_dim))
    else:
        return nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())
    MLP = nn.Sequential(*list_hidden)
    return (MLP)


class Model(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, hidden_layers, grid_size, spline_order, n_targets, dropout,
                 embedding_layer=False):
        super(Model, self).__init__()
        self.n_layers = gnn_layers
        self.embedding_layer = embedding_layer
        lst = list()

        if args.model == 'GAT':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(GATConv(100, hidden_dim, heads=1))
            else:
                lst.append(GATConv(num_features, hidden_dim, heads=1))
            for i in range(gnn_layers - 1):
                lst.append(GATConv(hidden_dim, hidden_dim, heads=1))
        elif args.model == 'GLCN':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(GLCNConv(100, hidden_dim, heads=1))
            else:
                lst.append(GLCNConv(num_features, hidden_dim, heads=1))
            for i in range(gnn_layers - 1):
                lst.append(GLCNConv(hidden_dim, hidden_dim, heads=1))
        elif args.model == 'CFGAT':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(CFGATConv(100, hidden_dim, heads=1))
            else:
                lst.append(CFGATConv(num_features, hidden_dim, heads=1))
            for i in range(gnn_layers - 1):
                lst.append(CFGATConv(hidden_dim, hidden_dim, heads=1))
        elif args.model == 'KAAGAT':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(KAAGATConv(100, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                      spline_order=spline_order))
            else:
                lst.append(KAAGATConv(num_features, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                      spline_order=spline_order))
            for i in range(gnn_layers - 1):
                lst.append(KAAGATConv(hidden_dim, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                      spline_order=spline_order))
        elif args.model == 'KAAGLCN':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(KAAGLCNConv(100, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                       spline_order=spline_order))
            else:
                lst.append(KAAGLCNConv(num_features, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                       spline_order=spline_order))
            for i in range(gnn_layers - 1):
                lst.append(KAAGLCNConv(hidden_dim, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                       spline_order=spline_order))
        elif args.model == 'KAACFGAT':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(KAACFGATConv(100, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                        spline_order=spline_order))
            else:
                lst.append(
                    KAACFGATConv(num_features, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                 spline_order=spline_order))
            for i in range(gnn_layers - 1):
                lst.append(KAACFGATConv(hidden_dim, hidden_dim, heads=1, kan_layers=hidden_layers, grid_size=grid_size,
                                        spline_order=spline_order))
        elif args.model == 'GCN':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(GCNConv(100, hidden_dim))
            else:
                lst.append(GCNConv(num_features, hidden_dim))
            for i in range(gnn_layers - 1):
                lst.append(GCNConv(hidden_dim, hidden_dim))
        elif args.model == 'SAGE':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(SAGEConv(100, hidden_dim))
            else:
                lst.append(SAGEConv(num_features, hidden_dim))
            for i in range(gnn_layers - 1):
                lst.append(SAGEConv(hidden_dim, hidden_dim))
        elif args.model == 'GIN':
            if embedding_layer:
                self.node_emb = nn.Embedding(num_features, 100)
                lst.append(GINConv(nn=MLP([100, hidden_dim, hidden_dim])))
            else:
                lst.append(
                    GINConv(nn=MLP([num_features, hidden_dim, hidden_dim])))
            for i in range(gnn_layers - 1):
                lst.append(GINConv(nn=MLP([hidden_dim, hidden_dim, hidden_dim])))

        self.conv = nn.ModuleList(lst)
        self.mlp = make_mlp(hidden_dim, 64, n_targets, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.embedding_layer:
            x = self.node_emb(x).squeeze()
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.dropout(x)

        x = global_add_pool(x, data.batch)
        x = self.mlp(x)
        return x


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--model', type=str, default='GAT', help='GAT or KAAGAT')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
parser.add_argument('--patience', type=int, default=20, help='Patience for ealry stopping')
parser.add_argument('--n-gnn-layers', type=int, default=4, help='Number of message passing layers')
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
args = parser.parse_args()

# random seed
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')

train_dataset = ZINC('./dataset/ZINC', subset=True, split='train')
val_dataset = ZINC('./dataset/ZINC', subset=True, split='val')
test_dataset = ZINC('./dataset/ZINC', subset=True, split='test')

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

LR = [0.001, 0.01]
HIDDEN_DIM = [32, 64, 128]

if args.model == 'KAAGAT' or args.model == 'KAAGLCN' or args.model == 'KAACFGAT':
    N_LAYERS = [2, 3, 4]
    GRID_SIZE = [1, 2, 4]
    SPLINE_ORDER = [1, 2, 3]
else:
    N_LAYERS = [1]
    GRID_SIZE = [1]
    SPLINE_ORDER = [1]

best_val_mae = float('inf')
for lr in LR:
    for hidden_dim in HIDDEN_DIM:
        for n_layers in N_LAYERS:
            for grid_size in GRID_SIZE:
                for spline_order in SPLINE_ORDER:
                    print('Evaluating the following hyperparameters:')
                    print('lr:', lr, 'hidden_dim:', hidden_dim, 'n_layers:', n_layers, 'grid_size:', grid_size,
                          'spline_order:', spline_order)
                    model = Model(args.n_gnn_layers, 21, hidden_dim, n_layers, grid_size, spline_order, 1, args.dropout,
                                  True).to(device)
                    optimizer = Adam(model.parameters(), lr=lr)


                    def train(epoch):
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            loss = (model(data).squeeze() - data.y).abs().mean()
                            loss.backward()
                            total_loss += loss.item() * data.num_graphs
                            optimizer.step()
                        return total_loss / len(train_loader.dataset)


                    @torch.no_grad()
                    def test(loader):
                        model.eval()

                        total_error = 0
                        for data in loader:
                            data = data.to(device)
                            total_error += (model(data).squeeze() - data.y).abs().sum().item()
                        return total_error / len(loader.dataset)


                    early_stopper = EarlyStopper(patience=args.patience)
                    for epoch in range(1, args.epochs + 1):
                        loss = train(epoch)
                        val_mae = test(val_loader)

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            best_hyperparams = {'lr': lr, 'hidden_dim': hidden_dim, 'n_layers': n_layers,
                                                'grid_size': grid_size, 'spline_order': spline_order}
                            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_mae:.4f}')

                        if early_stopper.early_stop(val_mae):
                            print(f"Stopped at epoch {epoch}")
                            break

print('Best hyperparameters:')
print('lr:', best_hyperparams['lr'])
print('hidden_dim:', best_hyperparams['hidden_dim'])
print('n_layers:', best_hyperparams['n_layers'])
print('grid_size:', best_hyperparams['grid_size'])
print('spline_order:', best_hyperparams['spline_order'])

val_maes = []
test_maes = []
for run in range(5):
    print()
    print(f'Run {run}:')
    print()
    gc.collect()
    model = Model(args.n_gnn_layers, 21, best_hyperparams['hidden_dim'], best_hyperparams['n_layers'],
                  best_hyperparams['grid_size'], best_hyperparams['spline_order'], 1, args.dropout, True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', total_params)
    print()
    optimizer = Adam(model.parameters(), lr=best_hyperparams['lr'])


    def train(epoch):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = (model(data).squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = 0
        for data in loader:
            data = data.to(device)
            total_error += (model(data).squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)


    best_val_mae = test_mae = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        val_mae = test(val_loader)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_mae = test(test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')

        if early_stopper.early_stop(val_mae):
            print(f"Stopped at epoch {epoch}")
            break

    test_maes.append(test_mae)
    val_maes.append(best_val_mae)

test_mae = torch.tensor(test_maes)
print('===========================')
print(f'Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}')

# results
result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'mae', 'std'])

save_dir = os.path.join('..', 'results', 'graph_regression', '{}'.format('ZINC'))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}_KAGNNs.xlsx'.format(args.model))
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': 'ZINC',
                                                   'Model': args.model,
                                                   'mae': float(test_mae.mean()), 'std': float(test_mae.std())}
result_statistic.to_excel(save_path)
print('Mission completes.')
