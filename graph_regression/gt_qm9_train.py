import os
import sys
import gc
import argparse
import torch
from torch.optim import Adam
import pandas as pd
import dgl
import time

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.GT_KAN.graph_transformer_net import GraphTransformerNet
from model.GT_KAN.laplace_pos_enc import laplacian_positional_encoding


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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--model', type=str, default='GT', help='model to test')
parser.add_argument('--patience', type=int, default=20, help='Patience for ealry stopping')
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--pos_enc_dim', type=int, default=2, help='laplacian positional encoding dimension')
parser.add_argument('--hidden_dim', type=int, default=64)
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

dataset = QM9('./dataset/QM9')
dataset.data.y = dataset.data.y[:, 0:12]
dataset = dataset.shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

tenpercent = int(len(dataset) * 0.1)
test_dataset = dataset[:tenpercent].shuffle()
val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
train_dataset = dataset[2 * tenpercent:].shuffle()


def apply_lap_pos_enc(dataset, pos_enc_dim):
    """Applies Laplacian positional encoding to the entire dataset."""
    processed_data_list = []
    
    for data in dataset:
        temp_graph = pyg_to_dgl(data)
        temp_graph = laplacian_positional_encoding(temp_graph, pos_enc_dim)
        data.lap_pos_enc = temp_graph.ndata['lap_pos_enc']
        processed_data_list.append(data)
    
    return processed_data_list

train_data_processed = apply_lap_pos_enc(train_dataset, args.pos_enc_dim)
val_data_processed = apply_lap_pos_enc(val_dataset, args.pos_enc_dim)
test_data_processed = apply_lap_pos_enc(test_dataset, args.pos_enc_dim)

train_loader = DataLoader(train_data_processed, args.batch_size, shuffle=True)
val_loader = DataLoader(val_data_processed, args.batch_size, shuffle=False)
test_loader = DataLoader(test_data_processed, args.batch_size, shuffle=False)

net_params= {
    'kind': args.model,
    'in_dim': train_dataset.num_features,
    'hidden_dim': args.hidden_dim,
    'out_dim': args.hidden_dim,
    'n_heads': 1,
    'in_feat_dropout': args.in_feat_dropout,
    'dropout': args.dropout,
    'pos_enc_dim': args.pos_enc_dim,
    'L': args.layers,
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'device': device,
    'lap_pos_enc': True,
    'wl_pos_enc': False,
    'spline_order': args.spline_order,
    'grid_size': args.grid_size,
    'hidden_layers': args.hidden_layers,
    'batch_size': args.batch_size,
    'n_classes': 12,
    'edge_feat': False,
    'num_atom_type': 1,
    'num_bond_type':torch.unique(train_dataset.edge_attr, dim=0).size(0)
}

LR = [0.001]
HIDDEN_DIM = [64, 128]
N_LAYERS = [2]
GRID_SIZE = [1]
SPLINE_ORDER = [2]

best_val_mae = float('inf')
for lr in LR:
    for hidden_dim in HIDDEN_DIM:
        for n_layers in N_LAYERS:
            for grid_size in GRID_SIZE:
                for spline_order in SPLINE_ORDER:
                    net_params['hidden_dim'] = hidden_dim
                    net_params['out_dim'] = hidden_dim
                    net_params['L'] = n_layers
                    net_params['grid_size'] = grid_size
                    net_params['spline_order'] = spline_order

                    print('Evaluating the following hyperparameters:')
                    print('lr:', lr, 'hidden_dim:', hidden_dim, 'n_layers:', n_layers, 'grid_size:', grid_size,
                          'spline_order:', spline_order)
                    model = GraphTransformerNet(net_params).to(net_params['device'])
                    optimizer = Adam(model.parameters(), lr=lr)
                    loss_function = torch.nn.L1Loss()


                    def train(epoch):
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            graph = pyg_to_dgl(data)
                            graph = graph.to(device)
                            data = data.to(device)
                            optimizer.zero_grad()
                            out = model(graph, graph.ndata['feat'], None, data.batch, data.lap_pos_enc)
                            loss = loss_function(out.squeeze(), data.y)
                            loss.backward()
                            total_loss += loss.item() * data.num_graphs
                            optimizer.step()
                        return total_loss / len(train_loader.dataset)


                    @torch.no_grad()
                    def test(loader):
                        model.eval()

                        total_error = torch.zeros([1, 12]).to(device)
                        for data in loader:
                            graph = pyg_to_dgl(data)
                            graph = graph.to(device)
                            data = data.to(device)
                            out = model(graph, graph.ndata['feat'], None, data.batch, data.lap_pos_enc)
                            total_error += ((data.y * std - out * std).abs() / std).sum(dim=0)
                        
                        total_error = total_error / len(loader.dataset)

                        return total_error.mean().item()


                    early_stopper = EarlyStopper(patience=args.patience)
                    for epoch in range(1, args.epochs + 1):
                        start = time.time()
                        loss = train(epoch)
                        val_mae = test(val_loader)

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            best_hyperparams = {'lr': lr, 'hidden_dim': hidden_dim, 'n_layers': n_layers,
                                                'grid_size': grid_size, 'spline_order': spline_order}
                            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_mae:.4f}')
                        
                        end = time.time()
                        print('Time: {:.4f}s'.format(end - start))
                        
                        if early_stopper.early_stop(val_mae):
                            print(f"Stopped at epoch {epoch}")
                            break

print('Best hyperparameters:')
print('lr:', best_hyperparams['lr'])
print('hidden_dim:', best_hyperparams['hidden_dim'])
print('n_layers:', best_hyperparams['n_layers'])
print('grid_size:', best_hyperparams['grid_size'])
print('spline_order:', best_hyperparams['spline_order'])

net_params['hidden_dim'] = best_hyperparams['hidden_dim']
net_params['out_dim'] = best_hyperparams['hidden_dim']
net_params['L'] = best_hyperparams['n_layers']
net_params['grid_size'] = best_hyperparams['grid_size']
net_params['spline_order'] = best_hyperparams['spline_order']

val_maes = []
test_maes = []
for run in range(5):
    print()
    print(f'Run {run}:')
    print()
    gc.collect()
    model = GraphTransformerNet(net_params).to(net_params['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', total_params)
    print()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.L1Loss()


    def train(epoch):
        model.train()

        total_loss = 0
        for data in train_loader:
            graph = pyg_to_dgl(data)
            graph = graph.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            out = model(graph, graph.ndata['feat'], None, data.batch, data.lap_pos_enc)
            loss = loss_function(out.squeeze(), data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = torch.zeros([1, 12]).to(device)
        for data in loader:
            graph = pyg_to_dgl(data)
            graph = graph.to(device)
            data = data.to(device)
            out = model(graph, graph.ndata['feat'], None, data.batch, data.lap_pos_enc)
            total_error += ((data.y * std - out * std).abs() / std).sum(dim=0)
        
        total_error = total_error / len(loader.dataset)

        return total_error.mean().item()


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

save_dir = os.path.join('..', 'results', 'graph_regression', '{}'.format('qm9'))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}_KAGNNs.xlsx'.format(args.model))
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': 'qm9',
                                                   'Model': args.model,
                                                   'mae': float(test_mae.mean()), 'std': float(test_mae.std())}
result_statistic.to_excel(save_path)
print('Mission completes.')