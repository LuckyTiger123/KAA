import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import dgl
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.SAN_KAN.SAN_nodeLPE import SAN_NodeLPE
from model.SAN_KAN.module import laplace_decomp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PPI')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model', type=str, default='KAA_SAN', help='the used model type')
parser.add_argument('--heads', type=int, default=1, help='the head number')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=200, help='the epoch number')
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

path = './dataset/PPI'
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

def apply_pos_enc(dataset, max_freqs):
    """Applies Laplacian positional encoding to the entire dataset."""
    processed_data_list = []
    
    for data in dataset:
        temp_graph = pyg_to_dgl(data)
        temp_graph = laplace_decomp(temp_graph, max_freqs)
        data.eigvecs = temp_graph.ndata['EigVecs']
        data.eigvalues = temp_graph.ndata['EigVals']
        processed_data_list.append(data)
    
    return processed_data_list

train_data_processed = apply_pos_enc(train_dataset, args.max_freqs)
val_data_processed = apply_pos_enc(val_dataset, args.max_freqs)
test_data_processed = apply_pos_enc(test_dataset, args.max_freqs)

train_loader = DataLoader(train_data_processed, 1, shuffle=True)
val_loader = DataLoader(val_data_processed, 2, shuffle=False)
test_loader = DataLoader(test_data_processed, 2, shuffle=False)


net_params = {
    'kind': args.model,
    'in_dim': train_dataset.num_features,
    'GT_hidden_dim': args.hidden_dim,
    'GT_out_dim': args.hidden_dim,
    'n_classes': train_dataset.num_classes,
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

loss_op = torch.nn.BCEWithLogitsLoss()

def train():
    """Training"""
    model.train()
    total_loss = 0
    for data in train_loader:
        graph = pyg_to_dgl(data)
        graph = graph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(graph, graph.ndata['feat'], data.eigvecs, data.eigvalues)
        loss = loss_op(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    "Testing"
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        graph = pyg_to_dgl(data)
        graph = graph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(graph, graph.ndata['feat'], data.eigvecs, data.eigvalues)
        preds.append((out > 0).float().cpu())
    
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

test_f1_list = []
for round in range(args.train_round):
    best_val_f1 = 0
    best_test_f1 = 0
    model = SAN_NodeLPE(net_params).to(net_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('----------------------------------------------------------')
    print('For the {}-th round'.format(round))
    for epoch in range(1, args.epoch_num + 1):
        loss = train()
        val_f1 = test(val_loader)
        test_f1 = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
            f'Test: {test_f1:.4f}')
        if val_f1 > best_val_f1:
            best_val_f1= val_f1
            best_test_f1 = test_f1
    test_f1_list.append(best_test_f1)
    print(f"Acc this round: is {best_test_f1}")
f1_avg = float(np.average(test_f1_list))
f1_std = float(np.std(test_f1_list))
print(f"Final results: the f1 is {f1_avg}, the std is {f1_std}")

save_dir = os.path.join('..', 'result', 'graph_classification', '{}'.format(args.dataset))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}.xlsx'.format(args.model))

result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'f1', 'std'])
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': args.dataset, 'Model': args.model,
                                                'f1': f1_avg,
                                                'std': f1_std}
result_statistic.to_excel(save_path)