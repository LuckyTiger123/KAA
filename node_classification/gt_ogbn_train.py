import os
import argparse
import numpy as np
import pandas as pd
import dgl
import random
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model.GT_KAN.laplace_pos_enc import laplacian_positional_encoding, laplacian_positional_encoding_fast
from model.GT_KAN.graph_transformer_net import GraphTransformerNet

SEED = 20
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
dgl.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="KAA_GT", help="select your model")
parser.add_argument("--dataset", type=str, default="ogbn", help="select your dataset")
parser.add_argument("--num_heads", type=int, default=1, help="number of attention heads")
parser.add_argument("--num_layers", type=int, default=2, help="number of transformer layers")
parser.add_argument("--pos_enc_dim", type=int, default=8, help="dimensionality of positional encoding")
parser.add_argument("--train_round", type=int, default=5, help="round of training")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--out_dim", type=int, default=128)
parser.add_argument("--in_feat_dropout", type=float, default=0)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--spline_order", type=int, default=0)
parser.add_argument("--grid_size", type=int, default=0)
parser.add_argument("--hidden_layers", type=int, default=0)
args = parser.parse_args([])

def pyg_to_dgl_ogbn(data):
    """Converts a PyTorch Geometric (PyG) data object to a Deep Graph Library (DGL) graph object."""
    spt = data.adj_t
    row = spt.storage.row()
    col = spt.storage.col()
    
    g = dgl.graph((row, col))

    g.ndata['feat'] = data.x
    g.ndata['label'] = data.y
    
    return g

save_dir = os.path.join('..', 'result', 'node_classification', 'ogbn')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}.xlsx'.format(args.model))

if args.dataset == "ogbn":
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/',
                                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    data = dataset[0]
    graph = pyg_to_dgl_ogbn(data)
    graph = laplacian_positional_encoding_fast(graph, args.pos_enc_dim)

num_feats = int(graph.ndata['feat'].shape[1])
num_classes = len(torch.unique(graph.ndata['label']))

net_params = {
    'kind': args.model,
    'in_dim': num_feats,
    'hidden_dim': args.hidden_dim,
    'out_dim': args.out_dim,
    'n_classes': num_classes,
    'n_heads': args.num_heads,
    'in_feat_dropout': args.in_feat_dropout,
    'dropout': args.dropout,
    'L': args.num_layers,
    'pos_enc_dim': args.pos_enc_dim,
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lap_pos_enc': True,
    'wl_pos_enc': False,
    'spline_order': args.spline_order,
    'grid_size': args.grid_size,
    'hidden_layers': args.hidden_layers
}

graph = graph.to(net_params['device'])
features = graph.ndata['feat'].to(net_params['device'])
labels = graph.ndata['label'].to(net_params['device'])
lap_pos_enc = graph.ndata['lap_pos_enc'].to(net_params['device'])


split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(net_params['device'])
targets = data.y.squeeze(1).to(net_params['device'])

evaluator = Evaluator(name='ogbn-arxiv')

def train(model, data, train_idx, optimizer, graph, features, lap_pos_enc, targets):
    model.train()

    optimizer.zero_grad()
    out = model(graph, features, None, lap_pos_enc)[train_idx]
    loss = F.nll_loss(out, targets[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator, graph, features, lap_pos_enc):
    model.eval()

    out = model(graph, features, None, lap_pos_enc)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

test_acc_list = []
for run in range(args.train_round):
    model = GraphTransformerNet(net_params).to(net_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, 1 + args.epoch):
        loss = train(model, data, train_idx, optimizer, graph, features, lap_pos_enc, targets)
        result = test(model, data, split_idx, evaluator, graph, features, lap_pos_enc)

        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
    test_acc_list.append(best_test_acc)
    print(f"Acc this round is {best_test_acc}")
acc_avg = float(np.average(test_acc_list))
acc_std = float(np.std(test_acc_list))
print(f"Acc mean is {acc_avg}, std is {acc_std}")

result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'hidden_dim', 'heads', 'pos_enc_dim', 'in_feature_drop', 'drop_rate', 'acc', 'std'])
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': args.dataset, 'Model': args.model,
                                                'hidden_dim': net_params['hidden_dim'], 'heads': args.num_heads,
                                                'pos_enc_dim': args.pos_enc_dim,
                                                'in_feature_drop': net_params['in_feat_dropout'],
                                                'drop_rate': net_params['dropout'],
                                                'acc': acc_avg,
                                                'std': acc_std}
result_statistic.to_excel(save_path)