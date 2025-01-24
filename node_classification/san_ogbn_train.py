import os
import pandas as pd
import numpy as np
import argparse
import dgl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


from model.SAN_KAN.laplace_decomp import laplace_decomp
from model.SAN_KAN.SAN_nodeLPE import SAN_NodeLPE

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="KAA_SAN", help="select your model")
parser.add_argument("--dataset", type=str, default="ogbn", help="select your dataset")
parser.add_argument("--num_heads", type=int, default=1, help="number of attention heads")
parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
parser.add_argument("--max_freqs", type=int, default=1, help="freq in lap decomp")
parser.add_argument("--train_round", type=int, default=5, help="round of training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--epoch", type=int, default=500, help="number of epochs")
parser.add_argument('--device_num', type=int, default=0, help='the device number')
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

save_dir = os.path.join('..', 'result', 'node_classification', '{}'.format(args.dataset))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}.xlsx'.format(args.model))


if args.dataset == "ogbn":
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/',
                                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    data = dataset[0]
    graph = pyg_to_dgl_ogbn(data)
    graph = laplace_decomp(graph, args.max_freqs)

num_feats = int(graph.ndata['feat'].shape[1])
num_classes = len(torch.unique(graph.ndata['label']))

net_params = {
    'kind': args.model,
    'in_dim': num_feats,
    'GT_hidden_dim': args.hidden_dim,
    'GT_out_dim': args.out_dim,
    'n_classes': num_classes,
    'GT_n_heads': args.num_heads,
    'dropout': args.dropout,
    'in_feat_dropout': args.in_feat_dropout,
    'GT_layers': args.num_layers,
    'max_freqs': args.max_freqs,
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'full_graph': False,
    'device': torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu'),
    'gamma': 1,
    'LPE_dim': 8,
    'LPE_n_heads': 2,
    'LPE_layers': 2,
    'spline_order': args.spline_order,
    'grid_size': args.grid_size,
    'hidden_layers': args.hidden_layers
}

graph = graph.to(net_params['device'])
features = graph.ndata['feat'].to(net_params['device'])
labels = graph.ndata['label'].to(net_params['device'])
eigen_vec = graph.ndata['EigVecs'].to(net_params['device'])
eigen_val = graph.ndata['EigVals'].to(net_params['device'])

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(net_params['device'])
targets = data.y.squeeze(1).to(net_params['device'])

evaluator = Evaluator(name='ogbn-arxiv')

def train(model, data, train_idx, optimizer, graph, features, eigen_vec, eigen_val, targets):
    model.train()

    optimizer.zero_grad()
    out = model(graph, features, eigen_vec, eigen_val)[train_idx]
    loss = F.nll_loss(out, targets[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator, graph, features, eigen_vec, eigen_val):
    model.eval()

    out = model(graph, features, eigen_vec, eigen_val)
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
    model = SAN_NodeLPE(net_params).to(net_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    test_acc = 0
    for epoch in range(1, 1 + args.epoch):
        loss = train(model, data, train_idx, optimizer, graph, features, eigen_vec, eigen_val, targets)
        result = test(model, data, split_idx, evaluator, graph, features, eigen_vec, eigen_val)

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
    print(f"Acc this round: is {best_test_acc}")
acc_avg = float(np.average(test_acc_list))
acc_std = float(np.std(test_acc_list))
print(f"acc mean is {acc_avg}, acc std is {acc_std}")

result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'acc', 'std'])
result_statistic.loc[result_statistic.shape[0]] = {'Dataset': args.dataset, 'Model': args.model,
                                                'acc': acc_avg,
                                                'std': acc_std}
result_statistic.to_excel(save_path)

