import os
import pandas as pd
import numpy as np
import argparse
import dgl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon

from model.SAN_KAN.module import laplace_decomp
from model.SAN_KAN.SAN_nodeLPE import SAN_NodeLPE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="KAA_SAN", help="select your model")
parser.add_argument("--dataset", type=str, default="Cora", help="select your dataset")
parser.add_argument("--num_heads", type=int, default=1, help="number of attention heads")
parser.add_argument("--num_layers", type=int, default=2, help="number of transformer layers")
parser.add_argument("--max_freqs", type=int, default=1, help="freq in lap decomp")
parser.add_argument("--train_round", type=int, default=5, help="round of training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--epoch", type=int, default=500, help="number of epochs")
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument('--in_feat_dropout', type=float, default=0)
args = parser.parse_args([])

def pyg_to_dgl(data):
    """Converts a PyTorch Geometric (PyG) data object to a Deep Graph Library (DGL) graph object."""
    edge_index = data.edge_index
    src = edge_index[0].to('cpu')
    dst = edge_index[1].to('cpu')
    
    g = dgl.graph((src, dst))
    
    g.ndata['feat'] = data.x.to('cpu')
    g.ndata['label'] = data.y.to('cpu')
    g.ndata['train_mask'] = data.train_mask.to('cpu')
    g.ndata['val_mask'] = data.val_mask.to('cpu')
    g.ndata['test_mask'] = data.test_mask.to('cpu')
    
    return g

def pyg_to_dgl_amazon(data):
    """Converts a PyTorch Geometric (PyG) data object to a Deep Graph Library (DGL) graph object for Amazon data."""
    edge_index = data.edge_index
    src = edge_index[0].to('cpu')
    dst = edge_index[1].to('cpu')
    
    g = dgl.graph((src, dst))

    perm = torch.randperm(data.x.size(0))
    train_index = perm[:int(data.x.size(0) * 0.1)]
    valid_index = perm[int(data.x.size(0) * 0.1):int(data.x.size(0) * 0.2)]
    test_index = perm[int(data.x.size(0) * 0.2):]

    train_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.x.size(0), dtype=torch.bool)

    train_mask[train_index] = 1
    val_mask[valid_index] = 1
    test_mask[test_index] = 1

    g.ndata['feat'] = data.x.to('cpu')
    g.ndata['label'] = data.y.to('cpu')
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    return g

save_dir = os.path.join('..', 'result', 'node_classification', '{}'.format(args.dataset))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '{}.xlsx'.format(args.model))

if args.dataset == "Cora":
    dataset = Planetoid(root = "./dataset/", name = "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    graph = pyg_to_dgl(data)
    graph = laplace_decomp(graph, args.max_freqs)
elif args.dataset == "CiteSeer":
    dataset = Planetoid(root = "./dataset/", name = "CiteSeer", transform=T.NormalizeFeatures())
    data = dataset[0]
    graph = pyg_to_dgl(data)
    graph = laplace_decomp(graph, args.max_freqs)
elif args.dataset == "PubMed":
    dataset = Planetoid(root = "./dataset/", name = "PubMed", transform=T.NormalizeFeatures())
    data = dataset[0]
    graph = pyg_to_dgl(data)
    graph = laplace_decomp(graph, args.max_freqs)
elif args.dataset == "Computers":
    dataset = Amazon(root = "./dataset/", name = "Computers", transform=T.NormalizeFeatures())
    data = dataset[0]
    graph = pyg_to_dgl_amazon(data)
    graph = laplace_decomp(graph, args.max_freqs)
elif args.dataset == "Photo":
    dataset = Amazon(root = "./dataset/", name = "Photo", transform=T.NormalizeFeatures())
    data = dataset[0]
    graph = pyg_to_dgl_amazon(data)
    graph = laplace_decomp(graph, args.max_freqs)

num_feats = int(graph.ndata['feat'].shape[1])
num_classes = len(torch.unique(graph.ndata['label']))

net_params = {
    'kind': args.model,
    'in_dim': num_feats,
    'GT_hidden_dim': args.hidden_dim,
    'GT_out_dim': args.hidden_dim,
    'n_classes': num_classes,
    'GT_n_heads': args.num_heads,
    'in_feat_dropout': args.in_feat_dropout,
    'GT_layers': args.num_layers,
    'max_freqs': args.max_freqs,
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'full_graph': False,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'gamma': 1,
    'LPE_dim': 8,
    'LPE_n_heads': 2,
    'LPE_layers': 2,
}

spline_order_list = [2]
grid_size_list = [1]
hidden_layers_list = [2]
dropout_list = [0.6]

graph = graph.to(net_params['device'])
features = graph.ndata['feat'].to(net_params['device'])
labels = graph.ndata['label'].to(net_params['device'])
eigen_vec = graph.ndata['EigVecs'].to(net_params['device'])
eigen_val = graph.ndata['EigVals'].to(net_params['device'])
train_mask = graph.ndata['train_mask'].to(net_params['device'])
val_mask = graph.ndata['val_mask'].to(net_params['device'])
test_mask = graph.ndata['test_mask'].to(net_params['device'])


def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph, features, eigen_vec, eigen_val)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    print(f'Training loss: {loss.item()}')

@torch.no_grad()
def test():
    model.eval()
    out = model(graph, features, eigen_vec, eigen_val)
    _, pred = out.max(dim=1)
    
    train_correct = pred[train_mask].eq(labels[train_mask]).sum().item()
    train_acc = train_correct / train_mask.sum().item()
    
    val_correct = pred[val_mask].eq(labels[val_mask]).sum().item()
    val_acc = val_correct / val_mask.sum().item()
    
    test_correct = pred[test_mask].eq(labels[test_mask]).sum().item()
    test_acc = test_correct / test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

result_statistic = pd.DataFrame(
    columns=['Dataset', 'Model', 'hidden_dim', 'heads', 'LPE_dim', 'LPE_n_heads', 'LPE_layers', 'kan_layers', 'grid_size', 'spline_order', 'max_freqs', 'in_feature_drop', 'drop_rate', 'acc', 'std'])
    
for drop_out in dropout_list:
    # for heads in heads_list:
        # for drop_out in dropout_list:
            for kan_layers in hidden_layers_list:
                for grid_size in grid_size_list:
                    for spline_order in spline_order_list:
                        net_params['dropout'] = drop_out
                        net_params['hidden_layers'] = kan_layers
                        net_params['grid_size'] = grid_size
                        net_params['spline_order'] = spline_order
                        test_acc_list = []
                        print('----------------------------------------------------------')
                        print(
                            'For the parameter setting: drop_rate {}, kan_layers {}, grid_size {}, spline_order {}'.format(
                                drop_out, kan_layers, grid_size, spline_order))
                        for rounds in range(args.train_round):
                            model = SAN_NodeLPE(net_params).to(net_params['device'])
                            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
                            best_val_acc = 0
                            test_acc = 0
                            print('----------------------------------------------------------')
                            print('For the {}-th round'.format(rounds))
                            for epoch in range(args.epoch):
                                print('For the {} epoch'.format(epoch))
                                train()
                                train_acc, val_acc, current_test_acc = test()
                                print(
                                    'The train acc is {}, the val acc is {}, the test acc is {}.'.format(train_acc,
                                                                                                         val_acc,
                                                                                                         current_test_acc))
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    test_acc = current_test_acc
                            test_acc_list.append(test_acc)
                        acc_avg = float(np.average(test_acc_list))
                        acc_std = float(np.std(test_acc_list))
                        print('The avg acc is {}, and the std is {}.'.format(acc_avg, acc_std))
                        result_statistic.loc[result_statistic.shape[0]] = {'Dataset': args.dataset, 'Model': args.model,
                                                'hidden_dim': net_params['GT_hidden_dim'], 'heads': args.num_heads,
                                                'LPE_dim': net_params['LPE_dim'],
                                                'LPE_n_heads': net_params['LPE_n_heads'],
                                                'LPE_layers': net_params['LPE_layers'],
                                                'kan_layers': kan_layers,
                                                'grid_size': grid_size,
                                                'spline_order': spline_order,
                                                'max_freqs': args.max_freqs,
                                                'in_feature_drop': net_params['in_feat_dropout'],
                                                'drop_rate': net_params['dropout'],
                                                'acc': acc_avg,
                                                'std': acc_std}
                        result_statistic.to_excel(save_path)
print('Mission completes.')