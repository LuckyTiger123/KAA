import os
import sys
import torch
import argparse
from torch_geometric.datasets import Planetoid
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

parser.add_argument('--model', type=str, default='KAACFGAT', help='the used model type')
parser.add_argument('--hidden_dim', type=int, default=8, help='the hidden dimension')
parser.add_argument('--heads', type=int, default=8, help='the head number')

parser.add_argument('--device_num', type=int, default=1, help='the device number')
parser.add_argument('--epoch_num', type=int, default=300, help='the epoch number')
parser.add_argument('--lr', type=float, default=0.005, help='the learning rate')
parser.add_argument('--drop_rate', type=float, default=0.3, help='the dropping rate')

parser.add_argument('--kan_layers', type=int, default=2, help='the kan layer number')
parser.add_argument('--grid_size', type=int, default=1, help='the grid size of kan')
parser.add_argument('--spline_order', type=int, default=1, help='the spline order of kan')

parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--dataset', type=str, default='Cora', help='the test dataset')
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
dataset = Planetoid(root="./dataset/", name=args.dataset,
                    transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)


# Model
class Model(torch.nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, output_dim, heads, drop_rate, kan_layers, grid_size, spline_order):
        super(Model, self).__init__()
        if kind == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate)
        elif kind == 'GLCN':
            self.conv1 = GLCNConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = GLCNConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate)
        elif kind == 'CFGAT':
            self.conv1 = CFGATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = CFGATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate)
        elif kind == 'KAAGAT':
            self.conv1 = KAAGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                    spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAAGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                                    spline_order=spline_order, dropout=drop_rate)
        elif kind == 'KAAGLCN':
            self.conv1 = KAAGLCNConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                     spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAAGLCNConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers,
                                     grid_size=grid_size, spline_order=spline_order, dropout=drop_rate)
        elif kind == 'KAACFGAT':
            self.conv1 = KAACFGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                      spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAACFGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers,
                                      grid_size=grid_size, spline_order=spline_order, dropout=drop_rate)
        else:
            raise NotImplementedError

        self.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


model = Model(kind=args.model, input_dim=dataset.num_features, hidden_dim=args.hidden_dim,
              output_dim=dataset.num_classes, heads=args.heads, drop_rate=args.drop_rate, kan_layers=args.kan_layers,
              grid_size=args.grid_size, spline_order=args.spline_order).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
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
