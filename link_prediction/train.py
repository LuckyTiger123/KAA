import os
import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dimension')
parser.add_argument('--model', type=str, default='KAAGAT', help='the used model type')
parser.add_argument('--heads', type=int, default=1, help='the head number')
parser.add_argument('--device_num', type=int, default=1, help='the device number')
parser.add_argument('--epoch_num', type=int, default=300, help='the epoch number')
parser.add_argument('--lr', type=float, default=0.01, help='the learning rate')
parser.add_argument('--drop_rate', type=float, default=0, help='the dropping rate')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--dataset', type=str, default='Cora', help='the test dataset')
parser.add_argument('--train_round', type=int, default=1, help='the train round number')
args = parser.parse_args()

# random generate train, validate, test mask
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')
# collect dataset
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
dataset = Planetoid(root="./dataset/", name=args.dataset,
                    transform=transform)
train_data, val_data, test_data = dataset[0]


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

        self.drop_rate = drop_rate

        self.reset_parameters()

    def encode(self, x, edge_index):
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


model = Model(kind=args.model, input_dim=dataset.num_features, hidden_dim=args.hidden_dim, output_dim=64,
              heads=args.heads, drop_rate=args.drop_rate, kan_layers=2, grid_size=1, spline_order=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.BCEWithLogitsLoss()

test_auc_list = []
for round in range(args.train_round):
    print('For the {} round'.format(round))
    best_val_auc = test_auc = 0
    model.reset_parameters()
    for epoch in range(args.epoch_num):
        loss = train()
        val_auc = test(val_data)
        tmp_test_auc = test(test_data)
        print('---------------------------------------------------------------------------')
        print('For the {} epoch, the train loss is {}, the val auc is {}, the test auc is {}.'.format(epoch,
                                                                                                      loss,
                                                                                                      val_auc,
                                                                                                      tmp_test_auc))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
    test_auc_list.append(test_auc)
auc_avg = float(np.average(test_auc_list))
auc_std = float(np.std(test_auc_list))
print('Mission completes.')
print('The avg auc is {}, and the std is {}.'.format(auc_avg, auc_std))
