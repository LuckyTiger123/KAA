import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dimension')
parser.add_argument('--model', type=str, default='GLCN', help='the used model type')
parser.add_argument('--heads', type=int, default=4, help='the head number')
parser.add_argument('--device_num', type=int, default=0, help='the device number')
parser.add_argument('--epoch_num', type=int, default=100, help='the epoch number')
parser.add_argument('--drop_rate', type=float, default=0, help='the dropping rate')
parser.add_argument('--seed', type=int, default=1, help='the random seed')
parser.add_argument('--train_round', type=int, default=2, help='the train round number')
args = parser.parse_args()

path = './dataset/PPI'

train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# random seed
random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Model
class Model(torch.nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, output_dim, heads, drop_rate, kan_layers, grid_size, spline_order):
        super(Model, self).__init__()
        if kind == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=6, dropout=drop_rate, concat=False)
        elif kind == 'GLCN':
            self.conv1 = GLCNConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = GLCNConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv3 = GLCNConv(hidden_dim * heads, output_dim, heads=6, dropout=drop_rate, concat=False)
        elif kind == 'CFGAT':
            self.conv1 = CFGATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv2 = CFGATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate)
            self.conv3 = CFGATConv(hidden_dim * heads, output_dim, heads=6, dropout=drop_rate, concat=False)
        elif kind == 'KAAGAT':
            self.conv1 = KAAGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                    spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAAGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                    grid_size=grid_size, spline_order=spline_order, dropout=drop_rate)
            self.conv3 = KAAGATConv(hidden_dim * heads, output_dim, heads=6, kan_layers=kan_layers, grid_size=grid_size,
                                    spline_order=spline_order, dropout=drop_rate, concat=False)
        elif kind == 'KAAGLCN':
            self.conv1 = KAAGLCNConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                     spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAAGLCNConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                     grid_size=grid_size, spline_order=spline_order, dropout=drop_rate)
            self.conv3 = KAAGLCNConv(hidden_dim * heads, output_dim, heads=6, kan_layers=kan_layers,
                                     grid_size=grid_size, spline_order=spline_order, dropout=drop_rate, concat=False)
        elif kind == 'KAACFGAT':
            self.conv1 = KAACFGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                      spline_order=spline_order, dropout=drop_rate)
            self.conv2 = KAACFGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                      grid_size=grid_size, spline_order=spline_order, dropout=drop_rate)
            self.conv3 = KAACFGATConv(hidden_dim * heads, output_dim, heads=6, kan_layers=kan_layers,
                                      grid_size=grid_size, spline_order=spline_order, dropout=drop_rate, concat=False)
        else:
            raise NotImplementedError

        self.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=args.drop_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()


device = torch.device('cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu')
model = Model(kind=args.model, input_dim=train_dataset.num_features, hidden_dim=args.hidden_dim,
              output_dim=train_dataset.num_classes, heads=args.heads, drop_rate=args.drop_rate,
              kan_layers=2, grid_size=1, spline_order=1).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


times = []
for epoch in range(1, 101):
    start = time.time()
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
