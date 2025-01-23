import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from model.GAT import GATConv
from model.GLCN import GLCNConv
from model.CFGAT import CFGATConv
from model.KAA_GAT import KAAGATConv
from model.KAA_GLCN import KAAGLCNConv
from model.KAA_CFGAT import KAACFGATConv


# from logger import Logger
class Model(torch.nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, output_dim, heads, num_layers, drop_rate, kan_layers, grid_size,
                 spline_order):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if kind == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'GLCN':
            self.convs.append(GLCNConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GLCNConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(GLCNConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'CFGAT':
            self.convs.append(CFGATConv(input_dim, hidden_dim, heads=heads, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(CFGATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(CFGATConv(hidden_dim * heads, output_dim, heads=1, dropout=drop_rate))
        elif kind == 'KAAGAT':
            self.convs.append(KAAGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                         spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                               spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAAGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                           spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAAGLCN':
            self.convs.append(
                KAAGLCNConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAAGLCNConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                                spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAAGLCNConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                            spline_order=spline_order, dropout=drop_rate))
        elif kind == 'KAACFGAT':
            self.convs.append(
                KAACFGATConv(input_dim, hidden_dim, heads=heads, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(
                    KAACFGATConv(hidden_dim * heads, hidden_dim, heads=heads, kan_layers=kan_layers,
                                 grid_size=grid_size, spline_order=spline_order, dropout=drop_rate))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            self.convs.append(
                KAACFGATConv(hidden_dim * heads, output_dim, heads=1, kan_layers=kan_layers, grid_size=grid_size,
                             spline_order=spline_order, dropout=drop_rate))

        self.dropout = drop_rate

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
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


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--model', type=str, default='GAT')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/',
                                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))

    data = dataset[0]
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = Model(args.model, data.num_features, args.hidden_channels,
                  dataset.num_classes, 1, args.num_layers, args.dropout, 2, 1, 1).to(device=device)

    evaluator = Evaluator(name='ogbn-arxiv')

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')


if __name__ == "__main__":
    main()
