import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
from model.GT_KAN.graph_transformer_layer import GraphTransformerLayer
from model.GT_KAN.mlp_readout_layer import MLPReadout
from torch_geometric.nn import MLP, global_add_pool

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        spline_order = net_params['spline_order']
        grid_size = net_params['grid_size']
        hidden_layers = net_params['hidden_layers']
        kind = net_params['kind']

        # self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(kind, hidden_dim, hidden_dim, num_heads, spline_order, grid_size, hidden_layers,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(kind, hidden_dim, out_dim, num_heads, spline_order, grid_size, hidden_layers, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        # self.mlp = MLP([hidden_dim, hidden_dim, n_classes], norm = None, dropout=[dropout, dropout])

        # self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        
        h = self.embedding_h(h)

        # method1: addition
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc

        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            # h = conv(g, h)
            h = conv(g, h).relu()
        # h = global_add_pool(h, batch)

        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)

        