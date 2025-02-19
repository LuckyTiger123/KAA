# Node-Level Tasks

## Node Classification

### Cora

#### KAAGAT

python node_classification/train.py
--dataset Cora
--model KAAGAT
--hidden_dim 8
--heads 8
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.8

#### KAAGLCN

python node_classification/train.py
--dataset Cora
--model KAAGLCN
--hidden_dim 8
--heads 8
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.8

#### KAACFGAT

python node_classification/train.py
--dataset Cora
--model KAACFGAT
--hidden_dim 8
--heads 8
--kan_layers 2
--grid_size 2
--spline_order 2
--drop_rate 0.8

#### KAAGT

python node_classification/kaa_gt_train.py
--model KAA_GT
--dataset Cora
--num_heads 1
--num_layers 2
--pos_enc_dim 8
--train_round 5
--lr 0.0001
--wd 5e-4
--epoch 500
--hidden_dim 128
--in_feat_dropout 0.2

#### KAASAN

python node_classification/kaa_san_train.py
--model KAA_SAN
--dataset Cora
--num_heads 1
--num_layers 2
--max_freqs 1
--train_round 5
--lr 0.001
--wd 5e-4
--epoch 500
--hidden_dim 200
--in_feat_dropout 0.1

### CiteSeer

#### KAAGAT

python node_classification/train.py
--dataset CiteSeer
--model KAAGAT
--hidden_dim 8
--heads 8
--kan_layers 3
--grid_size 4
--spline_order 3
--drop_rate 0.8

#### KAAGLCN

python node_classification/train.py
--dataset CiteSeer
--model KAAGLCN
--hidden_dim 8
--heads 8
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0.8

#### KAACFGAT

python node_classification/train.py
--dataset CiteSeer
--model KAACFGAT
--hidden_dim 8
--heads 8
--kan_layers 2
--grid_size 8
--spline_order 1
--drop_rate 0.8

#### KAAGT

python node_classification/kaa_gt_train.py
--model KAA_GT
--dataset CiteSeer
--num_heads 1
--num_layers 2
--pos_enc_dim 8
--train_round 5
--lr 0.0001
--wd 5e-4
--epoch 500
--hidden_dim 128
--in_feat_dropout 0.2

#### KAASAN


python node_classification/kaa_san_train.py
--model KAA_SAN
--dataset CiteSeer
--num_heads 1
--num_layers 2
--max_freqs 1
--train_round 5
--lr 0.001
--wd 5e-4
--epoch 500
--hidden_dim 1200
--in_feat_dropout 0.1

### PubMed

#### KAAGAT

python node_classification/train.py
--dataset PubMed
--model KAAGAT
--hidden_dim 8
--heads 8
--kan_layers 3
--grid_size 4
--spline_order 2
--drop_rate 0.3

#### KAAGLCN

python node_classification/train.py
--dataset PubMed
--model KAAGLCN
--hidden_dim 8
--heads 8
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0

#### KAACFGAT

python node_classification/train.py
--dataset PubMed
--model KAACFGAT
--hidden_dim 8
--heads 8
--kan_layers 2
--grid_size 2
--spline_order 2
--drop_rate 0

#### KAAGT

python node_classification/kaa_gt_train.py
--model KAA_GT
--dataset PubMed
--num_heads 1
--num_layers 2
--pos_enc_dim 8
--train_round 5
--lr 0.0001
--wd 5e-4
--epoch 500
--hidden_dim 128
--in_feat_dropout 0.2

#### KAASAN

python node_classification/kaa_san_train.py
--model KAA_SAN
--dataset PubMed
--num_heads 1
--num_layers 2
--max_freqs 1
--train_round 5
--lr 0.001
--wd 5e-4
--epoch 500
--hidden_dim 144
--in_feat_dropout 0.1

### ogbn-arxiv

#### KAAGAT

python node_classification/train_arxiv.py
--model KAAGAT
--hidden_channels 32
--num_layers 3
--dropout 0 

#### KAAGLCN

python node_classification/train_arxiv.py
--model KAAGLCN
--hidden_channels 32
--num_layers 3
--dropout 0

#### KAACFGAT

python node_classification/train_arxiv.py
--model KAACFGAT
--hidden_channels 32
--num_layers 3
--dropout 0

### Amazon-Photo

#### KAAGAT

python node_classification/train_amazon.py
--dataset Photo
--model KAAGAT
--hidden_dim 64
--heads 2
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.5

#### KAAGLCN

python node_classification/train_amazon.py
--dataset Photo
--model KAAGLCN
--hidden_dim 64
--heads 1
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAACFGAT

python node_classification/train_amazon.py
--dataset Photo
--model KAACFGAT
--hidden_dim 64
--heads 1
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0.3

#### KAAGT

python node_classification/kaa_gt_train.py
--model KAA_GT
--dataset Photo
--num_heads 1
--num_layers 2
--pos_enc_dim 8
--train_round 5
--lr 0.0001
--wd 5e-4
--epoch 500
--hidden_dim 128
--in_feat_dropout 0.2

#### KAASAN

python node_classification/kaa_san_train.py
--model KAA_SAN
--dataset Photo
--num_heads 1
--num_layers 2
--max_freqs 1
--train_round 5
--lr 0.001
--wd 5e-4
--epoch 500
--hidden_dim 200
--in_feat_dropout 0.1

### Amazon-Computers

#### KAAGAT

python node_classification/train_amazon.py
--dataset Computers
--model KAAGAT
--hidden_dim 64
--heads 1
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.5

#### KAAGLCN

python node_classification/train_amazon.py
--dataset Computers
--model KAAGLCN
--hidden_dim 32
--heads 1
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAACFGAT

python node_classification/train_amazon.py
--dataset Computers
--model KAACFGAT
--hidden_dim 64
--heads 1
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0.1

#### KAAGT

python node_classification/kaa_gt_train.py
--model KAA_GT
--dataset Computers
--num_heads 1
--num_layers 2
--pos_enc_dim 8
--train_round 5
--lr 0.0001
--wd 5e-4
--epoch 500
--hidden_dim 128
--in_feat_dropout 0.2

#### KAASAN

python node_classification/kaa_san_train.py
--model KAA_SAN
--dataset Computers
--num_heads 1
--num_layers 2
--max_freqs 1
--train_round 5
--lr 0.001
--wd 5e-4
--epoch 500
--hidden_dim 200
--in_feat_dropout 0.1


## Link Prediction

### Cora

#### KAAGAT

python link_prediction/train.py
--dataset Cora
--model KAAGAT
--hidden_dim 128
--heads 1
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0.5

#### KAAGLCN

python link_prediction/train.py
--dataset Cora
--model KAAGLCN
--hidden_dim 128
--heads 1
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.5

#### KAACFGAT

python link_prediction/train.py
--dataset Cora
--model KAACFGAT
--hidden_dim 128
--heads 2
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.5

#### KAAGT

python link_prediction/gt_train.py
--dataset Cora
--hidden_dim 144
--model KAA_GT
--heads 1
--epoch_num 1000
--lr 0.0001
--train_round 5
--pos_enc_dim 8
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_train.py
--dataset Cora
--hidden_dim 144
--model KAA_SAN
--heads 1
--lr 0.0001
--train_round 5
--max_freqs 2
--wd 5e-4
--layers 2
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### CiteSeer

#### KAAGAT

python link_prediction/train.py
--dataset Cora
--model KAAGAT
--hidden_dim 128
--heads 2
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0.5

#### KAAGLCN

python link_prediction/train.py
--dataset Cora
--model KAAGLCN
--hidden_dim 128
--heads 2
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0.5

#### KAACFGAT

python link_prediction/train.py
--dataset Cora
--model KAACFGAT
--hidden_dim 128
--heads 2
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0.5

#### KAAGT

python link_prediction/gt_train.py
--dataset CiteSeer
--hidden_dim 144
--model KAA_GT
--heads 1
--epoch_num 1000
--lr 0.0001
--train_round 5
--pos_enc_dim 8
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_train.py
--dataset CiteSeer
--hidden_dim 64
--model KAA_SAN
--heads 1
--lr 0.0001
--train_round 5
--max_freqs 2
--wd 5e-4
--layers 2
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### PubMed

#### KAAGAT

python link_prediction/train.py
--dataset Cora
--model KAAGAT
--hidden_dim 128
--heads 2
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0.1

#### KAAGLCN

python link_prediction/train.py
--dataset Cora
--model KAAGLCN
--hidden_dim 128
--heads 2
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0

#### KAACFGAT

python link_prediction/train.py
--dataset Cora
--model KAACFGAT
--hidden_dim 128
--heads 2
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0.1

#### KAAGT

python link_prediction/gt_train.py
--dataset PubMed
--hidden_dim 144
--model KAA_GT
--heads 1
--epoch_num 1000
--lr 0.0001
--train_round 5
--pos_enc_dim 8
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_train.py
--dataset PubMed
--hidden_dim 144
--model KAA_SAN
--heads 1
--lr 0.0001
--train_round 5
--max_freqs 2
--wd 5e-4
--layers 2
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

# Graph-Level Tasks

## Graph Classification

### PPI

#### KAAGAT

python graph_classification/train_ppi.py
--model KAAGAT
--hidden_dim 128
--heads 4
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAAGLCN

python graph_classification/train_ppi.py
--model KAAGLCN
--hidden_dim 128
--heads 4
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAACFGAT

python graph_classification/train_ppi.py
--model KAACFGAT
--hidden_dim 128
--heads 4
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAAGT

python graph_classification/gt_ppi_train.py
--dataset PPI
--batch_size 1
--model KAA_GT
--heads 1
--hidden_dim 128
--lr 0.001
--epoch_num 200
--train_round 5
--pos_enc_dim 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_ppi_train.py
--dataset PPI
--batch_size 1
--model KAA_SAN
--heads 1
--hidden_dim 128
--lr 0.001
--epoch_num 200
--train_round 5
--max_freqs 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### MUTAG

#### KAAGAT

python graph_classification/train_tu.py
--dataset MUTAG
--model KAAGAT
--hidden_dim 32
--heads 1
--num_layers 2
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0.1

#### KAAGLCN

python graph_classification/train_tu.py
--dataset MUTAG
--model KAAGLCN
--hidden_dim 32
--heads 1
--num_layers 2
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAACFGAT

python graph_classification/train_tu.py
--dataset MUTAG
--model KAACFGAT
--hidden_dim 32
--heads 1
--num_layers 2
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0.1

#### KAAGT

python graph_classification/gt_tu_train.py
--dataset MUTAG
--batch_size 1
--model KAA_GT
--heads 1
--hidden_dim 16
--lr 0.0001
--epoch_num 100
--train_round 5
--pos_enc_dim 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_tu_train.py
python san_tu_train.py
--dataset MUTAG
--batch_size 64
--model KAA_SAN
--heads 1
--hidden_dim 64
--lr 0.0001
--epoch_num 100
--train_round 5
--max_freqs 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### ENZYMES

#### KAAGAT

python graph_classification/train_tu.py
--dataset ENZYMES
--model KAAGAT
--hidden_dim 32
--heads 1
--num_layers 3
--kan_layers 2
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAAGLCN

python graph_classification/train_tu.py
--dataset ENZYMES
--model KAAGLCN
--hidden_dim 32
--heads 1
--num_layers 2
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAACFGAT

python graph_classification/train_tu.py
--dataset ENZYMES
--model KAACFGAT
--hidden_dim 32
--heads 2
--num_layers 2
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0

#### KAAGT

python graph_classification/gt_tu_train.py
--dataset ENZYMES
--batch_size 1
--model KAA_GT
--heads 1
--hidden_dim 16
--lr 0.0001
--epoch_num 100
--train_round 5
--pos_enc_dim 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_tu_train.py
--dataset ENZYMES
--batch_size 64
--model KAA_SAN
--heads 1
--hidden_dim 32
--lr 0.0001
--epoch_num 100
--train_round 5
--max_freqs 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### PROTEINS

#### KAAGAT

python graph_classification/train_tu.py
--dataset PROTEINS
--model KAAGAT
--hidden_dim 32
--heads 2
--num_layers 2
--kan_layers 2
--grid_size 1
--spline_order 2
--drop_rate 0

#### KAAGLCN

python graph_classification/train_tu.py
--dataset PROTEINS
--model KAAGLCN
--hidden_dim 32
--heads 1
--num_layers 2
--kan_layers 3
--grid_size 1
--spline_order 2
--drop_rate 0.1

#### KAACFGAT

python graph_classification/train_tu.py
--dataset PROTEINS
--model KAACFGAT
--hidden_dim 32
--heads 4
--num_layers 3
--kan_layers 3
--grid_size 1
--spline_order 1
--drop_rate 0

#### KAAGT

python graph_classification/gt_tu_train.py
--dataset PROTEINS
--batch_size 1
--model KAA_GT
--heads 1
--hidden_dim 64
--lr 0.0001
--epoch_num 100
--train_round 5
--pos_enc_dim 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python link_prediction/san_tu_train.py
--dataset PROTEINS
--batch_size 64
--model KAA_SAN
--heads 1
--hidden_dim 32
--lr 0.0001
--epoch_num 100
--train_round 5
--max_freqs 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

## Graph Regression

### ZINC

#### KAAGAT

python graph_regression/train_zinc.py
--model KAAGAT

#### KAAGLCN

python graph_regression/train_zinc.py
--model KAAGLCN

#### KAACFGAT

python graph_regression/train_zinc.py
--model KAACFGAT

#### KAAGT

python graph_regression/gt_zinc_train.py
--batch-size 128
--epochs 300
--model KAA_GT
--patience 20
--pos_enc_dim 2
--hidden_dim 64
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python graph_regression/san_zinc_train.py
--batch-size 128
--epochs 300
--model KAA_SAN
--patience 20
--max_freqs 8
--num_heads 1
--hidden_dim 64
--layers 2
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

### QM9

#### KAAGAT

python graph_regression/train_qm9.py
--model KAAGAT

#### KAAGLCN

python graph_regression/train_qm9.py
--model KAAGLCN

#### KAACFGAT

python graph_regression/train_qm9.py
--model KAACFGAT

#### KAAGT

python graph_regression/gt_qm9_train.py
--batch-size 128
--epochs 300
--model KAA_GT
--patience 20
--pos_enc_dim 2
--hidden_dim 64
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

#### KAASAN

python graph_regression/san_qm9_train.py
--batch-size 128
--epochs 300
--model KAA_SAN
--patience 20
--max_freqs 1
--hidden_dim 64
--heads 1
--layers 4
--in_feat_dropout 0
--dropout 0
--spline_order 2
--grid_size 1
--hidden_layers 2

