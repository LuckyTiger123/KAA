# KAA
This is a Pytorch code Implementation of the paper [*KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks*](https://arxiv.org/abs/2501.13456), which is accepted by the ICLR 2025. Specifically, we apply KAN to the scoring functions of attentive GNNs and achieve better theoretical expressiveness and experimental results.

![image-20250219152005843](./PIC/headline.png)

### Installation

We used the following packages under `Python 3.10`.

```
torch 2.4.0
torch_geometric 2.5.3
ogb 1.3.6
numpy 1.26.3
pandas 2.2.2
```

### Backbone Models

In our experiments, we introduced five backbone models: GAT, GLCN, CFGAT, GT, and SAN. Their original scoring functions and the KAA version of the scoring functions are shown in the table below.\text{LeakyReLU}(\cos(\mathbf{W}h_i,\mathbf{W}h_j))

|       | Original Version                                             | KAA Version                                                  |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GAT   | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\text{LeakyReLU}(a^{\top}\cdot[\mathbf{W}h_i \Vert \mathbf{W}h_j])" /><br/></p> | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\text{KAN}([h_i \Vert h_j])" /><br/></p> |
| GLCN  | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\text{ReLU}(a^{\top}\cdot \|h_i-h_j\|)" /><br/></p> | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\text{KAN}(\|h_i-h_j\|)" /><br/></p> |
| CFGAT | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\text{LeakyReLU}(\cos(\mathbf{W}h_i,\mathbf{W}h_j))" /><br/></p> | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\cos(\text{KAN}(h_i),\text{KAN}(h_j))" /><br/></p> |
| GT    | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\frac{1}{\sqrt{d}}(\mathbf{W}_q h_i)^{\top}\cdot \mathbf{W}_k h_j" /><br/></p> | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\frac{1}{\sqrt{d}}\text{KAN}(h_{i})^{\top}\cdot h_{j}" /><br/></p> |
| SAN   | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\frac{1}{\sqrt{d}(\gamma+1)}(\mathbf{W}_q h_i)^{\top}\cdot \mathbf{W}_k h_j" /><br/></p> | <p align="center"><br/>  <img src="https://latex.codecogs.com/svg.latex?\frac{1}{\sqrt{d}(\gamma+1)}\text{KAN}(h_{i})^{\top}\cdot h_{j}" /><br/></p> |

### Experimental Results

#### Node-Level Tasks

The performance of each model on node-level tasks is shown in the table below.

![image-20250219170335105](./PIC/node_level.png)

#### Graph-Level Tasks

The performance of each model on graph-level tasks is shown in the table below.

![image-20250219152931229](./PIC/graph_level.png)

### Reproducibility

To obtain the results of the KAA version of each model in the experiment, you can run the corresponding commands in the `run.sh`.

```
bash run.sh
```

### Citation

You can cite our paper by following bibtex.

```
@inproceedings{Fang2025KAAKA,
  title={KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks},
  author={Taoran Fang and Tianhong Gao and Chunping Wang and Yihao Shang and Wei Chow and Lei Chen and Yang Yang},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

