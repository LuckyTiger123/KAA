from scipy import sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
import torch

def laplacian_positional_encoding(g, pos_enc_dim, concat = False):
    """
    Graph positional encoding using Laplacian eigenvectors
    """
    # Compute the adjacency matrix and convert it to a SciPy sparse matrix
    A_torch = g.adjacency_matrix().to_dense()
    A = sp.csr_matrix(A_torch.numpy())
    
    # Degree matrix
    N = sp.diags(np.clip(g.in_degrees().numpy(), 1, None) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Compute eigenvalues and eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # in ascending order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # Store the positional encodings in the node features of the graph
    if concat == False:
        g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    else:
        lap_pos_enc = torch.from_numpy(EigVec).float()
        g.ndata['feat'] = torch.cat([g.ndata['feat'], lap_pos_enc], dim=-1) #using concat here

    return g

def laplacian_positional_encoding_fast(g, pos_enc_dim, concat = False):
    """
    Graph positional encoding using Laplacian eigenvectors with efficient computation.
    """

    src, dst = g.edges()
    src, dst = src.cpu().numpy(), dst.cpu().numpy()
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(g.number_of_nodes(), g.number_of_nodes()))
    adj = adj.tocsr()
    
    degrees = np.clip(g.in_degrees().cpu().numpy(), 1, None) ** -0.5
    D = sp.diags(degrees)
    
    L = sp.eye(g.number_of_nodes()) - D @ adj @ D
    L = L.asfptype()

    k = pos_enc_dim
    EigVal, EigVec = eigsh(L, k=k, which='SM')

    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    lap_pos_enc = torch.from_numpy(EigVec).float()

    if concat == True:
        g.ndata['feat'] = torch.cat([g.ndata['feat'], lap_pos_enc], dim=-1) #using concat here
    else:
        g.ndata['lap_pos_enc'] = lap_pos_enc

    return g