import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, ReLU
import torch_geometric
from torch_scatter import scatter, scatter_max, scatter_mul, scatter_add
# from torch_geometric.nn import global_mean_pool
from torch_geometric.data import (Data, Dataset, DataLoader)
from torch_geometric.nn.conv.gravnet_conv import GravNetConv
import torch_scatter


def make_fake_model(output_dim: int):
    """
    Makes a function that takes (x, batch), and returns a random tensor
    with shape (x.size(0), output_dim), like the model would
    """
    def fake_model(x: Tensor, batch: Tensor) -> Tensor:
        return torch.from_numpy(np.random.rand(x.size(0), output_dim)).type(torch.float)
    return fake_model


def calc_LV_Lbeta(
    beta: torch.Tensor, cluster_coords: torch.Tensor, # Predicted by model
    cluster_index_per_entry: torch.Tensor, # Truth hit->cluster index
    batch: torch.Tensor,
    qmin: float = .1,
    s_B: float = .1,
    bkg_cluster_index: int = 0 # cluster_index entries with this value are bkg
    ):
    n_hits = beta.size(0)
    cluster_space_dim = cluster_coords.size(1)
    # Transform indices-per-event to indices-per-batch
    # E.g. [ 0, 0, 1, 2, 0, 0, 1] -> [ 0, 0, 1, 2, 3, 3, 4 ]
    cluster_index, n_clusters_per_entry = batch_cluster_indices(cluster_index_per_entry, batch)
    n_clusters = cluster_index.max()+1

    q = beta.arctanh()**2 + qmin

    # Select the maximum charge node per cluster
    q_alpha, index_alpha = scatter_max(q, cluster_index) # max q per cluster
    x_alpha = cluster_coords[index_alpha]
    beta_alpha = beta[index_alpha]

    # print(f'n_hits={n_hits}, n_clusters={n_clusters}, cluster_space_dim={cluster_space_dim}')
    # print(f'x_alpha.size()={x_alpha.size()}')

    # Copy x_alpha by n_hit rows:
    # (n_clusters x cluster_space_dim) --> (n_hits x n_clusters x cluster_space_dim)
    x_alpha_expanded = x_alpha.expand(n_hits, n_clusters, cluster_space_dim)

    # Copy cluster_coord by n_cluster columns:
    # (n_hits x cluster_space_dim) --> (n_hits x n_clusters x cluster_space_dim)
    cluster_coords_expanded = (
        cluster_coords
        .repeat(1,n_clusters).reshape(n_hits, n_clusters, cluster_space_dim)
        )

    # Take the L2 norm; Resulting matrix should be n_hits x n_clusters
    norms = (cluster_coords_expanded - x_alpha_expanded).norm(dim=-1)
    assert norms.size() == (n_hits, n_clusters)

    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index)

    # Copy q_alpha by n_hit rows:
    # (n_clusters) --> (n_hits x n_clusters)
    q_alpha_expanded = q_alpha.expand(n_hits, -1)

    # Potential for hits w.r.t. the cluster they belong to
    V_belonging = M * q_alpha_expanded * norms**2

    # Potential for hits w.r.t. the cluster they DO NOT belong to
    V_notbelonging = 1. - (1-M) * q_alpha_expanded * norms
    V_notbelonging[V_notbelonging < 0.] = 0. # Min of 0

    # Count n_hits per entry in the batch (batch_size)
    _, n_hits_per_entry = torch.unique(batch, return_counts=True)
    # Expand: (batch_size) --> (nhits)
    # e.g. [2, 3, 1] --> [2, 2, 3, 3, 3, 1]
    n_hits_expanded = torch.gather(n_hits_per_entry, 0, batch).type(torch.LongTensor)
    # Alternatively, this should give the same:
    # n_hits_expanded = torch.repeat_interleave(n_hits_per_entry, n_hits_per_entry)

    # Final LV value:
    # (n_hits x 1) * (n_hits x n_clusters) / (n_hits x 1) --> float
    LV = torch.sum(
        q.unsqueeze(-1) * (V_belonging + V_notbelonging) / n_hits_expanded.unsqueeze(-1)
        )
    # Alternatively:
    # LV = torch.sum(q * (V_belonging + V_notbelonging).sum(dim=-1) / n_hits_expanded)

    # ____________________________________
    # Now calculate Lbeta
    # Lbeta also needs the formatted cluster_index and beta_alpha,
    # both of which are known in this scope, so for now it's easier to 
    # calculate it here. Moving it to a dedicated function at some
    # point would be better design.

    is_bkg = cluster_index_per_entry == bkg_cluster_index
    N_B = scatter_add(is_bkg, batch) # (batch_size)
    # Expand (batch_size) -> (n_hits)
    # e.g. [ 3, 2 ], [0, 0, 0, 0, 0, 1, 1, 1, 1] -> [3, 3, 3, 3, 3, 2, 2, 2, 2]
    N_B_expanded = torch.gather(N_B, 0, batch).type(torch.LongTensor)
    bkg_term = s_B * (N_B_expanded*beta)[is_bkg].sum()

    # n_clusters_per_entry: (batch_size)
    # (batch_size) --> (n_clusters)
    # e.g. [3, 2] --> [3, 3, 3, 2, 2]
    n_clusters_expanded = torch.repeat_interleave(n_clusters_per_entry, n_clusters_per_entry)
    assert n_clusters_expanded.size() == (n_clusters,)
    nonbkg_term = ((1.-beta_alpha)/n_clusters_expanded).sum()

    # Final Lbeta
    Lbeta = nonbkg_term + bkg_term

    return LV, Lbeta


def calc_Lp():
    # TODO
    # ------ Property loss ------
    xi = pred_beta[i_cluster > 0].arctanh()
    L_p = 1./xi.sum() * (xi * L(t,p)).sum()
    pass


def batch_cluster_indices(cluster_id: torch.Tensor, batch: torch.Tensor):
    """
    Turns cluster indices per event to an index in the whole batch

    Example:

    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    -->
    offset = torch.LongTensor([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    """
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.zeros(offset_values_nozero.size(0)+1)
    offset_values[1:] = offset_values_nozero
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).type(torch.LongTensor)
    return offset + cluster_id, n_clusters_per_event


def main():
    from gravnet_model import logger, debug, FakeDataset, MyModel, test_scatter_reduce

    batch_size = 4
    train_loader = DataLoader(FakeDataset(100), batch_size=batch_size, shuffle=True)

    # model = MyModel(output_dim=5)
    # print(model)
    # model = make_fake_model(output_dim=7)

    for i, data in enumerate(train_loader):
        print(i, data)

        n_hits = data.x.size(0)

        LV, Lbeta = calc_LV_Lbeta(
            torch.rand(n_hits), # predicted betas
            torch.rand(n_hits, 3), # predicted cluster coordinates, doing 2 dims now
            data.y.type(torch.LongTensor), # truth cluster-index per hit
            data.batch
            )

        print(LV, Lbeta)
        return

if __name__ == '__main__':
    main()