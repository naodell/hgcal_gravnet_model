import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add
from torch_geometric.data import DataLoader


def make_fake_model(output_dim: int):
    """
    Makes a function that takes (x, batch), and returns a random tensor
    with shape (x.size(0), output_dim), like the model would
    """
    def fake_model(x: Tensor, batch: Tensor) -> Tensor:
        return torch.from_numpy(np.random.rand(x.size(0), output_dim)).type(torch.float)
    return fake_model

def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    assert not torch.isnan(x).any()

DEBUG = False
def debug(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

def calc_LV_Lbeta(
    beta: torch.Tensor, cluster_coords: torch.Tensor, # Predicted by model
    cluster_index_per_event: torch.Tensor, # Truth hit->cluster index
    batch: torch.Tensor,
    qmin: float = .1,
    s_B: float = .1,
    bkg_cluster_index: int = 0 # cluster_index entries with this value are bkg/noise
    ):
    # First do some device assertions
    device = beta.device
    assert all(t.device == device for t in [beta, cluster_coords, cluster_index_per_event, batch])
    assert_no_nans(beta)
    assert_no_nans(cluster_coords)
    assert_no_nans(batch)
    assert_no_nans(cluster_index_per_event)

    # Some fixed event quantities
    n_hits = beta.size(0)
    cluster_space_dim = cluster_coords.size(1)

    # Transform indices-per-event to indices-per-batch
    # E.g. [ 0, 0, 1, 2, 0, 0, 1] -> [ 0, 0, 1, 2, 3, 3, 4 ]
    # or for non-ordered:
    # truth clus index: [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 2]
    # batch:            [2, 1, 1, 2, 1, 1, 2, 0, 0, 0, 1, 0, 0]
    # output:           [6, 3, 4, 5, 3, 4, 5, 1, 0, 1, 4, 0, 2]
    cluster_index, n_clusters_per_event = batch_cluster_indices(cluster_index_per_event, batch)
    assert cluster_index.device == device
    assert n_clusters_per_event.device == device

    # Sort all hits by this properly-offset cluster_index
    order = cluster_index.argsort()
    beta = beta[order]
    cluster_coords = cluster_coords[order]
    cluster_index_per_event = cluster_index_per_event[order]
    batch = batch[order]
    cluster_index = cluster_index[order]
    n_clusters = cluster_index.max()+1

    debug(f'\n\nIn calc_LV_Lbeta; n_hits={n_hits}, n_clusters={n_clusters}')

    q = beta.arctanh()**2 + qmin
    assert_no_nans(q)
    assert q.device == device

    # Select the maximum charge node per cluster
    q_alpha, index_alpha = scatter_max(q, cluster_index) # max q per cluster
    assert index_alpha.size() == (n_clusters,)
    assert_no_nans(q_alpha)
    assert q_alpha.device == device
    assert index_alpha.device == device

    debug('beta:', beta)
    debug('q:', q)
    debug('q_alpha:', q_alpha)
    debug('index_alpha:', index_alpha)
    debug('cluster_index:', cluster_index)
    debug('cluster_index_per_event:', cluster_index_per_event)

    x_alpha = cluster_coords[index_alpha]
    beta_alpha = beta[index_alpha]

    assert x_alpha.device == device
    assert beta_alpha.device == device
    assert_no_nans(x_alpha)
    assert_no_nans(beta_alpha)

    # debug(f'n_hits={n_hits}, n_clusters={n_clusters}, cluster_space_dim={cluster_space_dim}')
    # debug(f'x_alpha.size()={x_alpha.size()}')

    # Copy x_alpha by n_hit rows:
    # (n_clusters x cluster_space_dim) --> (n_hits x n_clusters x cluster_space_dim)
    x_alpha_expanded = x_alpha.expand(n_hits, n_clusters, cluster_space_dim)
    assert_no_nans(x_alpha_expanded)
    assert x_alpha_expanded.device == device

    # Copy cluster_coord by n_cluster columns:
    # (n_hits x cluster_space_dim) --> (n_hits x n_clusters x cluster_space_dim)
    cluster_coords_expanded = (
        cluster_coords
        .repeat(1,n_clusters).reshape(n_hits, n_clusters, cluster_space_dim)
        )
    assert cluster_coords_expanded.device == device
    assert_no_nans(cluster_coords_expanded)

    # Take the L2 norm; Resulting matrix should be n_hits x n_clusters
    norms = (cluster_coords_expanded - x_alpha_expanded).norm(dim=-1)
    assert norms.size() == (n_hits, n_clusters)
    assert_no_nans(norms)
    assert norms.device == device

    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index)
    assert M.device == device

    # Copy q_alpha by n_hit rows:
    # (n_clusters) --> (n_hits x n_clusters)
    q_alpha_expanded = q_alpha.expand(n_hits, -1)
    assert q_alpha_expanded.device == device

    # Potential for hits w.r.t. the cluster they belong to
    V_belonging = M * q_alpha_expanded * norms**2
    assert V_belonging.device == device

    # Potential for hits w.r.t. the cluster they DO NOT belong to
    V_notbelonging = 1. - (1-M) * q_alpha_expanded * norms
    V_notbelonging[V_notbelonging < 0.] = 0. # Min of 0
    assert V_notbelonging.device == device

    # Count n_hits per entry in the batch (batch_size)
    _, n_hits_per_event = torch.unique(batch, return_counts=True)
    assert n_hits_per_event.device == device
    # Expand: (batch_size) --> (nhits)
    # e.g. [2, 3, 1] --> [2, 2, 3, 3, 3, 1]
    n_hits_expanded = torch.gather(n_hits_per_event, 0, batch).long()
    assert n_hits_expanded.device == device
    # Alternatively, this should give the same:
    # n_hits_expanded = torch.repeat_interleave(n_hits_per_event, n_hits_per_event)

    # Final LV value:
    # (n_hits x 1) * (n_hits x n_clusters) / (n_hits x 1) --> float
    LV = torch.sum(
        q.unsqueeze(-1) * (V_belonging + V_notbelonging) / n_hits_expanded.unsqueeze(-1)
        )
    assert LV.device == device
    assert_no_nans(LV)
    # Alternatively:
    # LV = torch.sum(q * (V_belonging + V_notbelonging).sum(dim=-1) / n_hits_expanded)

    # ____________________________________
    # Now calculate Lbeta
    # Lbeta also needs the formatted cluster_index and beta_alpha,
    # both of which are known in this scope, so for now it's easier to 
    # calculate it here. Moving it to a dedicated function at some
    # point would be better design.

    is_bkg = cluster_index_per_event == bkg_cluster_index
    assert is_bkg.device == device
    N_B = scatter_add(is_bkg, batch) # (batch_size)
    assert N_B.device == device
    # Expand (batch_size) -> (n_hits)
    # e.g. [ 3, 2 ], [0, 0, 0, 0, 0, 1, 1, 1, 1] -> [3, 3, 3, 3, 3, 2, 2, 2, 2]
    N_B_expanded = torch.gather(N_B, 0, batch).long()
    assert N_B_expanded.device == device
    bkg_term = s_B * (N_B_expanded*beta)[is_bkg].sum()
    assert bkg_term.device == device

    # n_clusters_per_event: (batch_size)
    # (batch_size) --> (n_clusters)
    # e.g. [3, 2] --> [3, 3, 3, 2, 2]
    n_clusters_expanded = torch.repeat_interleave(n_clusters_per_event, n_clusters_per_event)
    assert n_clusters_expanded.size() == (n_clusters,)
    nonbkg_term = ((1.-beta_alpha)/n_clusters_expanded).sum()

    # Final Lbeta
    Lbeta = nonbkg_term + bkg_term
    debug(f'LV={LV:.6f}, Lbeta={Lbeta:.6f} (nonbkg={nonbkg_term:.4f}, bkg={bkg_term:.4f})')
    return LV, Lbeta


def softclip(array, start_clip_value):
    array /= start_clip_value
    array = torch.where(array>1, torch.log(array+1.), array)
    return array * start_clip_value

def huber_jan(array, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    """
    loss_squared = array**2
    array_abs = torch.abs(array)
    loss_linear = delta**2 + 2.*delta * (array_abs - delta)
    return tf.where(array_abs < delta, loss_squared, loss_linear)

def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    """
    return torch.where(d<=delta, .5*d**2, delta*(d-.5*delta))

def calc_L_energy(pred_energy, truth_energy):
    diff = torch.abs(pred_energy - truth_energy)
    L = 10. * torch.exp(-0.1 * diff**2 ) + 0.01*diff
    return softclip(L, 10.)

def calc_L_time(pred_time, truth_time):
    return softclip(huber(torch.abs(pred_time-truth_time), 2.), 6.)

def calc_L_position(pred_position: torch.Tensor, truth_position: torch.Tensor):
    d_squared = ((pred_position-truth_position)**2).sum(dim=-1)
    return softclip(huber(torch.sqrt(d_squared/100. + 1e-2), 10.), 3.)

def calc_L_classification(pred_pid, truth_pid):
    raise NotImplementedError

def calc_Lp(
    pred_beta: torch.Tensor, truth_cluster_index,
    pred_cluster_properties, truth_cluster_properties
    ):
    """
    Property loss

    Assumes:
    0 : energy,
    1 : time,
    2,3 : boundary crossing position,
    4 : pdgid
    """
    xi = torch.zeros_like(pred_beta)
    xi[truth_cluster_index > 0] = pred_beta[truth_cluster_index > 0].arctanh()

    L_energy = calc_L_energy(pred_cluster_properties[:,0], truth_cluster_properties[:,0])
    L_time = calc_L_time(pred_cluster_properties[:,1], truth_cluster_properties[:,1])
    L_position = calc_L_position(pred_cluster_properties[:,2:4], truth_cluster_properties[:,2:4])
    # L_classification = calc_L_classification(pred_cluster_properties[:,4], pred_cluster_properties[:,4]) TODO

    Lp = 0
    xi_sum = xi.sum()
    for L in [ L_energy, L_time, L_position ]:
        Lp += 1./xi_sum * (xi * L).sum()
    debug(f'Lp={Lp}')
    return Lp


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
    device = cluster_id.device
    assert cluster_id.device == batch.device
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.zeros(offset_values_nozero.size(0)+1, device=device)
    offset_values[1:] = offset_values_nozero
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def test_batch_cluster_indices():
    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    assert torch.all(torch.isclose(batch_cluster_indices(cluster_id, batch)[0], output))
    # Should also work fine with randomly shuffled data:
    shuffle = torch.randperm(cluster_id.size(0))
    assert torch.all(torch.isclose(
        batch_cluster_indices(cluster_id[shuffle], batch[shuffle])[0],
        output[shuffle]
        ))
    print(cluster_id[shuffle])
    print(batch[shuffle])
    print(batch_cluster_indices(cluster_id[shuffle], batch[shuffle])[0])
    print(output[shuffle])



def test_calc_LV_Lbeta():
    from gravnet_model import logger, debug, FakeDataset
    batch_size = 4
    train_loader = DataLoader(FakeDataset(100), batch_size=batch_size, shuffle=True)
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
    
def test_calc_Lp():
    from dataset import TauDataset
    for i, data in enumerate(DataLoader(TauDataset('data/taus'), batch_size=4, shuffle=True)):
        print(i, data)
        n_hits = data.x.size(0)
        Lp = calc_Lp(
            torch.rand(n_hits), # predicted betas
            data.y.type(torch.LongTensor), # truth cluster-index per hit
            torch.rand(n_hits, 5), # predicted cluster properties, 5
            data.cluster_properties
            )
        print(Lp)
        break


def get_clustering_np(betas: np.array, X: np.array, tbeta=.1, td=2.):
    n_points = betas.shape[0]
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-betas[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=.1, td=2.):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def generate_fake_betas_and_coords(N=20, n_clusters=2, cluster_space_dim=2, seed=1, add_background=True):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(
        n_samples=N,
        centers=n_clusters, n_features=cluster_space_dim,
        random_state=seed
        )
    centers = []
    # y consists of numbers from 0 to n_clusters, not indices
    y_indexed = np.zeros_like(y)
    # Pick centers: Find closest point to geometrical center
    for i_center in range(n_clusters):
        x = X[y==i_center]
        closest = np.argmin(np.linalg.norm(x - np.mean(x, axis=0), axis=-1))
        # This index is for only points that belong to this center;
        # transform it back to an index of all points
        global_index = np.nonzero(y==i_center)[0][closest]
        np.testing.assert_array_equal(X[global_index], x[closest])
        centers.append(global_index)
        y_indexed[y==i_center] = global_index
    centers = np.array(centers)
    # Generate some betas: random, except the center indices get a high value
    betas = 1e-3*np.random.rand(N)
    betas[centers] += 0.1 + 1e-3*np.random.rand(n_clusters)
    return betas, X, y_indexed


def add_background(betas, X, y, N=20):
    cluster_space_dim = X.shape[1]
    # Get dimensions of the clustering space
    cluster_space_min = np.min(X, axis=0)
    cluster_space_max = np.max(X, axis=0)
    cluster_space_width = cluster_space_max - cluster_space_min
    # Generate background: distributed anywhere except close to the centers
    # Loop for 100*N, but break at N successfully generated bkg points
    x_centers = X[np.unique(y)]
    x_noise = []
    for i in range(100*N):
        p = cluster_space_min + np.random.rand(cluster_space_dim)*cluster_space_width
        d = np.linalg.norm(p - x_centers, axis=-1)
        if np.all(d > 3.): x_noise.append(p)
        if len(x_noise) == N: break
    else:
        print(f'Warning: Generated only {len(x_noise)} bkg points, N={N} where requested')
    x_noise = np.array(x_noise)
    n_noise = x_noise.shape[0]
    if n_noise > 0:
        shuffle = np.random.permutation(X.shape[0] + n_noise)
        # Add to the input arrays
        X = np.concatenate((X, x_noise))[shuffle]
        y = np.concatenate((y, -1*np.ones(n_noise, dtype=np.int32)))[shuffle]
        betas = np.concatenate((betas, 1e-3*np.random.rand(n_noise)))[shuffle]
    return betas, X, y


def test_get_clustering():
    betas, X, y = add_background(*generate_fake_betas_and_coords(n_clusters=3), 20)
    clustering_np = get_clustering_np(betas, X, td=3.)
    print(y)
    print(clustering_np)
    np.testing.assert_array_equal(y, clustering_np)
    clustering_torch = get_clustering(torch.from_numpy(betas), torch.from_numpy(X), td=3.).numpy()
    print(clustering_torch)
    np.testing.assert_array_equal(y, clustering_torch)


def main():
    # pass
    # test_get_clustering()
    test_batch_cluster_indices()

if __name__ == '__main__':
    main()