import glob
import numpy as np
np.random.seed(1001)

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_scatter import scatter
from torch_geometric.data import (Data, Dataset, DataLoader, dataloader)


class FakeDataset(Dataset):
    """
    Random number dataset to test with.
    Generates numbers on the fly, but also caches them so .get(i) will return
    something consistent
    """
    def __init__(self, n_events=100):
        super(FakeDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events

    def get(self, i):
        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(10, 100)
            n_clusters = min(np.random.randint(1, 6), n_hits)
            x = np.random.rand(n_hits, 5)
            y = (np.random.rand(n_hits) * n_clusters).astype(np.int8)
            # Also make a cluster 'truth': energy, boundary_x, boundary_y, pid (4)
            y_cluster = np.random.rand(n_clusters, 4)
            # pid (last column) should be an integer; do 3 particle classes now
            y_cluster[:,-1] = np.floor(y_cluster[:,-1] * 3)
            self.cache[i] = Data(
                x = torch.from_numpy(x).type(torch.float),
                y = torch.from_numpy(y),
                y_cluster = torch.from_numpy(y_cluster),
                n_clusters = torch.IntTensor([n_clusters])
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events


class TauDataset(Dataset):
    def __init__(self, path):
        super(TauDataset, self).__init__(path)
        self.npzs = list(sorted(glob.iglob(path + '/*.npz')))
    
    def get(self, i):
        d = np.load(self.npzs[i])
        x = d['recHitFeatures']
        cluster_index = incremental_cluster_index(d['recHitTruthClusterIdx'].squeeze())
        cluster_properties = np.hstack((
            d['recHitTruthEnergy'],
            d['recHitTruthPosition'],
            d['recHitTruthTime'],
            d['recHitTruthID'],
            ))
        assert cluster_properties.shape == (x.shape[0], 5)
        order = cluster_index.argsort()
        return Data(
            x = torch.from_numpy(x[order]).type(torch.float),
            y = torch.from_numpy(cluster_index[order]).type(torch.int),
            cluster_properties = torch.from_numpy(cluster_properties[order]).type(torch.float),
            inpz = torch.Tensor([i])
            )

    def __len__(self):
        return len(self.npzs)
    def len(self):
        return len(self.npzs)

    def split(self, fraction):
        """
        Creates two new instances of TauDataset with a fraction of events split
        """
        left = self.__class__(self.root)
        right = self.__class__(self.root)
        split_index = int(fraction*len(self))
        left.npzs = self.npzs[:split_index]
        right.npzs = self.npzs[split_index:]
        return left, right


def incremental_cluster_index(cluster_index_nonzeroindices, noise_index=-1):
    # Build a map that translates the random indices to ordered starting from zero
    # E.g. [ -1 -1 13 -1 13 13 42 -1 -1] -> [ 0 0 1 0 1 1 2 0 0 ]
    # cluster_index_map = { -1 : 0 } # Always translate -1 (noise) to index 0
    cluster_index_map = {}
    for cluster_index in np.unique(cluster_index_nonzeroindices):
        if cluster_index == noise_index: continue # Filled with zeroes automatically
        cluster_index_map[cluster_index] = len(cluster_index_map)+1

    # Build the new cluster_index array, element by element
    cluster_index = np.zeros_like(cluster_index_nonzeroindices)
    for old_index, new_index in cluster_index_map.items():
        cluster_index[cluster_index_nonzeroindices == old_index] = new_index

    return cluster_index

def test_ordered_cluster_index():
    a = [ -1, -1, 13, -1, 13, 13, 42, -1, -1]
    b = [ 0, 0, 1, 0, 1, 1, 2, 0, 0 ]
    np.testing.assert_array_equal(ordered_cluster_index(a), b)


def test_full_chain():

    dataset = TauDataset('data/taus')
    print(dataset.get(0))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    from gravnet_model import GravnetModel

    model = GravnetModel(input_dim=9, output_dim=8)

    for data in dataloader:
        print(f'Sending data {data}')
        # print(data.y)
        out = model(data.x, data.batch)
        break

    print(out)

    betas = out[:,0]
    cluster_space_coordinates = out[:,1:3]
    pred_cluster_properties = out[:,3:]

    from objectcondensation import calc_LV_Lbeta, calc_L_energy, calc_Lp
    LV, Lbeta = calc_LV_Lbeta(
        betas,
        cluster_space_coordinates,
        data.y.type(torch.LongTensor),
        data.batch
        )
    print(LV, Lbeta)

    L_energy = calc_L_energy(
        pred_cluster_properties[:,0],
        data.cluster_properties[:,0]
        )
    print(L_energy)

    Lp = calc_Lp(
        betas, data.y.type(torch.LongTensor),
        pred_cluster_properties,
        data.cluster_properties
        )
    print(Lp)


def main():
    test_full_chain()

if __name__ == '__main__':
    main()