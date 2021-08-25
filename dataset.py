import glob
import numpy as np
np.random.seed(1001)

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_scatter import scatter
from torch_geometric.data import (Data, Dataset, DataLoader, dataloader)

from sklearn.datasets import make_blobs


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


class BlobsDataset(Dataset):
    """
    Dataset around sklearn.datasets.make_blobs
    """
    
    def __init__(self, n_events=100, seed_offset=0):
        super(BlobsDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events
        self.cluster_space_dim = 2
        self.seed_offset = seed_offset

    def get(self, i):
        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(50, 70)
            n_clusters = min(np.random.randint(2, 4), n_hits)
            n_bkg = np.random.randint(10, 20)
            # Generate the 'signal'
            X, y = make_blobs(
                n_samples=n_hits,
                centers=n_clusters, n_features=self.cluster_space_dim,
                random_state=i+self.seed_offset
                )
            y += 1 # To reserve index 0 for background
            # Add background
            cluster_space_min = np.min(X, axis=0)
            cluster_space_max = np.max(X, axis=0)
            cluster_space_width = cluster_space_max - cluster_space_min
            X_bkg = cluster_space_min + np.random.rand(n_bkg, self.cluster_space_dim)*cluster_space_width
            y_bkg = np.zeros(n_bkg)
            X = np.concatenate((X,X_bkg))
            y = np.concatenate((y,y_bkg))
            # Calculate geom centers
            truth_cluster_props = np.zeros((n_hits+n_bkg,2))
            for i in range(1,n_clusters+1):
                truth_cluster_props[y==i] = np.mean(X[y==i], axis=0)
            # shuffle
            order = np.random.permutation(n_hits+n_bkg)
            # order = y.argsort()
            X = X[order]
            y = y[order]
            truth_cluster_props = truth_cluster_props[order]
            self.cache[i] = Data(
                x = torch.from_numpy(X).float(),
                y = torch.from_numpy(y).long(),
                truth_cluster_props = torch.from_numpy(truth_cluster_props).float()
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events
        

class TauDataset(Dataset):
    """
    Features in x:

    recHitEnergy,
    recHitEta,
    zeroFeature, #indicator if it is track or not
    recHitTheta,
    recHitR,
    recHitX,
    recHitY,
    recHitZ,
    recHitTime
    (https://github.com/cms-pepr/HGCalML/blob/master/modules/datastructures/TrainData_NanoML.py#L211-L221)
    """
    def __init__(self, path, flip=True):
        super(TauDataset, self).__init__(path)
        self.npzs = list(sorted(glob.iglob(path + '/*.npz')))
        self.flip = flip
    
    def get(self, i):
        d = np.load(self.npzs[i])
        x = d['recHitFeatures']
        if self.flip and np.mean(x[:,7]) < 0:
            # Negative endcap: Flip z-dependent coordinates
            print(f'Flipping {i}')
            x[:,1] *= -1 # eta
            x[:,7] *= -1 # z
        cluster_index = incremental_cluster_index_np(d['recHitTruthClusterIdx'].squeeze())
        truth_cluster_props = np.hstack((
            d['recHitTruthEnergy'],
            d['recHitTruthPosition'],
            d['recHitTruthTime'],
            d['recHitTruthID'],
            ))
        assert truth_cluster_props.shape == (x.shape[0], 5)
        order = cluster_index.argsort()
        return Data(
            x = torch.from_numpy(x[order]).type(torch.float),
            y = torch.from_numpy(cluster_index[order]).type(torch.int),
            truth_cluster_props = torch.from_numpy(truth_cluster_props[order]).type(torch.float),
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


def incremental_cluster_index_np(cluster_index_nonzeroindices, noise_index=-1):
    """
    Build a map that translates the random indices to ordered starting from zero
    E.g. [ -1 -1 13 -1 13 13 42 -1 -1] -> [ 0 0 1 0 1 1 2 0 0 ]
    """
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


def unique_via_cpu(input):
    """
    torch.unique is supposedly very slow on the gpu.
    This copies the tensor to cpu, performs unique, and returns output
    """
    return torch.unique(x.cpu(), sorted=False).to(x.device)


def incremental_cluster_index(input: torch.Tensor, noise_index=None):
    """
    Build a map that translates arbitrary indices to ordered starting from zero

    By default the first unique index will be 0 in the output, the next 1, etc.
    E.g. [13 -1 -1 13 -1 13 13 42 -1 -1] -> [0 1 1 0 1 0 0 2 1 1]

    If noise_index is not None, the output will be 0 where input==noise_index:
    E.g. noise_index=-1, [13 -1 -1 13 -1 13 13 42 -1 -1] -> [1 0 0 1 0 1 1 2 0 0]

    If noise_index is not None but the input does not contain noise_index, 0
    will still be reserved for it:
    E.g. noise_index=-1, [13 4 4 13 4 13 13 42 4 4] -> [1 2 2 1 2 1 1 3 2 2]
    """
    unique_indices, locations = torch.unique(input, return_inverse=True, sorted=True)

    cluster_index_map = torch.arange(unique_indices.size(0))
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return torch.gather(cluster_index_map, 0, locations).long()


def test_incremental_cluster_index():
    input = torch.LongTensor([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    assert torch.allclose(
        incremental_cluster_index(input),
        torch.LongTensor([1, 0, 0, 1, 0, 1, 1, 2, 0, 0])
        )
    # Noise index should get 0 if it is supplied:
    assert torch.allclose(
        incremental_cluster_index(input, noise_index=13),
        torch.LongTensor([0, 1, 1, 0, 1, 0, 0, 2, 1, 1])
        )
    # 0 should still be reserved for noise_index even if it is not present:
    assert torch.allclose(
        incremental_cluster_index(input, noise_index=-99),
        torch.LongTensor([2, 1, 1, 2, 1, 2, 2, 3, 1, 1])
        )


def test_ordered_cluster_index():
    a = [ -1, -1, 13, -1, 13, 13, 42, -1, -1]
    b = [ 0, 0, 1, 0, 1, 1, 2, 0, 0 ]
    np.testing.assert_array_equal(ordered_cluster_index(a), b)


def test_full_chain():
    dataset = TauDataset('data/taus')
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
    pred_cluster_props = out[:,3:]

    from objectcondensation import calc_LV_Lbeta, calc_L_energy, calc_Lp
    LV, Lbeta = calc_LV_Lbeta(
        betas,
        cluster_space_coordinates,
        data.y.type(torch.LongTensor),
        data.batch
        )
    print(LV, Lbeta)

    L_energy = calc_L_energy(
        pred_cluster_props[:,0],
        data.truth_cluster_props[:,0]
        )
    print(L_energy)

    Lp = calc_Lp(
        betas, data.y.type(torch.LongTensor),
        pred_cluster_props,
        data.truth_cluster_props
        )
    print(Lp)


def test_blobs():
    dataloader = DataLoader(BlobsDataset(4), batch_size=2, shuffle=True)
    from gravnet_model import GravnetModel
    model = GravnetModel(input_dim=2, output_dim=5)

    for data in dataloader:
        print(f'Sending data {data}')
        out = model(data.x, data.batch)
        break

    betas = torch.sigmoid(out[:,0])
    cluster_space_coordinates = out[:,1:3]
    pred_cluster_props = out[:,3:]

    print(betas)

    from objectcondensation import calc_LV_Lbeta, calc_L_energy, calc_Lp
    import objectcondensation
    objectcondensation.DEBUG = True

    LV, Lbeta = calc_LV_Lbeta(
        betas,
        cluster_space_coordinates,
        data.y.long(),
        data.batch,
        bkg_cluster_index=0
        )
    print(LV, Lbeta)


def test_z_flipping():
    dataset = TauDataset('data/taus')

    def print_some_numbers(event):
        print('New event')
        for i in range(9):
            print('  ', i, ':', event.x[:,i].numpy())

    event = dataset.get(0)
    print_some_numbers(event)

    event = dataset.get(1)
    print_some_numbers(event)

    dataset.flip = False
    event = dataset.get(1)
    print_some_numbers(event)


def main():
    test_z_flipping()
    # test_full_chain()
    # test_blobs()
    # test_incremental_cluster_index()

if __name__ == '__main__':
    main()