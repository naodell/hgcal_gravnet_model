import logging
import numpy as np
np.random.seed(1001)

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_scatter import scatter
# from torch_geometric.nn import global_mean_pool
from torch_geometric.data import (Data, Dataset, DataLoader)

# from torch_geometric.nn.conv.gravnet_conv import GravNetConv
from torch_cmspepr import GravNetConv


def setup_logger(name="hgcalmb", fmt=None):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        if fmt is None:
            fmt = logging.Formatter(
                fmt=f"\033[33m[%(levelname)s|%(funcName)s|%(lineno)s]:\033[0m %(message)s",
                )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger
logger = setup_logger() 
debug = logger.debug


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
            # Also make a cluster 'truth': energy, time, boundary_x, boundary_y, pid (4)
            y_cluster = np.random.rand(n_clusters, 5)
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


def batch_to_numpy(data):
    """
    Transforms a torch geometric Data object to regular np arrays for debugging
    """
    x = data.x.numpy()
    batch = data.batch.numpy()
    y = data.y.numpy()
    n_events = max(batch)+1
    x_per_event = []
    y_per_event = []
    for i in range(n_events):
        x_per_event.append(x[batch == i])
        y_per_event.append(y[batch == i])
    return x_per_event, y_per_event


def test_scatter_reduce():
    """
    Only for testing whether the torch.scatter calls return what
    you would expect them to
    """
    for data in DataLoader(FakeDataset(4), batch_size=4, shuffle=True):
        x = data.x
        batch = data.batch
        x_batch0 = x[batch == 0].numpy() # Take only the first batch for testing
        # Compare the first element of scatter to a manual calculation with numpy
        # on the first batch
        np.testing.assert_almost_equal(
            np.mean(x_batch0, axis=0),
            scatter(x, batch, dim=0, reduce='mean')[0].numpy()
            )
        np.testing.assert_almost_equal(
            np.min(x_batch0, axis=0),
            scatter(x, batch, dim=0, reduce='min')[0].numpy()
            )
        np.testing.assert_almost_equal(
            np.max(x_batch0, axis=0),
            scatter(x, batch, dim=0, reduce='max')[0].numpy()
            )
        return


def global_exchange(x: Tensor, batch: Tensor) -> Tensor:
    """
    Adds columns for the means, mins, and maxs per feature, per batch.
    Assumes x: (n_hits x n_features), batch: (n_hits),
    and that the batches are sorted!
    """
    device = x.device
    batch_numbers, batch_counts = torch.unique(batch, return_counts=True)
    batch_size = batch_counts.size()[0]
    n_hits, n_features = x.size()

    # minmeanmax: (batch_size x 3*n_features)
    meanminmax = torch.cat((
        scatter(x, batch, dim=0, reduce='mean'),
        scatter(x, batch, dim=0, reduce='min'),
        scatter(x, batch, dim=0, reduce='max')
        ), dim=-1)
    assert meanminmax.size() == (batch_size, 3*n_features)

    # (batch_size x 3*n_features) --> (n_hits x 3*n_features),
    # by copying the minmeanmax per batch by the counts per batch.
    # Not sure if for-loop can be avoided here.
    # This assumes the batch tensor is sorted!
    meanminmax = torch.cat(
        [
            meanminmax[i_batch].unsqueeze(dim=0).expand(batch_count, 3*n_features)
            for i_batch, batch_count in zip(batch_numbers, batch_counts)
            ],
        dim=0
        )
    assert meanminmax.size() == (n_hits, 3*n_features)

    # Add as columns to feature tensor
    out = torch.cat((meanminmax, x), dim=-1)
    assert out.size() == (n_hits, 4*n_features)

    assert all(t.device == device for t in [batch_numbers, batch_counts, meanminmax, out, x, batch])
    return out


# FROM https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7113-9.pdf:

# GravNet model: The model consists of four blocks. Each
# block starts with concatenating the mean of the vertex
# features to the vertex features, three dense layers with
# 64 nodes and tanh activation, and one GravNet layer
# with S = 4 coordinate dimensions, FLR = 22 features to
# propagate, and FOUT = 48 output nodes per vertex. For
# each vertex, 40 neighbours are considered. The output
# of each block is passed as input to the next block and
# added to a list containing the output of all blocks. This
# determines the full vector of vertex features passed to a
# final dense layer with 128 nodes and ReLU activation

# In all cases, each output vertex of these model building blocks
# is fed through one dense layer with ReLU activation and three
# nodes, followed by a dense layer with two output nodes and
# softmax activation. This last processing step deter- mines the
# energy fraction belonging to each shower. Batch normalisation
# is applied in all models to the input and after each block.

class GravNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int, out_channels: int = 96,
        space_dimensions: int = 4, propagate_dimensions: int = 22, k: int = 40
        ):
        super(GravNetBlock, self).__init__()
        # Includes all layers up to the global_exchange
        self.gravnet_layer = GravNetConv(
                in_channels, out_channels,
                space_dimensions, propagate_dimensions, k
                )
        self.post_gravnet = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.Tanh(),
            )
        self.output = nn.Sequential(
            nn.Linear(4*96, 96),
            nn.Tanh(),
            nn.BatchNorm1d(96)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.gravnet_layer(x, batch)
        x = self.post_gravnet(x)
        assert x.size(1) == 96
        x = global_exchange(x, batch)
        x = self.output(x)
        assert x.size(1) == 96
        return x


class GravnetModel(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=4,
        n_gravnet_blocks: int=4,
        n_postgn_dense_blocks: int=4,
        ):
        super(GravnetModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks

        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        self.input = nn.Linear(4*input_dim, 64)

        # Note: out_channels of the internal gravnet layer
        # not clearly specified in paper
        self.gravnet_blocks = nn.ModuleList([
            GravNetBlock(64 if i==0 else 96) for i in range(self.n_gravnet_blocks)
            ])

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend([
                nn.Linear(4*96 if i==0 else 128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                ])
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        
        # Output block
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        device = x.device

        # print('forward called on device', device)
        x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = [] # To store intermediate outputs
        for gravnet_block in self.gravnet_blocks:
            x = gravnet_block(x, batch)
            x_gravnet_per_block.append(x)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4*96)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.output(x)
        assert x.device == device
        return x


def test_model_sizes():
    batch_size = 4
    train_loader = DataLoader(FakeDataset(100), batch_size=batch_size, shuffle=True)

    model = GravnetModel()
    print(model)

    for i, data in enumerate(train_loader):
        print(i, data)
        x = model(data.x, data.batch)
        print(x.size())
        return

def main():
    pass

if __name__ == '__main__':
    main()
