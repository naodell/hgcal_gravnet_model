import os, os.path as osp
import argparse
from time import strftime
import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import objectcondensation
from gravnet_model import GravnetModel, debug
from dataset import TauDataset
from lrscheduler import CyclicLRWithRestarts
from scripts.helper_funcs import convert_parallel_model

torch.manual_seed(1009)

def loss_fn(out, data, s_c=1., return_components=False):
    device = out.device
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    # pred_cluster_properties = out[:,3:]
    assert all(t.device == device for t in [
        pred_betas, pred_cluster_space_coords, data.y,
        data.batch,
        # pred_cluster_properties, data.truth_cluster_props
        ])
    out_oc = objectcondensation.calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch,
        return_components=return_components
        )
    if return_components:
        return out_oc
    else:
        LV, Lbeta = out_oc
        return LV + Lbeta + 1
    # Lp = objectcondensation.calc_Lp(
    #     pred_betas,
    #     data.y.long(),
    #     pred_cluster_properties,
    #     data.truth_cluster_props
    #     )
    # return Lp + s_c*(LV + Lbeta)

def train(model, data_loader, device, epoch):
    print('Training epoch', epoch)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler.step()
    pbar = tqdm.tqdm(data_loader, total=len(data_loader))
    pbar.set_postfix({'loss': '?'})
    for i, data in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        result = model(data.x, data.batch)
        loss = loss_fn(result, data)
        loss.backward()
        optimizer.step()
        # scheduler.batch_step()
        pbar.set_postfix({'loss': float(loss)})
        # if i == 2: raise Exception

def test(model, data_loader, device, epoch):
    N_test = len(data_loader)
    loss_components = {}

    def update(components):
        for key, value in components.items():
            if not key in loss_components: loss_components[key] = 0.
            loss_components[key] += value

    with torch.no_grad():
        model.eval()
        for data in tqdm.tqdm(data_loader, total=len(data_loader)):
            data = data.to(device)
            result = model(data.x, data.batch)
            update(loss_fn(result, data, return_components=True))

    # Divide by number of entries
    for key in loss_components:
        loss_components[key] /= N_test

    # Compute total loss and do printout
    print('test ' + objectcondensation.formatted_loss_components_string(loss_components))
    test_loss = 1 + loss_components['L_V']+loss_components['L_beta']
    print(f'Returning {test_loss}')
    return test_loss

def write_checkpoint(checkpoint_number=None, best=False):
    timestamp = strftime('%b%d_%H%M%S')
    ckpt_filename = 'ckpt_best.pth.tar' if best else f'ckpt_{timestamp}_{checkpoint_number}.pth.tar'
    ckpt = osp.join('checkpoints', ckpt_filename)
    if best: 
        print('Saving epoch {0} as new best'.format(checkpoint_number))

    #os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(dict(model=model.state_dict()), ckpt)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('-c', '--checkpoint', action='store_true', help = 'Load checkpoint file with model weights.')
            #default = None,
            #type = str
            #)
    args = parser.parse_args()
    if args.verbose: 
        objectcondensation.DEBUG = True

    n_epochs = 100
    batch_size = 16

    shuffle = True
    dataset = TauDataset('data/taus')
    dataset.blacklist([ # Remove a bunch of bad events
        'data/taus/110_nanoML_98.npz',
        'data/taus/113_nanoML_13.npz',
        'data/taus/124_nanoML_77.npz',
        'data/taus/128_nanoML_70.npz',
        'data/taus/149_nanoML_90.npz',
        'data/taus/153_nanoML_22.npz',
        'data/taus/26_nanoML_93.npz',
        'data/taus/32_nanoML_45.npz',
        'data/taus/5_nanoML_51.npz',
        'data/taus/86_nanoML_97.npz',
        ])

    if args.dry:
        keep = .005
        print(f'Keeping only {100.*keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)

    train_dataset, test_dataset = dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModel(input_dim=9, output_dim=4)
    if args.checkpoint:
        print('loading from checkpoint...')
        state_dict = torch.load('checkpoints/ckpt_best.pth.tar')['model']
        state_dict = convert_parallel_model(state_dict)
        model.load_state_dict(state_dict)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f'Using {n_gpus - 1} GPUS')
        # this is setup for running on schmittgpu machine and leaves one GPU open
        model = nn.DataParallel(model, device_ids=[1, 2, 3], output_device=1)
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('Using device', device)

    model.to(device)

    epoch_size = len(train_loader.dataset)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    min_loss = 1e9
    for i_epoch in range(n_epochs):
        train(model, train_loader, device, i_epoch)
        test_loss = test(model, test_loader, device, i_epoch)

        if not args.dry: 
            write_checkpoint(i_epoch)

        if test_loss < min_loss:
            min_loss = test_loss
            if not args.dry: 
                write_checkpoint(i_epoch, best=True)


def debug():
    objectcondensation.DEBUG = True
    dataset = TauDataset('data/taus')
    dataset.npzs = [
        # 'data/taus/49_nanoML_84.npz',
        # 'data/taus/37_nanoML_4.npz',
        'data/taus/26_nanoML_93.npz',
        # 'data/taus/142_nanoML_75.npz',
        ]
    for data in DataLoader(dataset, batch_size=len(dataset), shuffle=False): break
    print(data.y.sum())
    model = GravnetModel(input_dim=9, output_dim=4)
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.batch)
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    out_oc = objectcondensation.calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch.long()
        )

def run_profile():
    from torch.profiler import profile, record_function, ProfilerActivity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 2
    n_batches = 2
    shuffle = True
    dataset = TauDataset('data/taus')
    dataset.npzs = dataset.npzs[:batch_size*n_batches]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Running profiling for {len(dataset)} events, batch_size={batch_size}, {len(loader)} batches')

    model = GravnetModel(input_dim=9, output_dim=8).to(device)
    epoch_size = len(loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print('Start limited training loop')
    model.train()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            pbar = tqdm.tqdm(loader, total=len(loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = loss_fn(result, data)
                print(f'loss={float(loss)}')
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
    # Other valid keys:
    # cpu_time, cuda_time, cpu_time_total, cuda_time_total, cpu_memory_usage,
    # cuda_memory_usage, self_cpu_memory_usage, self_cuda_memory_usage, count

if __name__ == '__main__':
    main()
    #debug()
    #run_profile()

