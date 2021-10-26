import os, os.path as osp
import argparse
from time import strftime
import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import objectcondensation
from gravnet_model import GravnetModel, debug
from dataset import TauDataset
from lrscheduler import CyclicLRWithRestarts

from scripts.helper_funcs import convert_parallel_model
from scripts.plot_to_files import
from scripts.nadam import Nadam

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
        return LV + Lbeta + 1.

    # Lp = objectcondensation.calc_Lp(
    #     pred_betas,
    #     data.y.long(),
    #     pred_cluster_properties,
    #     data.truth_cluster_props
    #     )
    # return Lp + s_c*(LV + Lbeta)

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

def write_checkpoint(model, checkpoint_number=None, best=False):
    timestamp = strftime('%b%d_%H%M%S')
    ckpt_filename = 'ckpt_best_new.pth.tar' if best else f'ckpt_{timestamp}_{checkpoint_number}.pth.tar'
    ckpt = osp.join('checkpoints', ckpt_filename)
    if best: 
        print('Saving epoch {0} as new best'.format(checkpoint_number))

    #os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(dict(model=model.state_dict()), ckpt)

def distributed_training(rank, world_size, args):

    # creates default process (what's rank, world_size?)
    dist.init_process_group('gloo', rank=rank world_size=world_size)
    torch.manual_seed(42)

    # initialize model
    model = GravnetModel(input_dim=9, output_dim=4)
    if args.checkpoint:
        print('loading from checkpoint...')
        state_dict = torch.load('checkpoints/ckpt_best.pth.tar')['model']
        state_dict = convert_parallel_model(state_dict)
        model.load_state_dict(state_dict)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # initialize optimizer
    optimizer = Nadam(model.parameters(), lr=1e-5, weight_decay=1e-4, schedule_decay=4e-3)

    # get the data
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True,
            sampler=train_sampler
           )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    min_loss = 1e9
    for i_epoch in range(n_epochs):

        # run training loop
        print('Training epoch', epoch)
        model.train()

        # scheduler.step()
        pbar = tqdm.tqdm(data_loader, total=len(data_loader))
        pbar.set_postfix({'loss': '?'})
        n_data = len(data_loader)
        for i, data in enumerate(pbar):
            data = data.cuda(non_blocking=True)

            # forward pass
            result = model(data.x, data.batch)
            loss = loss_fn(result, data)

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.batch_step()
            pbar.set_postfix({'loss': float(loss)})
            tb_writer.add_scalar('train loss', loss, i + n_data*epoch)

        test_loss = test(ddp_model, test_loader, rank, i_epoch)
        tb_writer.add_scalar('test loss', test_loss, i_epoch)

        if not args.dry: 
            write_checkpoint(ddp_model, i_epoch)

        if test_loss < min_loss:
            min_loss = test_loss
            if not args.dry: 
                write_checkpoint(ddp_model, i_epoch, best=True)

    tb_writer.flush()
    dist.destroy_process_group()

if __name__ == '__main__':
    #debug()
    #run_profile()

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

    world_size = 3
    n_epochs = 100
    batch_size = 4
    torch.manual_seed(101)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(distributed_training, nprocs=3, args=(world_size, args))


    model = GravnetModel(input_dim=9, output_dim=4)
    if args.checkpoint:
        print('loading from checkpoint...')
        state_dict = torch.load('checkpoints/ckpt_best.pth.tar')['model']
        state_dict = convert_parallel_model(state_dict)
        model.load_state_dict(state_dict)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:

        # get one gpu as device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using {n_gpus - 1} GPUS')
        # this is setup for running on schmittgpu machine and leaves one GPU open
        model = nn.DataParallel(model, device_ids=[0, 2, 3], output_device=0)
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('Using device', device)

    model.to(device)

    epoch_size = len(train_loader.dataset)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")
    tb_writer = SummaryWriter(log_dir='logs')

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = Nadam(model.parameters(), 
            lr=1e-5, 
            weight_decay=1e-4,
            schedule_decay=4e-3
            )

    min_loss = 1e9
    for i_epoch in range(n_epochs):
        train(model, train_loader, optimizer, device, i_epoch, tb_writer=tb_writer)
        test_loss = test(model, test_loader, device, i_epoch)
        tb_writer.add_scalar('test loss', test_loss, i_epoch)

        if not args.dry: 
            write_checkpoint(model, i_epoch)

        if test_loss < min_loss:
            min_loss = test_loss
            if not args.dry: 
                write_checkpoint(model, i_epoch, best=True)

    tb_writer.flush()
    dist.destroy_process_group()

