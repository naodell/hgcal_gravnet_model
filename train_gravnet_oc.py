import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.data import DataLoader
import objectcondensation
from gravnet_model import GravnetModel, debug
from dataset import TauDataset
from lrscheduler import CyclicLRWithRestarts
import argparse

torch.manual_seed(1009)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    args = parser.parse_args()
    if args.verbose: objectcondensation.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    n_epochs = 400
    batch_size = 4

    shuffle = True
    dataset = TauDataset('data/taus')
    if args.dry:
        keep = .005
        print(f'Keeping only {100.*keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)
    train_dataset, test_dataset = dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModel(input_dim=9, output_dim=4).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    loss_offset = 1. # To prevent a negative loss from ever occuring

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
        out = objectcondensation.calc_LV_Lbeta(
            pred_betas,
            pred_cluster_space_coords,
            data.y.long(),
            data.batch,
            return_components=return_components
            )
        if return_components:
            return out[2]
        else:
            LV, Lbeta = out[:2]
            return LV + Lbeta + loss_offset
        # Lp = objectcondensation.calc_Lp(
        #     pred_betas,
        #     data.y.long(),
        #     pred_cluster_properties,
        #     data.truth_cluster_props
        #     )
        # return Lp + s_c*(LV + Lbeta)

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        # scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
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
        except Exception:
            print('Exception encountered:', data, ', npzs:')
            print('  ' + '\n  '.join([train_dataset.npzs[int(i)] for i in data.inpz]))
            raise

    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}
        def update(components):
            for key, value in components.items():
                if not key in loss_components: loss_components[key] = 0.
                loss_components[key] += value
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                update(loss_fn(result, data, return_components=True))
        # Divide by number of entries
        for key in loss_components:
            loss_components[key] /= N_test
        # Compute total loss and do printout
        total_loss = loss_components['V']+loss_components['beta']
        print('test ' + objectcondensation.formatted_loss_components_string(loss_components))
        print(f'Returning {loss_offset + total_loss}')
        return loss_offset + total_loss

    ckpt_dir = strftime('ckpts_gravnet_%b%d')
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    min_loss = 1e9
    for i_epoch in range(n_epochs):
        train(i_epoch)
        write_checkpoint(i_epoch)
        test_loss = test(i_epoch)
        if test_loss < min_loss:
            min_loss = test_loss
            write_checkpoint(i_epoch, best=True)

def debug():
    dataset = TauDataset('data/taus')
    dataset.npzs = [
        'data/taus/112_nanoML_93.npz',
        'data/taus/106_nanoML_29.npz',
        'data/taus/100_nanoML_56.npz',
        'data/taus/110_nanoML_81.npz',
        ]
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = GravnetModel(input_dim=9, output_dim=8)

    with torch.no_grad():
        model.eval()
        for data in loader:
            result = model(data.x, data.batch)
            loss = loss_fn(result, data)
            print(result)


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
    # debug()
    # run_profile()