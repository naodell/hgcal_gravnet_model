import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.data import DataLoader
import objectcondensation
from gravnet_model import GravnetModel, debug
from dataset import BlobsDataset
from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1009)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    do_checkpoints = True
    # do_checkpoints = False
    n_epochs = 400
    batch_size = 64

    train_loader = DataLoader(BlobsDataset(6000), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(BlobsDataset(1000), batch_size=batch_size, shuffle=True)

    model = GravnetModel(input_dim=2, output_dim=5).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    def loss_fn(data, out):
        device = out.device
        pred_betas = torch.sigmoid(out[:,0])
        pred_cluster_space_coords = out[:,1:3]
        assert pred_betas.device == device
        assert pred_cluster_space_coords.device == device
        assert data.y.device == device
        assert data.batch.device == device

        # Ignore cluster properties for now
        # pred_cluster_properties = out[:,3:]
        # assert pred_cluster_properties.device == device
        # assert data.cluster_properties.device == device

        LV, Lbeta = objectcondensation.calc_LV_Lbeta(
            pred_betas,
            pred_cluster_space_coords,
            data.y.long(),
            data.batch
            )
        return LV + Lbeta

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
                loss = loss_fn(data, result)
                loss.backward()
                optimizer.step()
                # scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
        except Exception:
            print('Exception encountered:', data)
            raise

    def test(epoch):
        with torch.no_grad():
            model.eval()
            loss = 0.
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                loss += loss_fn(data, result)
            loss /= len(test_loader)
            print(f'Avg test loss: {loss}')
        return loss

    ckpt_dir = strftime('ckpts_blobs_%b%d')
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if do_checkpoints:
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


if __name__ == '__main__':
    main()