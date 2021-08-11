import tqdm
import torch
from torch_geometric.data import DataLoader
import objectcondensation
from gravnet_model import GravnetModel, debug
from dataset import TauDataset
from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1009)

def objectcondensation_loss(out, data, s_c=1.):
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:3]
    pred_cluster_properties = out[:,3:]
    LV, Lbeta = objectcondensation.calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        data.y.type(torch.LongTensor),
        data.batch
        )
    Lp = objectcondensation.calc_Lp(
        pred_betas,
        data.y.type(torch.LongTensor),
        pred_cluster_properties,
        data.cluster_properties
        )
    return Lp + s_c*(LV + Lbeta)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 4
    shuffle = True
    dataset, _ = TauDataset('data/taus').split(.1)
    train_dataset, test_dataset = dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModel(input_dim=9, output_dim=8).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        # scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for data in pbar:
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = objectcondensation_loss(result, data)
                print(f'loss={float(loss)}')
                loss.backward()
                optimizer.step()
                # scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
        except Exception:
            print('Exception encountered:', data, ', npzs:')
            print('  ' + '\n  '.join([train_dataset.npzs[int(i)] for i in data.inpz]))
            raise

    def test(epoch):
        with torch.no_grad():
            model.eval()
            loss = 0.
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                loss += objectcondensation_loss(result, data)
            loss /= len(test_loader)
            print(f'Avg test loss: {loss}')

    for i_epoch in range(20):
        train(i_epoch)
        test(i_epoch)

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
            loss = objectcondensation_loss(result, data)
            print(result)


if __name__ == '__main__':
    main()
    # debug()