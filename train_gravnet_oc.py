import tqdm
import torch
from torch_geometric.data import DataLoader
import objectcondensation, dataset
from gravnet_model import GravnetModel
from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1004)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 4
    shuffle = False
    all_dataset, _ = dataset.TauDataset('data/taus').split(.1)
    train_dataset, test_dataset = all_dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModel(input_dim=9, output_dim=8).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    def objectcondensation_loss(out, data, s_c=1.):
        pred_betas = out[:,0]
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

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for data in pbar:
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = objectcondensation_loss(result, data)
                loss.backward()
                optimizer.step()
                scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
        except Exception:
            print('Exception encountered:', data)
            for i in data.inpz:
                print(train_dataset.npzs[i])
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

if __name__ == '__main__':
    main()