import torch
from torch_geometric.data import DataLoader
import numpy as np
import tqdm

from torch_cmspepr.dataset import TauDataset
from torch_cmspepr.gravnet_model import GravnetModel
import torch_cmspepr.objectcondensation as oc
import scripts.plotting as plotting

def pred_plot(event, out, tbeta=.1, td=1.):
    betas = torch.sigmoid(out[:,0]).numpy()
    cluster_space_coords = out[:,1:4].numpy()
    clustering = oc.get_clustering_np(betas, cluster_space_coords, tbeta=tbeta, td=td)
    return plotting.get_plotly_pred(event, clustering)

def truth_plot(event):
    return plotting.get_plotly_truth(event)

def clusterspace_plot(event, out):
    return plotting.get_plotly_clusterspace(event, out[:,1:])

def pred_clusterspace_plot(event, out, tbeta=.1, td=1.):
    betas = torch.sigmoid(out[:,0]).numpy()
    cluster_space_coords = out[:,1:4].numpy()
    clustering = oc.get_clustering_np(betas, cluster_space_coords, tbeta=tbeta, td=td)
    return plotting.get_plotly_clusterspace(event, out[:,1:], clustering)

def main():
    _, test_dataset = TauDataset('local_data/npzs_all').split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GravnetModel(input_dim=9, output_dim=4)
    ckpt = 'checkpoints/ckpts_gravnet_Sep13_2022/ckpt_32.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

    tbeta = .2
    td = .1
    nmax = 10

    desc_str = f'tbeta{tbeta:.1f}_td{td:.1f}'.replace('.', 'p')

    with torch.no_grad():
        model.eval()
        for i, data in tqdm.tqdm(enumerate(test_loader), total=nmax):
            if i == nmax: break
            out = model(data.x, data.batch)
            outfile = f'plots_%b%d_{desc_str}/{i:03d}.html'
            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    pred_plot(data, out, tbeta, td),
                    plotting.get_plotly_truth(data)
                    )
                )
            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    pred_clusterspace_plot(data, out, tbeta, td),
                    plotting.get_plotly_clusterspace(data, out[:,1:])
                    ),
                mode = 'a'
                )
            

if __name__ == "__main__":
    main()
