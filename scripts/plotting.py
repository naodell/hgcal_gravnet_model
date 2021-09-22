import numpy as np
import uuid
import torch


class ColorWheel:
    '''Returns a consistent color when given the same object'''
    def __init__(self, colors=None, seed=44):
        if colors is None:
            import matplotlib._color_data as mcd
            self.colors = list(mcd.XKCD_COLORS.values())
        else:
            self.colors = colors
        np.random.seed(seed)
        np.random.shuffle(self.colors)
        self._original_colors = self.colors.copy()
        self.assigned_colors = {}
        
    def __call__(self, thing):
        key = id(thing)
        if key in self.assigned_colors:
            return self.assigned_colors[key]
        else:
            color = self.colors.pop()
            self.assigned_colors[key] = color
            if not(self.colors): self.colors = self._original_colors.copy()
            return color
    
    def assign(self, thing, color):
        """Assigns a specific color to a thing"""
        key = id(thing)
        self.assigned_colors[key] = color
        if color in self.colors: self.colors.remove(color)


def get_plotly_pred(event, clustering):
    import plotly.graph_objects as go
    colorwheel = ColorWheel()
    colorwheel.assign(-1, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    for cluster_index in np.unique(clustering):
        x = event.x[clustering == cluster_index].numpy()
        energy = x[:,0]
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,7], y=x[:,5], z=x[:,6],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data


def get_plotly_truth(event):
    import plotly.graph_objects as go
    colorwheel = ColorWheel()
    colorwheel.assign(0, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    for cluster_index in np.unique(event.y):
        x = event.x[event.y == cluster_index].numpy()
        energy = x[:,0]
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,7], y=x[:,5], z=x[:,6],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data


def get_plotly_clusterspace(event, cluster_space_coords, clustering=None):
    assert cluster_space_coords.size(1) == 3
    import plotly.graph_objects as go

    colorwheel = ColorWheel()
    colorwheel.assign(0, '#bfbfbf')
    colorwheel.assign(-1, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    if clustering is None: clustering = event.y

    for cluster_index in np.unique(clustering):
        x = cluster_space_coords[clustering == cluster_index].numpy()
        energy = event.x[:,0].numpy()
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,0], y=x[:,1], z=x[:,2],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data

def side_by_side_html(data1, data2,
                      info=None, title1=None, title2=None, width=600, height=None, include_plotlyjs='cdn',
                      return_divids=False
                     ):
    import plotly.graph_objects as go

    scene = dict(
        xaxis_title='z', yaxis_title='x', zaxis_title='y',
        aspectmode='cube'
        )
    if info:
        scene.update(dict(
            xaxis_range=[320., info['zmax']],
            yaxis_range=[info['xmin'], info['xmax']],
            zaxis_range=[info['ymin'], info['ymax']],
            ))

    if height is None: 
        height = width

    fig1 = go.Figure(data=data1, **(dict(layout_title_text=title1) if title1 else {}))
    fig1.update_layout(width=width, height=height, scene=scene)

    fig2 = go.Figure(data=data2, **(dict(layout_title_text=title2) if title2 else {}))
    fig2.update_layout(width=width, height=height, scene=scene)

    fig1_html = fig1.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    divid1 = fig1_html.split('<div id="',1)[1].split('"',1)[0]
    divid2 = fig2_html.split('<div id="',1)[1].split('"',1)[0]

    id1 = str(uuid.uuid4())[:6]
    id2 = str(uuid.uuid4())[:6]

    html = (
        f'<div style="width: 47%; display: inline-block">\n{fig1_html}\n</div>'
        f'\n<div style="width: 47%; display: inline-block">\n{fig2_html}\n</div>'
        f'\n<script>'
        f'\nvar graphdiv_{id1} = document.getElementById("{divid1}");'
        f'\nvar graphdiv_{id2} = document.getElementById("{divid2}");'
        f'\nvar isUnderRelayout_{id1} = false'
        f'\ngraphdiv_{id1}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id1})'
        f'\n    if (!isUnderRelayout_{id1}) {{'
        f'\n        Plotly.relayout(graphdiv_{id2}, {{"scene.camera": graphdiv_{id1}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id1} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id1} = true;'
        f'\n    }})'
        f'\nvar isUnderRelayout_{id2} = false'
        f'\ngraphdiv_{id2}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id2})'
        f'\n    if (!isUnderRelayout_{id2}) {{'
        f'\n        Plotly.relayout(graphdiv_{id1}, {{"scene.camera": graphdiv_{id2}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id2} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id2} = true;'
        f'\n    }})'
        f'\n</script>'
        )
    return (html, id1, id2) if return_divids else html

def write_html(outfile, html, mode='w'):
    import os, os.path as osp
    from time import strftime
    outfile = strftime(outfile)
    outdir = osp.dirname(osp.abspath(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    with open(outfile, mode) as f:
        f.write(html)
