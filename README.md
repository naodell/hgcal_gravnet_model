 1. Assuming you have CUDA installed, check the version with nvidia-smi. Then
install pytorch with a CUDA version <= whatever nvidia-smi reports (i.e. if
your CUDA version is 11.0, install torch for CUDA 10.4, not 11.1). See the
install instructions here: https://pytorch.org/ . For me it was basically:

```
	conda create -n hgcal-ml python=3.9
	conda activate hgcal-ml
	conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia
```
    
Optionally, torchvision and torchaudio can also be installed.

2. Get torch geometric. This part used to be harder but nowadays the following
seems to work quite well:

```
    conda install pytorch-geometric -c rusty1s -c conda-forge
```

3. Get the pytorch_cmspepr package:

```
    git clone https://github.com/cms-pepr/pytorch_cmspepr.git
    pip install -e pytorch_cmspepr
```

4. For now, clone this repo to use the latest gravnet model:

```
	git clone https://github.com/naodell/hgcal_gravnet_model.git
```
