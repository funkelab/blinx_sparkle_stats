# *sparkles ✨*

*sparkles* is a deep learning model that predicts priors for [*blinx*](github.com/funkelab/blinx).

# Installation

For a CPU installation:

```bash
conda create -n sparkles python pytorch cpuonly numpy matplotlib zarr jax -c pytorch
git clone https://github.com/funkelab/blinx_sparkle_stats.git sparkles
cd sparkles
pip install .
```

For a GPU installation:

```bash
conda create -n sparkles_gpu python pytorch pytorch-cuda cudatoolkit cudatoolkit-dev cudnn jaxlib=0.4.31=*cuda* jax=0.4.31 cuda-nvcc numpy matplotlib zarr -c pytorch -c nvidia
git clone https://github.com/funkelab/blinx_sparkle_stats.git sparkles
cd sparkles
pip install .
```

Note that this installation pins JAX to 0.4.31. You might need to pin certain cuda or jax versions to ensure compatability. 

# Scripts
See the `scripts` directory for example scripts regarding dataset creation, model training, and prediction. 