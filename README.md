# Cosmological Parameter Estimation and Inference using Deep Summaries

This repostitory includes the code used in the submitted paper. The examples in the `notebooks` folder show how to use the code and reproduce most of the figures presented in the paper. Please read the __Installation__ section to install the required packages. 

## Notebooks

The examples in the `notebooks` folder give an introduction to the following things:

1. `data_generation.ipynb` This notebook shows how one can generate the data used in the *experiments* section of the paper. Note that the entire dataset needs more than 1TB of free space. We do not recommend to generate the entire dataset on a local machine.

2. `GCNN.ipynb` This notebook gives and introduction into the graph convolutional neural networks (GCNN) used in the paper. It shows how to use the loss function presented in the paper and provides functions to create first order parameter estimators.

3. `GP_ABC.ipynb` How to use Gaussian process regression on ABC log-likelihood estimates. It provides functions that estimate the ABC log-likelihood and their uncertainties, as well as a Gaussian process regression module. It can reproduce the constraints (Figure 4) of the __2D__ model presented in the paper.

4. `ground_truth.ipynb` This notebook implements the true likelihood function of the cosmological model used in the paper. We also provide the MCMC chain of the __2D__ model.

# Installation

Please note, that some of the packages required for the installation are not working properly on Windows. We strongly recomment doing eveything in a fresh virtual python environment. The code has been developed and tested with python version 3.6, it does __not__ work with python 2.7. To install the requirements proceed as follows. First you can install all requirements from the requirements file

```
pip install -r requirements.txt
```

There are two packages that can lead to some problems. The first one, `horovod`, needs to be installed __after__ the other requirements. Theoretically, one can install it with `pip`:

```
pip install horovod==0.21.1
```

Note that you should add the `--no-cache` flag, if you already have installed `horovod` in a different environment for a different `tensorflow` version. If you have trouble installing `horovod` please consult their [website](https://horovod.readthedocs.io/en/stable/install_include.html).

The second one is the [Core Cosmology Library (CCL)](https://github.com/LSSTDESC/CCL) which has a python wrapper called `pyccl`. If you use `conda`, the installation should be as easy as:

```
conda install -c conda-forge pyccl
```

To install `pyccl` with `pip` you need to install additional packages like `CMake`. The procedure is described on their [website](https://github.com/LSSTDESC/CCL).

The examples in the `notebooks` folder use [jupyter notebooks](https://jupyter.org/). If you already have a `jupyter` installtion on your machine, you can register your environment there, as described [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html). Otherwise you can freshly install it

```
pip install jupyterlab notebook
```

and start it up with

```
jupyter notebook
```
