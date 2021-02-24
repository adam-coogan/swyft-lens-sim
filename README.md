# swyft-lens-sim

![Lensing simulator samples](https://github.com/adam-coogan/swyft-lens-sim/blob/master/resources/samples.gif)

Minimal lensing model for coupling to swyft. See `model`, `noise`, `prior`, `Head` and the docstrings in scripts/conf/definitions.py for details. Setting up the model requires the files in scripts/conf/resources.

## Dependencies

### External

* Cuda/cuDNN, gcc and cmake
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d): clone, `cd` into the directory and install with `pip install .`.

### Our repositories

Clone these and install with `pip install .`.
* [clipppy](https://github.com/kosiokarchev/clipppy/): a wrapper around pyro.
* [pyrofit-utils](https://github.com/kosiokarchev/pyrofit-utils): utilities for probabilistic programming in the pytorch ecosystem.
* [pyrofit_lensing](https://github.com/cweniger/pyrofit_lensing): this contains all the lensing-specific code, data and notebooks. Use the `production` branch. The installation does not install everything required to run the code in `experiments`. To install their dependencies, run `pip install -r requirements.txt`.

### Possible issues

* The most annoying dependency is keops. You can test whether your installation is working by following [these instructions](https://www.kernel-operations.io/keops/python/installation.html#testing-your-installation).
* The first time you run the lensing code, keops has to compile a bunch of cuda code. This takes a while, so if you're running on a Lisa login node your job might get killed before compilation finishes. In this case you may get weird errors when you rerun the code. These can usually be fixed by clearing the keops cache (`import pykeops`, `pykeops.clean_pykeops()`) and running the code on a compute node so your job won't get killed.

## Notes

Currently clipppy does not take a `device` argument, and instead uses `torch.set_default_tensor_type` to decide when to use CUDA vs CPU tensors. As a result, `simulator` switches the default tensor type at the beginning and end of the function.
