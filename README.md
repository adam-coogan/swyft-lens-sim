# swyft-lens-sim

Minimal lensing simulator for coupling to swyft. See `simulator`, `noise`, `prior`, `Head` and the docstrings in scripts/conf/definitions.py for details. Setting up the simulator requires the files in scripts/conf/resources.

## Dependencies

### External

* Cuda/cuDNN, gcc and cmake
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d): clone, `cd` into the directory and install with `pip install .`.

### Our repositories

Clone these and install with `pip install .`.
* [clipppy](https://github.com/kosiokarchev/clipppy/): a wrapper around pyro.
* [pyrofit-utils](https://github.com/kosiokarchev/pyrofit-utils): utilities for probabilistic programming in the pytorch ecosystem.
* [pyrofit_lensing](https://github.com/cweniger/pyrofit_lensing): this contains all the lensing-specific code, data and notebooks. Use the `production` branch. The installation does not install everything required to run the code in `experiments`. To install their dependencies, run `pip install -r requirements.txt`.

## Notes

Currently clipppy does not take a `device` argument, and instead uses `torch.set_default_tensor_type` to decide when to use CUDA vs CPU tensors. As a result, `simulator` switches the default tensor type at the beginning and end of the function.
