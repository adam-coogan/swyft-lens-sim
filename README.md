# swyft-lens-sim

Minimal lensing simulator for coupling to swyft.

## Dependencies

### External

* Cuda/cuDNN, gcc and cmake
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d): clone, `cd` into the directory and install with `pip`.

### Our repositories

Clone these and install with `pip`.
* [clipppy](https://github.com/kosiokarchev/clipppy/): a wrapper around pyro.
* [pyrofit-utils](https://github.com/kosiokarchev/pyrofit-utils): utilities for probabilistic programming in the pytorch ecosystem.
* [pyrofit_lensing](https://github.com/cweniger/pyrofit_lensing): this contains all the lensing-specific code, data and notebooks. Use the `production` branch. The installation does not install everything required to run the code in `experiments`. To install their dependencies, run `pip install -r requirements.txt`.
