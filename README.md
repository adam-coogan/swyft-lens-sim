# swyft-lens-sim

![Lensing simulator samples](https://github.com/adam-coogan/swyft-lens-sim/blob/master/resources/samples.gif)

Minimal lensing model for coupling to swyft. See scripts/conf/definitions.py for details. The model requires the files in scripts/conf/resources.

## Dependencies

* A GPU with cuda/cuDNN, gcc and cmake
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d): clone and install with `pip install .`.

Clone these and install with `pip install .`:
* [clipppy](https://github.com/kosiokarchev/clipppy/)
* [pyrofit-utils](https://github.com/kosiokarchev/pyrofit-utils)
* [pyrofit_lensing](https://github.com/cweniger/pyrofit_lensing), `production` branch: the lensing code. To install dependencies for notebooks in the experiments directory, run `pip install -r requirements.txt` in the root directory of the repository.

## Possible issues

* The keops compilation required the first time the model runs takes a while (≥ minutes, ≤ 1 hour). If the job is interrupted during compilation, you may get errors when you rerun the code. To fix these, clear the keops cache with `import pykeops` and `pykeops.clean_pykeops()`.
* You can test your keops installation using [these instructions](https://www.kernel-operations.io/keops/python/installation.html#testing-your-installation).
* Currently clipppy does not take a `device` argument, and uses `torch.set_default_tensor_type` to decide when to use CUDA vs CPU tensors. As a result, `simulator` switches the default tensor type at the beginning and end of the function. This has not been an issue so far, but is worth keeping in mind.
