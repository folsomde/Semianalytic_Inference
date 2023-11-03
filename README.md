# README
This repository contains code relevant to Folsom et al. (2023), "Probabilistic Inference of the Structure and Orbit of Milky Way Satellites with Semi-Analytic Modeling." The code provided demonstrates how to infer distributions of various properties of Milky Way satellites and is designed with the intention of being fairly easy to interpret and extend.

There are four datasets associated with this paper, [available on Zenodo](https://doi.org/10.5281/zenodo.10068112). These datasets correspond to each combination of the baryonic feedback model (NIHAO emulator vs APOSTLE emulator) and stellar mass -- halo mass relation (RP17 vs B13). See the paper for more detail on the datasets themselves.

The `sample_notebook.ipynb` file provides an example of how to use the code, and it acts as a quickstart guide to performing other inferences. Each of the modules has documentation provided in the form of docstrings (accessible in HTML format [here](https://rawcdn.githack.com/folsomde/Semianalytic_Inference/main/docs/index.html)) which can be consulted for more information.

For further inquiry, feel free to contact Dylan Folsom (dfolsom@princeton.edu)

## Dependencies 
Required for analysis scripts:
 - `SatGen` (available via [anaconda](https://anaconda.org/conda-forge/satgen) or [GitHub](https://github.com/shergreen/SatGen))
 - [`astropy`](https://www.astropy.org/)
 - [`scipy`](https://scipy.org/)
 - [`numpy`](https://numpy.org)

Required for the `sample_notebook.ipynb`:
 - [`matplotlib`](https://matplotlib.org/)

## License
The contents of this repository are licensed under the [MIT license](https://spdx.org/licenses/MIT.html), and the underlying datasets used in the analysis are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
