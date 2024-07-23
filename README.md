# DINDiff


This is the source code for the manuscript:

Alexander Barth, Julien Brajard, Aida Alvera-Azcárate, Bayoumy Mohamed, Charles Troupin, and Jean-Marie Beckers,
Ensemble reconstruction of missing satellite data using a denoising diffusion model: application to chlorophyll a concentration in the Black Sea, 2024
https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1075/

Submitted to Ocean Science


The code is tested with Julia 1.9 and the julia package listed in [`Project.toml`](Project.toml).
Information about julia's package manager is available: https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project

## Code

* [`src/diffusion_model.jl`](src/diffusion_model.jl): common function for defining the model and data loading
* [`src/training.jl`](src/training.jl): script for training
* [`src/inference.jl`](src/inference.jl): script for inference

