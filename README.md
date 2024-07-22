# DINDiff


This is the source code for the manuscript:

Alexander Barth, Julien Brajard, Aida Alvera-Azcárate, Bayoumy Mohamed, Charles Troupin, and Jean-Marie Beckers,
Ensemble reconstruction of missing satellite data using a denoising diffusion model: application to chlorophyll a concentration in the Black Sea, 2024
https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1075/

Submitted to Ocean Science


The code is tested with Julia 1.9 and the julia package listed in `Project.toml`.
Information about julia's package manager is available: https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project

## Code

* `src/diffusion_model.jl`: model and data loading
* `src/training.jl`: script for training
* `src/inference.jl`: script for inference

