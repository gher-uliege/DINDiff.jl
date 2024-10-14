[![DOI](https://zenodo.org/badge/832065698.svg)](https://zenodo.org/doi/10.5281/zenodo.13165362)

# DINDiff

This is the source code for the manuscript:

Alexander Barth, Julien Brajard, Aida Alvera-Azcárate, Bayoumy Mohamed, Charles Troupin, and Jean-Marie Beckers,
Ensemble reconstruction of missing satellite data using a denoising diffusion model: application to chlorophyll a concentration in the Black Sea, 2024
https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1075/

Accepted in Ocean Science



https://github.com/user-attachments/assets/2eb271a1-de9e-4e71-8bf6-82f9e565ba83




## Installation

The code is tested with Julia 1.9 and the julia package listed in [`Project.toml`](Project.toml).
Information about julia's package manager is available: https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project

After downloading the source code, all dependencies of the project can be installed with:

```julia
using Pkg
Pkg.activate("/path/to/DINDiff.jl")
Pkg.instantiate()
```

where `/path/to/DINDiff.jl` is the path to the folder containing the file `Project.toml`.

## Code

* [`src/diffusion_model.jl`](src/diffusion_model.jl): common function for defining the model and data loading
* [`src/training.jl`](src/training.jl): script for training
* [`src/inference.jl`](src/inference.jl): script for inference

## Data files

A minimal NetCDF file for training has the following structure:

```
netcdf patches_64_64_0.8 {
dimensions:
	lon = 64 ;
	lat = 64 ;
	time = UNLIMITED ; // (851926 currently)
variables:
	double lon(time, lon) ;
	double lat(time, lat) ;
	double time(time) ;
		time:units = "days since 1970-01-01" ;
	float CHL(time, lat, lon) ;
		CHL:_FillValue = -9999.f ;
}
```

The variable can be of course for every application. 
