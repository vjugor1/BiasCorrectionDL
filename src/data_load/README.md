# Loading ERA with weather bench
Requires congig file defyning variables, `year_start` and `year_end`as well as the path to load data to
# Loading CMIP. Multiple options
* Copernicus API. `load_cmip.py` heavily relies on cdsapi available with credentials written in `.cdsapirc` file in `$ROOT` dir. For valid keywords, check the [Copernicus link](https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=form). The data available with vpn
* Load required wget with [ESGF database](https://aims2.llnl.gov/search) and run `bash <wget.sh> -s`. With option `-s` no authorization is required.
* Load with esgf-pyclient environment implemented in [WindUtils](https://github.com/makboard/WindUtils/tree/main/CMIP).