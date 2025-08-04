# **Climate downscaling neural networks benchmark**


Current reppository apply neural networks for climate downscaling task on a different scales.

We heavily rely on [ClimateLearn framework](https://github.com/aditya-grover/climate-learn/tree/main) developed for this task, among others.


# Docker:

For building Docker container refer to the options available in `environements` folder. Being created, main container folder is `app`, which is used for all absolute paths in further benchmark.

The general expected structure of folders inside the container is as follows:

``` bash
.
├── app
│   ├── configs
│   |    ├── inference
│   |    ├── load
│   |    └── train
│   ├── data
│   |    ├── processed
|   |    └── raw
│   ├── environments
│   |    ├── conda
|   |    └── poetry
│   ├── notebooks
│   ├── src
│   |   ├── benchmark
│   |   ├── climatelearn
│   |   ├── data_load
|   |──README.md

```

# Config files
To tune the process, we apply [hydra package](https://hydra.cc/docs/intro/) and use config files from `configs` folder. Subfolders there are responsible for various stages:
* `configs/load` folder includes configs for loading **ERA5** (0.25 degree resolution) and **CMIP6** data. 
The loading itself is performed with  [load_era.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/data_load/load_era.py) script.

Pay attention that for loading **ERA** data with 5.625 and 2.8125 degrees we follow the ClimaeLearn procedure. You might find the original version in [download.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/climate_learn/data/download.py) script.

Also we use the elevation data loaded with [elevation.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/data_load/elevation.py) script.

Static variables (aka `constants`) require regridding for aligning with raw input data, since we use ERA5 0.25 degree values for tasks with various resolution. The procedure is implemented in [constants_regrid.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/data_load/constants_regrid.py)

* `configs/train` folder includes configs for training of 3 different setups based on **ERA5**, **CMIP6** and **E-OBS** datasets. It is supposed that user might set here the training procedure, choosing the data sources, variables used, the model and its parameters.

* `configs/inference` required if you have already trained some models (i.e. you have saved `.ckpt` files), and want to evaluate them with test data without retraining. Additionally, it includes some coordinate bounds to plot the model outputs at the area of interest (see below for details).

While loading, put all data in `data\raw` folder.

# Data preprocessing - Training - Evaluation

Please, refer to the notebook  `notebooks/Quickstart.ipynb` with the main models pipeline.

# Citation

Please cite us ... TBD as soon as the paper is accepted. Stay tuned!