# **Climate downscaling neural networks benchmark**


Current reppository apply neural networks for climate downscaling task on a different scales.

We heavily rely on [ClimateLearn framework](https://github.com/aditya-grover/climate-learn/tree/main) developed for this task, among others.


# Config files
To tune the process, we apply [hydra package](https://hydra.cc/docs/intro/) and use config files from `configs` folder. Subfolders there are responsible for various stages:
* `configs/load` folder includes configs for loading **ERA5** (0.25 degree resolution) and **CMIP6** data. 

Pay attention that for loading **ERA** data with 5.625 and 2.8125 degrees we follow the ClimaeLearn procedure. You might find the original version in [download.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/climate_learn/data/download.py) script.
* `configs/train` folder includes configs for training of 3 different setups based on **ERA5**, **CMIP6** and **E-OBS** datasets. It is supposed that user might set here the training procedure, choosing the data sources, variables used, the model and its parameters.
* `configs/inference` required if you have already trained some models (i.e. you have saved `.ckpt` files), and want to evaluate them with test data without retraining. Also, it includes some coordinate bounds to plot the model outputs at the area of interest (see below for details).

Also we use the evaluation data loaded with [evaluation.py](https://github.com/vjugor1/BiasCorrectionDL/blob/main/src/data_load/evaluation.py) script.

# Data preprocessing
We have slightly modified the original ClimateLearn procedure to preprocess raw data, but it remains largely unchanged. To execute it, follow the pipeline in the `notebooks/process_raw.ipynb` notebook to convert the data to the necessary format.

# Training
Ensure that you have completed the preprocessing of the raw data, as this step requires the preprocessed version.

This stage refers to the configuration files from the `configs/train` folder.

The example for training the model from console:

```
python era5_era5_dl.py
```
Single run performs the training of the single model. To train another architecture/parameters, edit the configuration file and run training again.

# Evaluation

Once you have trained the model you could refer to the checkpoint and test that model with test data. Please, specify the details in `.yaml` file at `configs/inference` folder. As a part of the pipeline you could collect the desired metrics of desired model (list everything the config file) with `save_metrics.py` script. Also, one might plot output of the model with `plots.py` script. The bounds of area of interest could be defined in `configs/inference` folder. Run procedure like this:
```
python plots.py
```
Pay attention to the necessity to define config path in that script explicitly.
```

# Docker:

For building Docker container refer to the options available in `environements` folder.

# Citation

Please city us ... TBD as soon as the paper is accepted. Stay tuned!