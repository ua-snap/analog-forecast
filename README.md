# Analog Forecast

Make forecasts from similar meteorological conditions (analogs) using ERA5. 

## Instructions

To run the basic analog forecast program from the command line:

1. Create the conda environment via `conda env create -f environment.yml`
2. Activate the conda environment (`conda activate analog-forecast`)
3. Store the path to the ERA5 daily data in the `DATA_DIR` environment variable: `export DATA_DIR=/atlas_scratch/kmredilla/analog_forecast/`
4. Store the path to the root direcotry of the repository in the `PROJECT_DIR` environment variable: `export PROJECT_DIR=$PWD`
5. Run the analog_forecast.py script with the desired options:

```
python analog_forecast.py -v sst -d 2021-12-01 -s alaska
```

### Spatial domains

The following four spatial domains are available for both analog search and forecast (bbox bounds in W,S,E,N):

```
"alaska": (-180, 44, -125, 76)
"northern_hs": (-180, 0, 180, 90)
"panarctic": (-180, 55, 180, 90)
"north_pacific": (120, 0, 240, 90)
```

## Contents

`analog_forecast.py`: this is the main module for generating the analog forecast. It can be called from the command line or imported and run in a separate python environment, such as a notebook.
`config.py`: config file with some useful constants
`luts.py`: like config but for lookup tables (dicts)
`qc.ipynb`: notebook for quality checking different parts of the algorithm
`run_forecast.ipynb`: notebook for running the analog forecast program. Includes code for downloading ERA5 data if you want to generate a forecast for a date not included in the historical data archive, and interactive visualization of forecast error if forecast dates are found in the historical archive.

## `scripts/`

Scripts and modules for preparing the data. Use these scripts to download the necessary data. To actually run any of these scripts, set the `PROJECT_DIR` environment variable to the path of this repo on the local filesystem and add it to the `PYTHONPATH` variable. E.g. run the following from the directory containing this file:

```
export PROJECT_DIR=$PWD
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
```

### `skill_profiling/`

The contents of this subfolder are for evaluating the skill of the analog forecast method. The project directory needs to be in the PYTHONPATH variable as well for working in here. 

Also, if working from Atlas or Chinook, you will need a script for initializing `conda` in the shell generated by the slurm scheduler. So do this:

```
export CONDA_INIT_SCRIPT=/path/to/conda_init.sh
```

Where the `conda_init.sh` looks something like this:

```
__conda_setup="$('/home/UA/kmredilla/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/UA/kmredilla/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/UA/kmredilla/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/UA/kmredilla/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
```