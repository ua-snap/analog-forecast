# Analog Forecast

Make forecasts from similar meteorological conditions (analogs) using ERA5. 

## Instructions

1. Create the conda environment via `conda env create -f environment.yml`
2. Activate the conda environment (`conda activate analog-forecast`)
3. Store the path to the ERA5 daily data in the `DATA_DIR` environment variable: `export DATA_DIR=/atlas_scratch/kmredilla/analog_forecast/`
4. Run the analog_forecast.py script with the desired options:

```
python analog_forecast.py -v sst -d 2021-12-01 -s alaska
```

## Contents

### `skill_profiling/`

The contents of this subfolder are for evaluating the skill of the analog forecast method. To work with the code there, set the `PROJECT_DIR` environment variable to the path of this repo on the local filesystem and add it to the `PYTHONPATH` variable, e.g. run the following from the directory containing this file:

```
export PROJECT_DIR=$PWD
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
```
