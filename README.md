# Dynamic CT Reconstruction from Limtied Views with Implicit Neural Representations and Parametric Motion Fields

## Description
This repo implements code from the ICCV 2021 paper *Dynamic CT Reconstruction from Limited Views with Implicit Neural 
Representations and Parametric Motion Fields*. The preprint can be found here https://arxiv.org/abs/2104.11745

## Installation
The code is easiest to install with a fresh conda environment. 

1) Clone the repo `https://github.com/awreed/DynamicCTReconstruction`
2) Update conda `conda update conda --all`
3) Update anaconda `conda update anaconda`
4) Create a new virtual environment `conda create --name <env_name> python=3.8`
5) Install the requrirements listed in *requirements.txt* `conda install -n <env_name> requirements.txt`

## Running the Code
The pipeline ingests a *.ini* file that defines necessary parameters. An example script for reconstructing one of the 
aluminum objects is provided in *alum_config.ini*.

To run the reconstruction pipeline with this script `./run_pipeline.sh`

The code should run for 800 epochs. Code output is saved under /data/alum_s03_001_save_data/ and contains *.mp4* movies
of the object reconstruction at each 100 epochs (e.g. *static_600.mp4*) . Additionally, a ground truth rendering for the 
reconstruction (10 frames shown) can be found in */data/alum_s03_001/S03_001static.mp4*