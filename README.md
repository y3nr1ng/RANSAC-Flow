# RANSAC-Flow
An attempt to reimplement RANSAC-Flow, a generic two-stage image alignment process.

TODO some description about this project and its origin

## Quick Start
### Prerequisite
TODO package requirement

If you already have `conda` installed, we have prepared an environment file that can help you get up and running. After cloning the repository, navigate to project root, and do
```
conda env create -f conda-dev.yml
```
This will create a new environment called `ransacflow`.

### Demo
Please be aware that this project is designed to work under *editable mode*!
After cloning this repository, navigate to project root (you should see `setup.py`) and
do
```
pip install -e .
```
this will install all needed dependency, and soft link this project as an installed package.
You can now use this repository by
```
import ransacflow
```
or continue with our demo notebooks under `notebooks/`.

It needs to download files to get it up and running, and these destination folder are parallel to the project folder.

TODO what notebook to run
## Development
### Dataset
TODO how to download dataset

### Training
TODO train on/how

### Evaluation

## Acknowledgement
* The original RANSAC-Flow [publication](https://arxiv.org/abs/2004.01526) and [repository](https://github.com/XiSHEN0220/RANSAC-Flow).
