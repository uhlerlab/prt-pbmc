# Deep learning-based analysis reveals patient-level proton radiation therapy trajectories using single-cell PBMC chromatin images 

This repository contains code for the paper "Deep learning-based analysis reveals patient-level proton radiation therapy trajectories using single-cell PBMC chromatin images" which analyzes PBMC chromatin images from 5 timepoints from patients undergoing proton therapy and healthy volunteers to create patient trajectories and associate these with therapy outcomes.

## Data

The dataset used in this project can be downloaded at TODO.

## Repository overview

* `notebooks` contains jupyter notebooks for segmenting and pre-processing the dataset and training models used in the paper's results. See [`notebooks/README.md`](notebooks/README.md) for further details.
* `figure_notebooks` contains jupyter notebooks to reproduce the paper's main and supplementary figures. See [`figure_notebooks/README.md`](figure_notebooks/README.md) for further details.
* `scripts` contains scripts for randomizing the plate layouts and extracting chrometric features from the pre-processed images. See [`scripts/README.md`](scripts/README.md) for further details.
* `meta` contains select metadata needed for the figures and plate layout generation.

## Dependencies:
**Python:**

This repository was developed using Python 3.9. You can use [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to create a virtual environment for a specific Python version. Additional required packages are listed in [`requirements.txt`](requirements.txt) and can be installed using the following command:
```
pip install -r requirements.txt
```
Installing dependencies can take a few minutes or up to an hour dependending on how many packages need to be downloaded rather than reusing cached versions.

**Operating system and hardware:**

We developed this code on a machine running Rocky Linux 8.8 (Green Obsidian) and equipped with an NVIDIA RTX A6000 GPU.

## Citation
```
TODO
```

