# Deep learning-based analysis reveals patient-level cancer therapy trajectories using single-cell PBMC chromatin images  

This repository contains code for the paper "Deep learning-based analysis reveals patient-level cancer therapy trajectories using single-cell PBMC chromatin images" which analyzes PBMC chromatin images from 5 timepoints from patients undergoing proton radiation therapy and healthy volunteers to create patient trajectories and associate these with therapy outcomes.

## Data

The dataset used in this project can be downloaded at TODO.

## Repository overview

* `notebooks` contains jupyter notebooks for segmenting and pre-processing the dataset and training models used in the paper's results. See [`notebooks/README.md`](notebooks/README.md) for further details.
* `figure_notebooks` contains jupyter notebooks to reproduce the paper's main and supplementary figures. See [`figure_notebooks/README.md`](figure_notebooks/README.md) for further details.
* `scripts` contains scripts for randomizing the plate layouts and extracting chrometric features from the pre-processed images. See [`scripts/README.md`](scripts/README.md) for further details.
* `meta` contains select metadata needed for the figures and plate layout generation.
* `foundation_models` contains loaders and vendored model code for the frozen foundation-model
  feature extractors (DinoBloom, SubCell) used in one supplementary figure.

The `.py` files at the top level are the shared modules the notebooks import, not runnable scripts:

* `data.py` loads and merges the pre-bundled per-plate images, metadata, and optional features.
* `models.py` contains the cell-level classifiers and the multiple instance learning models.
* `training.py` contains the training loops.
* `eval.py` contains evaluation and clustering helpers.
* `util.py` contains small shared helpers.

## Pipeline order

1. [`notebooks/segmentation.ipynb`](notebooks/segmentation.ipynb) to segment the 3D microscopy images.
2. [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb) to apply quality control, extract cell
   crops, and bundle the data.
3. `scripts/extract_chrometric.py` if you want the chrometric features.
4. The cell-level model and MIL notebooks to train the models.
5. [`notebooks/trajectory_scores.ipynb`](notebooks/trajectory_scores.ipynb) to compute the trajectories
   from the healthy vs. cancer classifiers. This has to be run before the MIL trajectory notebooks, which
   use the trajectory groups as labels.
6. [`figure_notebooks`](figure_notebooks) to reproduce the figures.

## Dependencies:
**Python:**

This repository was developed using Python 3.9. You can use [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to create a virtual environment for a specific Python version. Additional required packages are listed in [`requirements.txt`](requirements.txt) and can be installed using the following command:
```
pip install -r requirements.txt
```
Installing dependencies can take a few minutes or up to an hour dependending on how many packages need to be downloaded rather than reusing cached versions.

A few steps need additional packages that we installed in separate environments, as they conflict with
the requirements above:

* Segmentation and pre-processing ([`notebooks/segmentation.ipynb`](notebooks/segmentation.ipynb),
  [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb)) additionally need `nd2` to read the
  raw images, and segmentation needs `pyclesperanto-prototype` for the voronoi-otsu labeling. From version
  `0.24.5` on the latter requires `numpy` below version 2.
* Chrometric feature extraction ([`scripts/extract_chrometric.py`](scripts/extract_chrometric.py)) needs
  `nmco` from the [chrometrics](https://github.com/GVS-Lab/chrometrics) repository, which is not on PyPI.
  Clone it and point the `sys.path.append` at the top of the script at your copy.
* The frozen foundation-model features
  ([`notebooks/cell_level_models_pretrained_foundation.ipynb`](notebooks/cell_level_models_pretrained_foundation.ipynb))
  need `transformers==4.45.1`, the version SubCell requires.

**Operating system and hardware:**

We developed this code on a machine running Rocky Linux 8.8 (Green Obsidian) and equipped with an NVIDIA RTX A6000 GPU.

## Citation
```
TODO
```

---

*Claude Code assisted with the analysis code for the revisions to this work.*

