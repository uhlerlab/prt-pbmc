# Notebooks

This folder contains the notebooks for segmentation, pre-processing, and training the models needed for the results in the paper.

## Segmentation and pre-processing

* [`segmentation.ipynb`](segmentation.ipynb) contains the code for segmenting the 3D microspy images using voronoi-otsu labeling. 
* [`preprocessing.ipynb`](preprocessing.ipynb) contains the code for applying quality control exclusion criteria, extracting cell crops
from the segmented images, and bundling the data for faster loading with PyTorch.

## Cell-level models

We trained cell-level classifiers to classify between time point 1 cells from cancer patients and healthy 
volunteers as well as classify between 3 cancer types and healthy. We tried various architectures. 
All models except for `cell_level_models_within_plate_same_architecture_as_mil.ipynb` are trained holding
out one plate at a time and evaluating on the held-out plate.

* [`cell_level_models_pretrained_resnet.ipynb`](cell_level_models_pretrained_resnet.ipynb)
* [`cell_level_models_resnet_from_scratch.ipynb`](cell_level_models_resnet_from_scratch.ipynb)
* [`cell_level_models_chrometric.ipynb`](cell_level_models_chrometric.ipynb)
* [`cell_level_models_pretrained_dino.ipynb`](cell_level_models_pretrained_dino.ipynb)
* [`cell_level_models_same_architecture_as_mil.ipynb`](cell_level_models_same_architecture_as_mil.ipynb)
* [`cell_level_models_within_plate_same_architecture_as_mil.ipynb`](cell_level_models_within_plate_same_architecture_as_mil.ipynb)

## Trajectory scores

Using the healthy vs. cancer cell-level classifiers we compute "similarity to healthy" scores for each time point 
to create patient trajectories and group them into 3 classes.

* [`trajectory_scores.ipynb`](trajectory_scores.ipynb)

## Multiple instance learning (MIL)

We used MIL to classify basgs of cells from the same patient for several tasks:

* [`healthy_vs_cancer_mil.ipynb`](healthy_vs_cancer_mil.ipynb) contains the code for training MIL models to classify
  between time point 1 cells from cancer patients and healthy volunteers. 
* [`cancer_type_mil.ipynb`](cancer_type_mil.ipynb) contains the code for training MIL models to classify between
  time point 1 cells from cancer patients and healthy volunteers while further classifying between 3 cancer types. 
* [`head_neck_trajectory_mil.ipynb`](head_neck_trajectory_mil.ipynb) contains the code for training MIL models to
  classify between time point 1 cells from Head & Neck cancer patients with "low" and "up" trajectories, i.e,
  predicting whether they will return to a state similar to healthy after the therapy based on cell images before
  the therapy. 
* [`all_trajectory_mil.ipynb`](all_trajectory_mil.ipynb) contains the code for training MIL models to
  classify between time point 1 cells from all cancer patients in plates 2-14 with "low" and "up" trajectories, i.e,
  predicting whether they will return to a state similar to healthy after the therapy based on cell images before
  the therapy. 
