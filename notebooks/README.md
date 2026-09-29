# Notebooks

This folder contains the notebooks for segmentation, pre-processing, and training the models needed for the results in the paper.

## Segmentation and pre-processing

* [`segmentation.ipynb`](segmentation.ipynb) contains the code for segmenting the 3D microscopy images using voronoi-otsu labeling. 
* [`preprocessing.ipynb`](preprocessing.ipynb) contains the code for applying quality control exclusion criteria, extracting cell crops
from the segmented images, and bundling the data for faster loading with PyTorch.

## Cell-level models

We trained cell-level classifiers to classify between time point 1 cells from cancer patients and healthy 
volunteers as well as classify between 3 cancer types and healthy. We tried various architectures. 
All models except for `cell_level_models_within_plate_same_architecture_as_mil.ipynb` are trained holding
out one plate at a time and evaluating on the held-out plate. Because each healthy volunteer appears on
two adjacent plates, the held-out plate's patients are also dropped from the other plates in the training
set, so that no patient is seen during both training and evaluation.

* [`cell_level_models_pretrained_resnet.ipynb`](cell_level_models_pretrained_resnet.ipynb)
* [`cell_level_models_resnet_from_scratch.ipynb`](cell_level_models_resnet_from_scratch.ipynb)
* [`cell_level_models_chrometric.ipynb`](cell_level_models_chrometric.ipynb)
* [`cell_level_models_pretrained_dino.ipynb`](cell_level_models_pretrained_dino.ipynb)
* [`cell_level_models_pretrained_foundation.ipynb`](cell_level_models_pretrained_foundation.ipynb) uses
  DinoBloom and SubCell frozen as feature extractors, i.e., only the feature extractor differs from the
  pre-trained DINO models.
* [`cell_level_models_same_architecture_as_mil.ipynb`](cell_level_models_same_architecture_as_mil.ipynb)
* [`cell_level_models_within_plate_same_architecture_as_mil.ipynb`](cell_level_models_within_plate_same_architecture_as_mil.ipynb)

## Embedding and clustering

* [`umap_and_leiden_clustering.ipynb`](umap_and_leiden_clustering.ipynb) contains the code for computing the
  UMAP embedding of the pre-trained ResNet features of time point 1 cancer and healthy cells and clustering
  them with Leiden, including the silhouette score tuning used to pick the resolution.

## Trajectory scores

Using the healthy vs. cancer cell-level classifiers we compute "similarity to healthy" scores for each time point 
to create patient trajectories and group them into 3 classes.

* [`trajectory_scores.ipynb`](trajectory_scores.ipynb)

## Multiple instance learning (MIL)

We used MIL to classify bags of cells from the same patient for several tasks:

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
* [`cancer_type_sample_size_mil.ipynb`](cancer_type_sample_size_mil.ipynb) contains the code for retraining
  the cancer type MIL models while shrinking only the Head & Neck training set, to test how much of the
  weaker classification of the other cancer types is explained by their smaller number of samples and plates.
* [`mil_aggregator_comparison.ipynb`](mil_aggregator_comparison.ipynb) contains the code for training the
  cancer type MIL models with the gated attention aggregator replaced by a transformer or a mixture of
  aggregators, keeping the cell encoder and the cross-validation the same.

## Patient-level baselines and robustness checks

* [`simple_patient_level_models.ipynb`](simple_patient_level_models.ipynb) contains the code for classifying
  patients with logistic regression and random forests on the median of their pre-trained ResNet cell
  embeddings, as a simpler alternative to aggregating cell-level predictions.
* [`seed_stability_healthy_cancer.ipynb`](seed_stability_healthy_cancer.ipynb) contains the code for
  retraining the healthy vs. cancer cell-level and MIL models across several training initialization seeds
  to check how much the results vary with the initialization.
