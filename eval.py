import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scanpy
import anndata as ad
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader


def cluster_leiden(x, umap_df, resolution=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], seed=52352):
    cmap = mpl.colormaps['tab10']
    all_cluster_labels = {}

    for res in tqdm(resolution):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        n_neighbors = 10
        n_pcs = None # TODO
        adata = ad.AnnData(x)
        scanpy.tl.pca(adata, svd_solver='arpack')
        scanpy.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        scanpy.tl.leiden(adata, resolution=res, random_state=seed)
        cluster_labels = adata.obs['leiden'].to_numpy().astype(int)
        all_cluster_labels[res] = cluster_labels

        # Plot showing the actual clusters formed
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / len(np.unique(cluster_labels)))
        scatter = ax2.scatter(
            umap_df['umap_x'], umap_df['umap_y'], marker=".", s=30, lw=0, alpha=0.7, c=cluster_labels,
            edgecolor="k", cmap=cmap, vmax=10)
        # produce a legend with the unique colors from the scatter
        legend = ax2.legend(*scatter.legend_elements(), title="Clusters")
        ax2.add_artist(legend)

        ax2.set_title(f"Visualization of the clustered data for resolution {res}")

        n_clusters = np.max(cluster_labels) + 1
        if n_clusters == 1:
            ax1.axis('off')
            continue
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(x, cluster_labels)
        print(
            "\nFor resolution =",
            res,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cmap(i)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
    return all_cluster_labels


def eval_model_on_dataset(model, dataset, device, batch_size=1024, eval=True, resnet=False):
  model = model.to(device)
  if eval:
    model.eval()
  else:
    model.train()

  torch.manual_seed(1341241)  # for when eval mode isn't used
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  preds = []
  labs = []
  for imgs, lab in loader:
    imgs = imgs.to(device)
    if resnet:
      with torch.no_grad():
        prob = model(imgs.repeat(1, 3, 1, 1).to(device))
        pred = torch.argmax(prob, dim=-1)
    else:
      with torch.no_grad():
        _, pred, _ = model.forward(imgs)
    
    pred = pred.cpu().numpy().reshape(-1)
    preds.append(pred)
    lab = lab.cpu().numpy().reshape(-1)
    labs.append(lab)

  preds = np.concatenate(preds)
  labs = np.concatenate(labs)

  return preds, labs
