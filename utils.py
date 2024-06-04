# utils.py
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from sklearn.decomposition import PCA
import os
import seaborn as sns
from sklearn.preprocessing import normalize
import umap
import random


def set_seed(seed_value=42):
    """ Set seed for reproducibility. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def feature_preprocessing(features, pca_dim=150, whiten=True):
    # Perform PCA and whitening
    pca = PCA(n_components=pca_dim, whiten=whiten)
    features_pca = pca.fit_transform(features)
    
    # Apply L2 normalization
    features_normalized = normalize(features_pca, norm='l2', axis=1)
    return features_normalized


def plot_cluster_map(features, labels, epoch, save_path='results/cluster_maps_5'):

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette='viridis', s=100, alpha=0.6)
    plt.title(f'Cluster Map at Epoch {epoch}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    
    file_name = f'cluster_map_epoch_{epoch}.png'
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

    print(f'Cluster map saved: {os.path.join(save_path, file_name)}')

