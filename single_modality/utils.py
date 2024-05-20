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
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # Ensures that CUDA convolution is deterministic
    torch.backends.cudnn.benchmark = False


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target, _ in loader:
            data = data.to(device)
            output = model(data)
            features.extend(output['out_features'].cpu().numpy())
            labels.extend(target.numpy())
    return np.array(features), np.array(labels)


def perform_clustering(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_, kmeans.cluster_centers_


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def plot_filter_activations(model, layer_name):
    # Assume model is a dictionary of layers
    layer = dict(model.named_modules())[layer_name]
    weights = layer.weight.data.cpu().numpy()
    n_filters = weights.shape[0]

    # Plotting the first 10 filters
    fig, axes = plt.subplots(1, min(n_filters, 10), figsize=(20, 5))
    for i, ax in enumerate(axes):
        # Assuming the first channel for visualization
        ax.imshow(weights[i][0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i}')
    plt.show()


def maximize_activation(model, layer_name, filter_index, num_iterations=30, upscaling_steps=3, upscaling_factor=1.2):
    # Initialize random noise
    img = torch.randn(1, 3, 56, 56, device='cuda', requires_grad=True)
    optimizer = Adam([img], lr=0.05)

    # Register hook for the layer output
    layer_output = dict(model.named_modules())[layer_name].output

    for _ in range(upscaling_steps):
        img.data.clamp_(0, 1)  # Normalize image
        for _ in range(num_iterations):
            optimizer.zero_grad()
            model(img)
            # Negative loss for maximization
            loss = -layer_output[0, filter_index].mean()
            loss.backward()
            optimizer.step()

        # Upscale image
        up_size = (int(img.shape[2] * upscaling_factor),
                   int(img.shape[3] * upscaling_factor))
        img = F.interpolate(img, size=up_size, mode='bilinear',
                            align_corners=False).detach_().requires_grad_()

    # Convert tensor to PIL Image for visualization
    img = img.detach().cpu().squeeze(0)
    img = to_pil_image(img)
    plt.imshow(img)
    plt.title(f'Maximized Activation for Filter {filter_index}')
    plt.axis('off')
    plt.show()
    return img


def plot_metrics(epoch, nmis, reassignments, maps):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(range(epoch+1), nmis, label='NMI')
    plt.title('NMI over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('NMI')

    plt.subplot(132)
    plt.plot(range(epoch+1), reassignments, label='Reassignments')
    plt.title('Cluster Reassignments')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')

    plt.subplot(133)
    plt.plot(range(epoch+1), maps, label='mAP')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')

    plt.tight_layout()
    plt.show()


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def feature_preprocessing(features, pca_dim=150, whiten=True):
    # Perform PCA and whitening
    pca = PCA(n_components=pca_dim, whiten=whiten)
    features_pca = pca.fit_transform(features)
    
    # Apply L2 normalization
    features_normalized = normalize(features_pca, norm='l2', axis=1)
    return features_normalized


def plot_cluster_map(features, labels, epoch, save_path='results/cluster_maps_1000k'):
    """
    Plots and saves a 2D scatter plot of clustered data.

    Args:
    features (numpy.ndarray): High-dimensional feature array.
    labels (numpy.ndarray): Cluster labels for each feature.
    epoch (int): Current epoch number, used for file naming.
    save_path (str): Directory to save the plots.
    """
    # Ensuring the saving directory exists
    os.makedirs(save_path, exist_ok=True)
  
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette='viridis', s=100, alpha=0.6)
    plt.title(f'Cluster Map at Epoch {epoch}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    
    # Save the plot
    file_name = f'cluster_map_epoch_{epoch}.png'
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()  # Close the plot to free up memory

    print(f'Cluster map saved: {os.path.join(save_path, file_name)}')


def plot_umap(features, labels, cluster_ids, epoch, specific_label_index, save_path='results/umap_visuals'):
    os.makedirs(save_path, exist_ok=True)
    umap_base = umap.UMAP(n_components=2)
    mapped_features = umap_base.fit_transform(features)

    specific_label_mask = labels == specific_label_index
    other_label_mask = ~specific_label_mask

    plt.figure(figsize=(15, 8))

    # Plot other labels in red
    plt.scatter(mapped_features[other_label_mask, 0], mapped_features[other_label_mask, 1], color='red', s=0.2, label='Other Labels')

    # Plot specific label in green
    plt.scatter(mapped_features[specific_label_mask, 0], mapped_features[specific_label_mask, 1], color='green', s=0.8, label=f'Label {specific_label_index}')

    # Color by cluster
    sns.scatterplot(x=mapped_features[:, 0], y=mapped_features[:, 1], hue=cluster_ids, palette='viridis', s=0.5, alpha=0.6)

    plt.title('UMAP Visualization with Specific Label Highlight')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend()
    plt.grid(True)
    file_name = f'cluster_umap_epoch_{epoch}.png'
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()  # Close the plot to free up memory

    print(f'Cluster map saved: {os.path.join(save_path, file_name)}')
