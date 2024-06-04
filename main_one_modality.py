import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
from dataloader_mnist import get_loader
import matplotlib.pyplot as plt
from torch.optim import SGD
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from sklearn.preprocessing import normalize
import random


def set_seed(seed_value=42):
    """ Set seed for reproducibility. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def plot_cluster_map(features, labels, epoch, save_path='results/cluster_maps_10'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Cluster Map at Epoch {epoch}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, f'cluster_map_epoch_{epoch}.png'))
    plt.close()


class DeepClusterModel(nn.Module):
    def __init__(self, feature_size=512, n_clusters=50):
        super(DeepClusterModel, self).__init__()
        self.feature_extractor, feature_size = self.create_vgg16_feature_extractor()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_size, n_clusters)

    def create_vgg16_feature_extractor(self):
        vgg16 = models.vgg16_bn(weights=None)
        return_nodes = {'features.29': 'features'}  # Extract features from the last convolutional layer
        feature_extractor = create_feature_extractor(vgg16, return_nodes=return_nodes)
        feature_size = 512 
        return feature_extractor, feature_size

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)['features']
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits, features

    def reset_classifier_layer(self, n_clusters):
        self.classifier = nn.Linear(self.classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.classifier.weight)


def feature_preprocessing(features, pca_dim=150, whiten=True):
    # Perform PCA and whitening
    pca = PCA(n_components=pca_dim, whiten=whiten)
    features_pca = pca.fit_transform(features)

    # Apply L2 normalization
    features_normalized = normalize(features_pca, norm='l2', axis=1)
    return features_normalized


def train(epochs=10, n_clusters=50):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    loader = get_loader(image_dir='../image_files', batch_size=64)
    model = DeepClusterModel(n_clusters=n_clusters).to(device)
    print(f'Model device: {next(model.parameters()).device}')

    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
    writer = SummaryWriter(log_dir=f'runs/DeepCluster_VGG16_50_k')

    results = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        model.reset_classifier_layer(n_clusters)  # Reset classifier for new clusters every epoch

        # # Print feature extractor weights before extracting features
        # with torch.no_grad():
        #     for name, param in model.feature_extractor.named_parameters():
        #         if 'weight' in name:
        #             print(f'Epoch {epoch + 1}: {name} - mean {param.mean().item()}, std {param.std().item()}')

        features_list = []
        true_labels_list = []
        pseudolabels_list = []
        nmi, ari = None, None

        # Extract features and true labels (Clustering Step)
        for images, true_labels, pseudolabels, idx in tqdm(loader, desc="Processing Batches"):
            images = images.to(device)
            with torch.no_grad():
                logits, features = model(images)
                features_list.append(features.cpu().numpy())
                true_labels_list.extend(true_labels.cpu().numpy())
                pseudolabels_list.extend(pseudolabels.cpu().numpy())

        true_labels = np.array(true_labels_list)
        features = np.concatenate(features_list, axis=0)
        features_reduced = feature_preprocessing(features)

        # Cluster features using KMeans (Clustering Step)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        new_pseudolabels = kmeans.fit_predict(features_reduced)

        plot_cluster_map(features_reduced, new_pseudolabels, epoch + 1)

        loader.dataset.update_pseudolabels(new_pseudolabels)

        # Train with new pseudo-labels (Training Step)
        for images, _, batch_indices, _ in loader:
            images = images.to(device)
            batch_pseudolabels = torch.tensor(new_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)
            optimizer.zero_grad()  # Clear previous gradients
            logits, _ = model(images)
            loss = F.cross_entropy(logits, batch_pseudolabels)
            loss.backward()
            optimizer.step()  # Update model parameters

        # Compute metrics
        nmi = compute_nmi(true_labels, new_pseudolabels)
        ari = adjusted_rand_score(true_labels, new_pseudolabels)

        # Logging and monitoring
        writer.add_scalar('NMI', nmi, epoch)
        writer.add_scalar('ARI', ari, epoch)
        writer.add_scalar('Loss/train', loss.item(), epoch)
        print(f'Epoch {epoch + 1}/{epochs}, NMI: {nmi:.4f}, ARI: {ari:.4f}, Loss: {loss.item():.4f}')

        epoch_result = {
            'epoch': epoch + 1,
            'nmi': float(nmi) if nmi is not None else None,
            'ari': float(ari) if ari is not None else None,
            'loss': loss.item()
        }
        results.append(epoch_result)
        writer.flush()

    results_path = 'results/training_results_vgg16_k_50.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=4)

    writer.close()


if __name__ == '__main__':
    train()
