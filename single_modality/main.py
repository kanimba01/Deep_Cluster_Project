import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
from dataloader import get_loader
import matplotlib.pyplot as plt
from torch.optim import Adam
from utils import plot_cluster_map, apply_pca, compute_nmi, set_seed

import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


class DeepClusterModel(nn.Module):
    def __init__(self, feature_size=512):
        super(DeepClusterModel, self).__init__()
        self.feature_extractor, feature_size = self.create_resnet18_feature_extractor()
        self.classifier = nn.Linear(feature_size, 100)

    def create_resnet18_feature_extractor(self):
        resnet18 = models.resnet18(weights=False)
        return_nodes = {'avgpool': 'features'}  # Use avgpool layer for feature extraction
        feature_extractor = create_feature_extractor(resnet18, return_nodes=return_nodes)
        feature_size = resnet18.fc.in_features 
        return feature_extractor, feature_size

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)['features']
        features = features.view(features.size(0), -1)  # Flatten the features
        logits = self.classifier(features)
        return logits, features

    def reset_classifier(self, n_clusters):
        self.classifier = nn.Linear(self.classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.classifier.weight)


def train(epochs=50, n_clusters=1000):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    loader = get_loader(batch_size=64)  
    model = DeepClusterModel().to(device)
    print(f'Model device: {next(model.parameters()).device}')

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, total_steps=epochs * len(loader))
    writer = SummaryWriter(log_dir=f'runs/DeepCluster_1000k_one_modality')

    results = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        model.reset_classifier(n_clusters)  # Reset classifier for new clusters
        optimizer = Adam(model.parameters(), lr=0.0005)  # Reinitialize optimizer
        features_list = []
        true_labels_list = []
        pseudolabels_list = []
        nmi, ari = None, None

        # Extract features and true labels
        for images, true_labels, pseudolabels, idx in tqdm(loader, desc="Processing Batches"):
            images = images.to(device)
            # print(f'Images device: {images.device}, True labels device: {true_labels.device}, Pseudo labels device: {pseudolabels.device}')  # Debug print

            optimizer.zero_grad()
            logits, features = model(images)
            # print(f'Logits device: {logits.device}, Features device: {features.device}')  # Debug print

            features_list.append(features.detach().cpu().numpy())
            true_labels_list.extend(true_labels.cpu().numpy())
            pseudolabels_list.extend(pseudolabels.cpu().numpy())

        true_labels = np.array(true_labels_list)
        features = np.concatenate(features_list, axis=0)
        features_reduced = apply_pca(features)

        # Cluster features using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        new_pseudolabels = kmeans.fit_predict(features_reduced)

        if epoch % 5 == 0:
            plot_cluster_map(features_reduced, new_pseudolabels, epoch + 1)

        loader.dataset.update_pseudolabels(new_pseudolabels)

        # Train with new pseudolabels
        for images, _, batch_indices, _ in loader:
            images = images.to(device)
            batch_pseudolabels = torch.tensor(new_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = F.cross_entropy(logits, batch_pseudolabels)
            loss.backward()
            optimizer.step()

        scheduler.step()

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

    results_path = 'results/training_results_1000k.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=4)

    writer.close()


if __name__ == '__main__':
    train()
