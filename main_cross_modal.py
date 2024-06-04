import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import matplotlib.pyplot as plt
from torch.optim import SGD
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from dataloader import get_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def plot_cluster_map(features, labels, epoch, save_path='results/cluster_maps_100'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Cluster Map at Epoch {epoch}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, f'cluster_map_epoch_{epoch}.png'))
    plt.close()


class CrossModalDeepClusterModel(nn.Module):
    def __init__(self, feature_size=512, n_clusters=10):
        super(CrossModalDeepClusterModel, self).__init__()
        self.image_feature_extractor, image_feature_size = self.create_vgg16_feature_extractor()
        self.audio_feature_extractor, audio_feature_size = self.create_vgg16_feature_extractor()
        self.image_classifier = nn.Linear(image_feature_size, n_clusters)
        self.audio_classifier = nn.Linear(audio_feature_size, n_clusters)
    
    def create_vgg16_feature_extractor(self):
        vgg16 = models.vgg16_bn(weights=None)
        return_nodes = {'features.29': 'features'}
        feature_extractor = create_feature_extractor(vgg16, return_nodes=return_nodes)
        feature_size = 512
        return feature_extractor, feature_size

    def forward_image(self, x):
        features = self.image_feature_extractor(x)['features']
        features = nn.AdaptiveAvgPool2d((1, 1))(features)
        features = features.view(features.size(0), -1)
        logits = self.image_classifier(features)
        return logits, features

    def forward_audio(self, x):
        features = self.audio_feature_extractor(x)['features']
        features = nn.AdaptiveAvgPool2d((1, 1))(features)
        features = features.view(features.size(0), -1)
        logits = self.audio_classifier(features)
        return logits, features

    def reset_classifier_layer(self, n_clusters):
        self.image_classifier = nn.Linear(self.image_classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.image_classifier.weight)
        self.audio_classifier = nn.Linear(self.audio_classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.audio_classifier.weight)


def feature_preprocessing(features, pca_dim=150, whiten=True):
    # Perform PCA and whitening
    pca = PCA(n_components=pca_dim, whiten=whiten)
    features_pca = pca.fit_transform(features)
  
    # Apply L2 normalization
    features_normalized = normalize(features_pca, norm='l2', axis=1)
    return features_normalized


def train_cross_modal(epochs=10, n_clusters=100):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    loader = get_loader(image_dir='../image_files', audio_dir='../audio_files', batch_size=64)
    model = CrossModalDeepClusterModel(n_clusters=n_clusters).to(device)
    print(f'Model device: {next(model.parameters()).device}')

    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
    writer = SummaryWriter(log_dir=f'runs/CrossModalDeepCluster_100')

    results = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        model.reset_classifier_layer(n_clusters)  # Reset classifier for new clusters
        image_features_list = []
        audio_features_list = []
        true_labels_list = []
        image_pseudolabels_list = []
        audio_pseudolabels_list = []
        nmi, ari = None, None

        # Extract features and true labels for images and audio
        for images, audio, true_labels, _, idx in tqdm(loader, desc="Processing Batches"):
            images = images.to(device)
            audio = audio.to(device)

            with torch.no_grad():
                _, image_features = model.forward_image(images)
                _, audio_features = model.forward_audio(audio)

            image_features_list.append(image_features.cpu().numpy())
            audio_features_list.append(audio_features.cpu().numpy())
            true_labels_list.extend(true_labels.cpu().numpy())

        true_labels = np.array(true_labels_list)
        image_features = np.concatenate(image_features_list, axis=0)
        audio_features = np.concatenate(audio_features_list, axis=0)

        # Feature reduction
        image_features_reduced = feature_preprocessing(image_features)
        audio_features_reduced = feature_preprocessing(audio_features)

        # Cluster image and audio features using KMeans
        kmeans_image = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_audio = KMeans(n_clusters=n_clusters, random_state=0)
        image_pseudolabels = kmeans_audio.fit_predict(audio_features_reduced)
        audio_pseudolabels = kmeans_image.fit_predict(image_features_reduced)

        # Plot cluster maps for the image clusters
        plot_cluster_map(image_features_reduced, audio_pseudolabels, epoch + 1)
        
        # Update pseudo-labels
        loader.dataset.update_pseudolabels(image_pseudolabels, audio_pseudolabels)

        # Train with new pseudo-labels (cross-modal)
        for images, audio, _, batch_indices, _ in loader:
            images = images.to(device)
            audio = audio.to(device)

            # Using audio clusters to train the image model
            batch_image_pseudolabels = torch.tensor(audio_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)
            optimizer.zero_grad()
            image_logits, _ = model.forward_image(images)
            image_loss = F.cross_entropy(image_logits, batch_image_pseudolabels)
            image_loss.backward()
            optimizer.step()

            # Using image clusters to train the audio model
            batch_audio_pseudolabels = torch.tensor(image_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)
            optimizer.zero_grad()
            audio_logits, _ = model.forward_audio(audio)
            audio_loss = F.cross_entropy(audio_logits, batch_audio_pseudolabels)
            audio_loss.backward()
            optimizer.step()

        # Compute metrics
        nmi = compute_nmi(true_labels, audio_pseudolabels)
        ari = adjusted_rand_score(true_labels, audio_pseudolabels)

        # Logging and monitoring
        writer.add_scalar('NMI', nmi, epoch)
        writer.add_scalar('ARI', ari, epoch)
        writer.add_scalar('Loss/train', (image_loss.item() + audio_loss.item()) / 2, epoch)
        print(f'Epoch {epoch + 1}/{epochs}, NMI: {nmi:.4f}, ARI: {ari:.4f}, Loss: {(image_loss.item() + audio_loss.item()) / 2:.4f}')

        epoch_result = {
            'epoch': epoch + 1,
            'nmi': float(nmi) if nmi is not None else None,
            'ari': float(ari) if ari is not None else None,
            'loss': (image_loss.item() + audio_loss.item()) / 2
        }
        results.append(epoch_result)
        writer.flush()

    results_path = 'results/training_results_cross_modal_100.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=4)

    writer.close()


if __name__ == '__main__':
    train_cross_modal()
