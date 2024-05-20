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
import matplotlib.pyplot as plt
from torch.optim import Adam
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
from dataloader import get_loader


# Define utility functions
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_nmi(true_labels, cluster_labels):
    return normalized_mutual_info_score(true_labels, cluster_labels)


def plot_cluster_map(features, labels, epoch, save_path='results/cluster_maps_5'):
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
    def __init__(self, feature_size=512, n_clusters=100):
        super(CrossModalDeepClusterModel, self).__init__()
        self.image_feature_extractor, image_feature_size = self.create_resnet18_feature_extractor()
        self.audio_feature_extractor, audio_feature_size = self.create_resnet18_feature_extractor()
        self.image_classifier = nn.Linear(image_feature_size, n_clusters)
        self.audio_classifier = nn.Linear(audio_feature_size, n_clusters)
    
    def create_resnet18_feature_extractor(self):
        resnet18 = models.resnet18(weights=None)
        return_nodes = {'avgpool': 'features'}  # Use avgpool layer for feature extraction
        feature_extractor = create_feature_extractor(resnet18, return_nodes=return_nodes)
        feature_size = resnet18.fc.in_features  # Feature size from ResNet18's last linear layer input
        return feature_extractor, feature_size

    def forward_image(self, x):
        features = self.image_feature_extractor(x)['features']
        features = features.view(features.size(0), -1)  # Flatten the features
        logits = self.image_classifier(features)
        return logits, features

    def forward_audio(self, x):
        features = self.audio_feature_extractor(x)['features']
        features = features.view(features.size(0), -1)  # Flatten the features
        logits = self.audio_classifier(features)
        return logits, features

    def reset_classifier(self, n_clusters):
        self.image_classifier = nn.Linear(self.image_classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.image_classifier.weight)
        self.audio_classifier = nn.Linear(self.audio_classifier.in_features, n_clusters).to(next(self.parameters()).device)
        nn.init.kaiming_normal_(self.audio_classifier.weight)


def get_loader(batch_size=64):
    # Placeholder function to simulate a DataLoader for images and audio
    # Replace this with the actual DataLoader code for your dataset
    pass


def feature_preprocessing(features, pca_dim=150, whiten=True):
    # Perform PCA and whitening
    pca = PCA(n_components=pca_dim, whiten=whiten)
    features_pca = pca.fit_transform(features)
  
    # Apply L2 normalization
    features_normalized = normalize(features_pca, norm='l2', axis=1)
    return features_normalized


def train_cross_modal(epochs=50, n_clusters=100):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    loader = get_loader(image_dir='./data', spectrogram_folder='./spectrograms', batch_size=64)  # Adjust the paths as necessary
    model = CrossModalDeepClusterModel(n_clusters=n_clusters).to(device)
    print(f'Model device: {next(model.parameters()).device}')

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, total_steps=epochs * len(loader))
    writer = SummaryWriter(log_dir=f'runs/CrossModalDeepCluster')

    results = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        model.reset_classifier(n_clusters)  # Reset classifier for new clusters
        optimizer = Adam(model.parameters(), lr=0.0005)  # Reinitialize optimizer
        image_features_list = []
        audio_features_list = []
        true_labels_list = []
        pseudolabels_list = []
        nmi, ari = None, None

        # Extract features and true labels for images and audio
        for images, audio, true_labels, pseudolabels, idx in tqdm(loader, desc="Processing Batches"):
            images = images.to(device)
            audio = audio.to(device)

            optimizer.zero_grad()
            image_logits, image_features = model.forward_image(images)
            audio_logits, audio_features = model.forward_audio(audio)

            image_features_list.append(image_features.detach().cpu().numpy())
            audio_features_list.append(audio_features.detach().cpu().numpy())
            true_labels_list.extend(true_labels.cpu().numpy())
            pseudolabels_list.extend(pseudolabels.cpu().numpy())

        true_labels = np.array(true_labels_list)
        image_features = np.concatenate(image_features_list, axis=0)
        audio_features = np.concatenate(audio_features_list, axis=0)

        # Feature reduction (e.g., PCA)
        image_features_reduced = feature_preprocessing(image_features)
        audio_features_reduced = feature_preprocessing(audio_features)

        # Cluster image and audio features using KMeans
        kmeans_image = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_audio = KMeans(n_clusters=n_clusters, random_state=0)
        image_pseudolabels = kmeans_image.fit_predict(image_features_reduced)
        audio_pseudolabels = kmeans_audio.fit_predict(audio_features_reduced)

        # Cross-modal pseudolabels
        # Use image clusters to generate pseudolabels for audio
        loader.dataset.update_pseudolabels(image_pseudolabels, audio_pseudolabels)

        # Train with new pseudolabels (cross-modal)
        for images, audio, _, batch_indices, _ in loader:
            images = images.to(device)
            audio = audio.to(device)

            # Using image clusters to train audio model
            batch_audio_pseudolabels = torch.tensor(image_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)
            # Using audio clusters to train image model
            batch_image_pseudolabels = torch.tensor(audio_pseudolabels[batch_indices.numpy()]).to(device, dtype=torch.long)

            optimizer.zero_grad()
            image_logits, _ = model.forward_image(images)
            audio_logits, _ = model.forward_audio(audio)
            image_loss = F.cross_entropy(image_logits, batch_image_pseudolabels)
            audio_loss = F.cross_entropy(audio_logits, batch_audio_pseudolabels)
            loss = image_loss + audio_loss
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Compute metrics
        nmi = compute_nmi(true_labels, image_pseudolabels)
        ari = adjusted_rand_score(true_labels, image_pseudolabels)

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

    results_path = 'results/training_results_cross_modal.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=4)

    writer.close()

if __name__ == '__main__':
    train_cross_modal()