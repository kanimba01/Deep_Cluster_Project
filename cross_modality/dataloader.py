import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class IndexedDataset(Dataset):
    def __init__(self, image_dataset, spectrogram_folder, image_transform=None, spectrogram_transform=None, limit=None):
        self.image_dataset = image_dataset
        self.spectrogram_folder = spectrogram_folder
        self.image_transform = image_transform
        self.spectrogram_transform = spectrogram_transform
        self.limit = limit if limit is not None else len(self.image_dataset)
        self.pseudolabels = [None] * self.limit

    def update_pseudolabels(self, new_image_pseudolabels, new_audio_pseudolabels):
        self.image_pseudolabels = new_image_pseudolabels[:self.limit]
        self.audio_pseudolabels = new_audio_pseudolabels[:self.limit]

    def __getitem__(self, index):
        image, true_label = self.image_dataset[index]
        spectrogram_path = os.path.join(self.spectrogram_folder, f'{index}.pt')
        mel_spec = torch.load(spectrogram_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.spectrogram_transform:
            mel_spec = self.spectrogram_transform(mel_spec)

        image_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        audio_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, mel_spec, true_label, image_pseudolabel, audio_pseudolabel, index

    def __len__(self):
        return self.limit


def get_image_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(28, padding=4),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_spectrogram_transform():
    return transforms.Compose([
        # Example: Normalize spectrogram values (assuming they need normalization)
        transforms.Lambda(lambda x: (x - x.mean()) / x.std())
    ])


def get_loader(image_dir, spectrogram_folder, batch_size=64, limit=None):
    image_dataset = datasets.MNIST(root=image_dir, train=True, download=True)
    image_transform = get_image_transform()
    spectrogram_transform = get_spectrogram_transform()
    dataset = IndexedDataset(image_dataset, spectrogram_folder, image_transform=image_transform, spectrogram_transform=spectrogram_transform, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class IndexedDataset(Dataset):
    def __init__(self, image_folder, spectrogram_folder, image_transform=None, spectrogram_transform=None, limit=None):
        self.image_folder = image_folder
        self.spectrogram_folder = spectrogram_folder
        self.image_transform = image_transform
        self.spectrogram_transform = spectrogram_transform
        self.image_files = sorted(os.listdir(image_folder))[:limit]
        self.limit = len(self.image_files)
        self.pseudolabels = [None] * self.limit

    def update_pseudolabels(self, new_image_pseudolabels, new_audio_pseudolabels):
        self.image_pseudolabels = new_image_pseudolabels[:self.limit]
        self.audio_pseudolabels = new_audio_pseudolabels[:self.limit]

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_path).convert('L')
        true_label = int(self.image_files[index].split('_')[0])  # Assuming filename format is 'label_idx.png'

        spectrogram_path = os.path.join(self.spectrogram_folder, f'{index}.pt')
        mel_spec = torch.load(spectrogram_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.spectrogram_transform:
            mel_spec = self.spectrogram_transform(mel_spec)

        image_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        audio_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, mel_spec, true_label, image_pseudolabel, audio_pseudolabel, index

    def __len__(self):
        return self.limit

def get_image_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(28, padding=4),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_spectrogram_transform():
    return transforms.Compose([
        # Normalize spectrogram values
        transforms.Lambda(lambda x: (x - x.mean()) / x.std())
    ])

def get_loader(image_dir, spectrogram_folder, batch_size=64, limit=None):
    image_transform = get_image_transform()
    spectrogram_transform = get_spectrogram_transform()
    dataset = IndexedDataset(image_dir, spectrogram_folder, image_transform=image_transform, spectrogram_transform=spectrogram_transform, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class IndexedDataset(Dataset):
    def __init__(self, image_folder, spectrogram_folder, image_transform=None, spectrogram_transform=None, limit=10):
        self.image_folder = image_folder
        self.spectrogram_folder = spectrogram_folder
        self.image_transform = image_transform
        self.spectrogram_transform = spectrogram_transform
        self.image_files = sorted(os.listdir(image_folder))[:limit]
        self.limit = len(self.image_files)
        self.pseudolabels = [None] * self.limit

    def update_pseudolabels(self, new_image_pseudolabels, new_audio_pseudolabels):
        self.image_pseudolabels = new_image_pseudolabels[:self.limit]
        self.audio_pseudolabels = new_audio_pseudolabels[:self.limit]

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_path).convert('L')
        true_label = int(self.image_files[index].split('_')[2])

        # Extract the id part from the image filename
        image_id = self.image_files[index].split('_')[-1].split('.')[0]  # Extract the id part from image filename
        spectrogram_path = os.path.join(self.spectrogram_folder, f'mnist_digit_{true_label}_{image_id}.pt')
        mel_spec = torch.load(spectrogram_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.spectrogram_transform:
            mel_spec = self.spectrogram_transform(mel_spec)

        # Convert spectrogram from 1 channel to 3 channels
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.repeat(3, 1, 1)

        image_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        audio_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, mel_spec, true_label, image_pseudolabel, index  # Returning 5 values now

    def __len__(self):
        return self.limit

def get_image_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(28, padding=4),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_spectrogram_transform():
    return transforms.Compose([
        # Normalize spectrogram values
        transforms.Lambda(lambda x: (x - x.mean()) / x.std())
    ])

def get_loader(image_dir, spectrogram_folder, batch_size=64, limit=None):
    image_transform = get_image_transform()
    spectrogram_transform = get_spectrogram_transform()
    dataset = IndexedDataset(image_dir, spectrogram_folder, image_transform=image_transform, spectrogram_transform=spectrogram_transform, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


