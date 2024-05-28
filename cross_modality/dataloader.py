import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchaudio


class IndexedDataset(Dataset):
    def __init__(self, image_folder, audio_folder, image_transform=None, spectrogram_transform=None, limit=10):
        self.image_folder = image_folder
        self.audio_folder = audio_folder
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
        audio_path = os.path.join(self.audio_folder, f'mnist_digit_{true_label}_{image_id}.wav')
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert waveform to Mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )(waveform)
        
        # Convert to decibels
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.spectrogram_transform:
            mel_spectrogram = self.spectrogram_transform(mel_spectrogram)

        # Convert spectrogram from 1 channel to 3 channels if necessary
        if mel_spectrogram.shape[0] == 1:
            mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)

        image_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        audio_pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, mel_spectrogram, true_label, image_pseudolabel, index  # Returning 5 values now

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
        transforms.Lambda(lambda x: (x - x.mean()) / x.std())  # Normalize spectrogram values
    ])


def get_loader(image_dir, audio_dir, batch_size=64, limit=None):
    image_transform = get_image_transform()
    spectrogram_transform = get_spectrogram_transform()
    dataset = IndexedDataset(image_dir, audio_dir, image_transform=image_transform, spectrogram_transform=spectrogram_transform, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
