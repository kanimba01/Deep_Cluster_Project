import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class IndexedDataset(Dataset):
    """ Custom dataset that includes image file paths for features and indexes, 
        with dynamic pseudolabel updating capability and optional dataset limiting.
    """
    def __init__(self, dataset, transform=None, limit=None):
        self.dataset = dataset
        self.transform = transform
        self.limit = limit if limit is not None else len(self.dataset)
        # Initialize pseudolabels with None; will be updated after clustering
        self.pseudolabels = [None] * self.limit  

    def update_pseudolabels(self, new_pseudolabels):
        """ Update the dataset with new pseudolabels after clustering. """
        self.pseudolabels = new_pseudolabels[:self.limit]  # Ensure pseudolabel list matches the limit

    def __getitem__(self, index):
        image, true_label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        # Retrieve the pseudolabel if available, otherwise use true label as fallback
        pseudolabel = self.pseudolabels[index] if self.pseudolabels[index] is not None else true_label
        if image.shape[0] == 1:  # Ensure the image has 3 channels if the model expects 3-channel input
            image = image.repeat(3, 1, 1)
        return image, true_label, pseudolabel, index  # Include both true_label and pseudolabel

    def __len__(self):
        return self.limit


def get_transform():
    """ Image transformations for preprocessing. """
    return transforms.Compose([
        # Resize to the input size expected by the model
        transforms.RandomRotation(degrees=15),  # Rotate the image by up to 15 degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Flip the image horizontally with a probability of 0.5
        transforms.RandomVerticalFlip(p=0.5),    # Flip the image vertically with a probability of 0.5
        transforms.RandomCrop(28, padding=4),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Normalization
    ])


def get_loader(batch_size=128, limit=None):
    """ Prepare and return a data loader. """
    raw_dataset = datasets.MNIST(root='./data', train=True, download=True)
    transformed_dataset = IndexedDataset(raw_dataset, transform=get_transform(), limit=limit)
    return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
