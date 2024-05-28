# Deep Cluster Project

This project implements Deep Clustering fThis code implements the unsupervised training of convolutional neural networks, or convnets, as described in the paper [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)
The implementation was carried using pytorch and mac m1 GPU. the project consists of two part experiment:
1. **Single Modality Deep Clustering**: Processes a single modality visual type.
2. **Cross Modality Deep Clustering**: Processes visual and audio types of input data simultaneously.


### Requirements

The project requires the following Python packages:

- torch
- torchvision
- numpy
- scikit-learn
- tqdm
- tensorboard
- matplotlib
- Pillow

You can install all dependencies by running:

```sh
pip install -r requirements.txt