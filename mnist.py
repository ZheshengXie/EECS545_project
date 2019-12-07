from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets


def main():  # preprocess data and store the dataset in a single file
    raw_dataset_path = Path("0_raw_data")
    processed_dataset_path = Path("1_data") / "mnist.npy"

    transform = transforms.Compose([transforms.Resize(28)])
    dataset = datasets.MNIST(raw_dataset_path, train=True, download=True, transform=transform)
    data_shape = np.array(dataset[0][0]).shape

    print("Size:", len(dataset))

    processed_dataset = np.empty((len(dataset), *data_shape), dtype=np.uint8)
    for i in range(processed_dataset.shape[0]):
        data = np.array(dataset[i][0])  # (28, 28)
        processed_dataset[i] = data
    np.save(str(processed_dataset_path), processed_dataset)


if __name__ == "__main__":
    main()
