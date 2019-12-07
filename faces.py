import io
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class Faces(Dataset):
    def __init__(self, dataset_path):
        self.zip_file = ZipFile(dataset_path)
        self.file_list = self.zip_file.filelist
        assert self.file_list[0].filename == "faces/"
        del self.file_list[0]  # remove "faces/" from the file list
        self.transform = transforms.Compose([transforms.Resize(64)])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        raw_data = self.zip_file.read(filename)
        img = Image.open(io.BytesIO(raw_data))
        #img = self.transform(img)
        return img


def main():  # preprocess data and store the dataset in a single file
    raw_dataset_path = Path("0_raw_data") / "faces.zip"
    processed_dataset_path = Path("1_data") / "faces64.npy"

    dataset = Faces(raw_dataset_path)
    data_shape = np.array(dataset[0]).shape

    print("Size:", len(dataset))

    processed_dataset = np.empty((len(dataset), *data_shape), dtype=np.uint8)
    for i in range(processed_dataset.shape[0]):
        data = np.array(dataset[i])  # (96, 96, 3)
        processed_dataset[i] = data
    np.save(str(processed_dataset_path), processed_dataset)


if __name__ == "__main__":
    main()
