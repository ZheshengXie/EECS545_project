import logging
import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, dataset_config):  # data will be normalized to [0, 1] or [-1, 1]
        self.config = SimpleNamespace(**dataset_config)
        self.logger = logging.getLogger("training")
        dataset_path = Path("1_data") / self.config.path
        self.logger.info("Dataset path: " + str(dataset_path))
        self._dataset = np.load(str(dataset_path))
        # transforms
        transforms_list = [
            # Converts numpy.ndarray (H x W x C) in range [0, 255] to a torch.FloatTensor(C x H x W) in range [0.0, 1.0]
            transforms.ToTensor()
        ]
        if self.config.normalize == "[-1, 1]":  # [0, 1] -> [-1, 1]
            self.logger.info("Normalize to [-1, 1]")
            num_channels = 1 if len(self._dataset.shape) == 3 else 3
            transforms_list.append(transforms.Normalize([0.5] * num_channels, [0.5] * num_channels))
        else:
            self.logger.info("Normalize to [0, 1]")
        self.transforms = transforms.Compose(transforms_list)
        self.logger.info(
            "#Images: {}    ".format(self._dataset.shape[0]) +
            "Image Size: {}".format(tuple(self.transforms(self._dataset[0]).shape))
        )

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        data = self._dataset[idx]
        return self.transforms(data)


def transform_back(dataset_config, output):
    # revert normalization
    if dataset_config["normalize"] == "[-1, 1]":  # [-1, 1] -> [0, 1]
        output = output * 0.5 + 0.5
    output = (output * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    output = output.transpose(0, 2, 3, 1)  # [batch, C, H, W] -> [batch, H, W, C]
    if (output.shape[-1] == 1):
        output = output.squeeze(axis=-1)
    return output


def save_model(G, D, model_path):  # save G & D to file
    model_path.parent.mkdir(parents=True, exist_ok=True)  # create folder if not exists
    torch.save((G.cpu().state_dict(), D.cpu().state_dict()), model_path)


def load_model(G, D, model_path):  # load G & D from file
    state_G, state_D = torch.load(model_path)
    G.load_state_dict(state_G)
    D.load_state_dict(state_D)


def save_model_from_checkpoint(G, D, checkpoint_path, model_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    save_model(G, D, model_path)


def pack_sample_files(exp_dir):
    import zipfile
    zip_path = exp_dir / "samples.zip"
    print("Pack all sample files into {}".format(zip_path))
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for sample_path in exp_dir.glob("sample_*.npy"):
            zip_file.write(sample_path, arcname=Path(sample_path.parent.name) / sample_path.name)


def init_logger(logger_name, log_path):  # log to a certain file with level INFO
    show_in_console = True
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # save to file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # show in console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(formatter)
    if show_in_console:
        logger.addHandler(stream_handler)
    return logger


def import_class(module_class):  # import class by class name
    module_name, class_name = module_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
