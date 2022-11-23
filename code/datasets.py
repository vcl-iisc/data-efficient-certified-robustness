from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from zipdata import ZipData

import random
import bisect
import numpy as np
import os
import pickle
import torch

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "imagenet32", "cifar10"]

## Lim Dataset
class CustomDataset(torch.utils.data.Dataset):
    """Dataset wrapper to select subsets of training data"""

    def __init__(self, dataset, mode="random", per_cls_per=1, num_classes=10):

        self.dataset = dataset
        self.mode = mode
        self.per_cls_per = per_cls_per  ## 0.50
        self.per_cls = round(
            (per_cls_per * len(dataset)) / num_classes
        )  ## 50 -> 50/10 : 5
        print(f"Per-Class Samples: {self.per_cls}")
        self.num_classes = num_classes

        self.selects = self.get_selects()
        self.comb_dataset = self.get_comb_dataset()

        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def get_comb_dataset(self):
        comb_dataset = []

        for idx in range(len(self.dataset)):
            x, y = self.dataset[idx]
            sel = self.selects[idx]

            if sel:
                comb_dataset.append((x, y))  ## Append in the dataset

        return comb_dataset

    def get_selects(self):

        path = f"./assets/cifar10_{self.mode}_{self.per_cls}.npy"
        if os.path.exists(path):
            print(f"{path} Already Exists | Loading")
            selects = np.load(path)

        else:
            if not os.path.isdir("./assets/"):
                os.makedirs("./assets/")

            if self.mode == "random":
                ## Select 625 samples from each class at random

                cls_dict = {x: [] for x in range(self.num_classes)}
                for idx in range(len(self.dataset)):
                    _, y = self.dataset[idx]  ## Sample at curent index
                    cls_dict[y].append(idx)  ## Samples belonging to class 'y'

                ## Select 'per_cls' amount of samples from each class in cls_dict
                selects = np.array([False] * (len(self.dataset)))
                for cls_ in cls_dict.keys():
                    cls_idexes = cls_dict[cls_]
                    sel_cls_idexes = random.sample(cls_idexes, self.per_cls)

                    selects[
                        sel_cls_idexes
                    ] = True  ## These samples are selected ~ i.e. set as True

                np.save(path, selects)  ## Save numpy array
                print(f"Saved at @ {path}")
            else:
                raise Exception(f"{self.mode}_{self.per_cls} not implemented")

        return selects

    def check_selects(self):
        cls_counter = {x: 0 for x in range(self.num_classes)}

        for idx in range(len(self.dataset)):
            _, y = self.dataset[idx]
            if self.selects[idx]:
                cls_counter[y] += 1

        for cls_ in cls_counter.keys():
            print(f"Class: {cls_} | Samples Selected: {cls_counter[cls_]}")

    def __getitem__(self, i):
        x, y = self.comb_dataset[i]  ## Get sample and label

        x = self.transform(x)  ## Augmenting Data
        return (x, y)

    def __len__(self):
        return len(self.comb_dataset)


def get_dataset(dataset: str, split: str, *args) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)

    elif dataset == "imagenet32":
        return _imagenet32(split)

    elif dataset == "cifar10":
        if split == "train":
            per_cls_per = args[0]
        else:
            per_cls_per = 1.00

        return _cifar10(split, per_cls_per=per_cls_per)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset."""
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet32":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str, per_cls_per: float) -> Dataset:
    dataset_path = "./" + os.path.join(
        os.getenv("PT_DATA_DIR", "datasets"), "dataset_cache"
    )
    print(f"Loading Dataset @ {dataset_path}")
    if split == "train":
        print(f"{split} | Per Class Percentage: {per_cls_per}")
        print(f"using horizontal flip augmentation")

        trainset = datasets.CIFAR10(dataset_path, train=True, download=True)

        lim_trainset = CustomDataset(trainset, per_cls_per=per_cls_per)
        return lim_trainset

    elif split == "test":
        return datasets.CIFAR10(
            dataset_path, train=False, download=True, transform=transforms.ToTensor()
        )

    else:
        raise Exception("Unknown split name.")


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose(
            [
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )
    return datasets.ImageFolder(subdir, transform)


def _imagenet32(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv("PT_DATA_DIR", "datasets"), "Imagenet32")

    if split == "train":
        return ImageNetDS(
            dataset_path,
            32,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif split == "test":
        return ImageNetDS(
            dataset_path, 32, train=False, transform=transforms.ToTensor()
        )


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    base_folder = "Imagenet{}_train"
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [
        ["val_data", ""],
    ]

    def __init__(
        self, root, img_size, train=True, transform=None, target_transform=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, "rb") as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry["data"])
                    self.train_labels += [label - 1 for label in entry["labels"]]
                    self.mean = entry["mean"]

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape(
                (self.train_data.shape[0], 3, 32, 32)
            )
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, "rb")
            entry = pickle.load(fo)
            self.test_data = entry["data"]
            self.test_labels = [label - 1 for label in entry["labels"]]
            fo.close()
            self.test_data = self.test_data.reshape(
                (self.test_data.shape[0], 3, 32, 32)
            )
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


if __name__ == "__main__":
    dataset = get_dataset("imagenet32", "train")