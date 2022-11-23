import os
import random
import numpy as np

import tqdm
import torch

import torchattacks
from pgd import PGD
from gen_inter_samples import Interpolate
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity


## Lim Dataset
class CustomDataset_Adv(torch.utils.data.Dataset):
    """Dataset wrapper to select subsets of training data"""

    def __init__(
        self,
        dataset,
        mode="random",
        per_cls_per=1,
        num_classes=10,
        aug=None,
        clf=None,
        noise_sd=0.25,
        mix_coeff=0.5,
    ):

        self.dataset = dataset
        self.mode = mode
        self.aug = aug
        self.per_cls_per = per_cls_per
        self.per_cls = round((per_cls_per * len(dataset)) / num_classes)
        print(f"Per-Class Samples: {self.per_cls}")
        self.num_classes = num_classes
        self.clf = clf
        self.noise_sd = noise_sd
        self.mix_coeff = mix_coeff
        print(f"Using Mixing Coefficient: {mix_coeff}")

        self.selects = self.get_selects()
        self.dataset, self.selects = self.get_sel_dataset()
        print(f"Length of Filtered/Seleted Dataset: {len(self.dataset)}")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.to_tensor = transforms.ToTensor()

        if aug:
            self.comb_dataset = self.augment()
        else:
            self.aug_dataset = []
            self.comb_dataset = self.dataset

        print(
            f"OG Dataset: {len(self.dataset)} | Combined Dataset: {len(self.comb_dataset)} | Total Selects: {sum(self.selects)}"
        )

    def get_sel_dataset(self):
        sel_dataset = []
        selects = []

        for idx in range(len(self.dataset)):
            x, y = self.dataset[idx]
            sel = self.selects[idx]
            if sel:
                sel_dataset.append((x, y))
                selects.append(True)

        return sel_dataset, selects

    def get_selects(self):

        path = f"./assets/cifar10_{self.mode}_{self.per_cls}.npy"
        if os.path.exists(path):
            print(f"{path} Already Exists | Loading")
            selects = np.load(path)

        else:
            if self.mode == "random":

                cls_dict = {x: [] for x in range(self.num_classes)}
                for idx in range(len(self.dataset)):
                    _, y = self.dataset[idx]
                    cls_dict[y].append(idx)

                selects = np.array([False] * (len(self.dataset)))
                for cls_ in cls_dict.keys():
                    cls_idexes = cls_dict[cls_]
                    sel_cls_idexes = random.sample(cls_idexes, self.per_cls)

                    selects[sel_cls_idexes] = True

                np.save(path, selects)
                print(f"Saved at @ {path}")
            else:
                raise Exception(f"{self.mode}_{self.per_cls} not implemented")

        return selects

    def augment(self):

        path = f"./assets/{self.per_cls_per}_{self.aug}_inter.npy"
        if os.path.isfile(path):
            print(f"{path} already generated!")
            aug_dataset = np.load(path, allow_pickle=True)
            return aug_dataset

        if "pgd" in self.aug:
            print(f"Using PGD")
            attack = PGD(self.clf, eps=8 / 255, alpha=2 / 255, steps=20)
        else:
            print(f"Attack Not Found!")
            raise NotImplementedError

        attack_inter = Interpolate(self.clf, steps=50, lr=0.001, norm_mode="clamp")
        print(attack, attack_inter)

        print(f"-" * 100)
        aug_dataset = []
        for idx in tqdm.tqdm(range(len(self.dataset)), leave=False, ascii=True):

            x, y = self.dataset[idx]

            x = self.to_tensor(x)  ## Convert to a tensor first
            x = x.unsqueeze(0).cuda()
            y_bound = torch.from_numpy(np.array(y)).unsqueeze(0).cuda()

            x_bound = attack(x, y_bound)  ## Boundary Sample

            logit_og, _ = self.clf(x)
            logit_og = logit_og.detach()

            logit_bound, _ = self.clf(x_bound)
            logit_bound = logit_bound.detach()

            tar_logit_inter = (
                self.mix_coeff * logit_og + (1 - self.mix_coeff) * logit_bound
            )
            x_inter = attack_inter(x, tar_logit_inter)  ## Interpolated Sample

            aug_dataset.append(
                (
                    x.cpu().squeeze(0).detach(),
                    x_bound.cpu().squeeze(0).detach(),
                    x_inter.cpu().squeeze(0).detach(),
                    y,
                )
            )

        ## Save this dataset:
        aug_dataset = np.asarray(aug_dataset, dtype=object)
        np.save(path, aug_dataset)

        return aug_dataset

    def __getitem__(self, i):

        x, x_adv, x_inter, y = self.comb_dataset[i]  ## Get sample and label

        x = self.transform(x)
        x_adv = self.transform(x_adv)
        x_inter = self.transform(x_inter)

        return (x, x_adv, x_inter, y)

    def __len__(self):
        return len(self.comb_dataset)


def _cifar10_decrop(
    split: str, per_cls_per: float, aug: str, clf, mix_coeff: float
) -> Dataset:
    dataset_path = "./" + os.path.join(
        os.getenv("PT_DATA_DIR", "datasets"), "dataset_cache"
    )
    print(f"Loading Dataset @ {dataset_path}")
    if split == "train":
        print(f"{split} | Per Class Percentage: {per_cls_per}")

        trainset = datasets.CIFAR10(dataset_path, train=True, download=True)
        lim_trainset = CustomDataset_Adv(
            trainset, per_cls_per=per_cls_per, aug=aug, clf=clf, mix_coeff=mix_coeff
        )
        return lim_trainset

    elif split == "test":
        return datasets.CIFAR10(
            dataset_path, train=False, download=True, transform=transforms.ToTensor()
        )

    else:
        raise Exception("Unknown split name.")
