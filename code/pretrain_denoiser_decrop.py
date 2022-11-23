# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# File for training denoisers with at most one classifier attached to

from architectures import (
    DENOISERS_ARCHITECTURES,
    get_architecture,
    IMAGENET_CLASSIFIERS,
)
from datasets import get_dataset, DATASETS
from test_denoiser import test, test_with_classifier
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import (
    AverageMeter,
    accuracy,
    init_logfile,
    log,
    copy_code,
    requires_grad_,
    MMD_Loss,
    seed_everything,
)

import argparse
import numpy as np
import os
import time
import torch

import wandb
from datasets_decrop import _cifar10_decrop as get_dataset_decrop

import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--dataset", type=str, choices=DATASETS)
parser.add_argument("--arch", type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument(
    "--outdir", type=str, help="folder to save denoiser and training log)"
)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--batch", default=256, type=int, metavar="N", help="batchsize (default: 256)"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-3,
    type=float,
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--lr_step_size",
    type=int,
    default=30,
    help="How often to decrease learning by gamma.",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--gpu", default=None, type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
parser.add_argument(
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--noise_sd",
    default=0.0,
    type=float,
    help="standard deviation of noise distribution for data augmentation",
)
parser.add_argument(
    "--objective",
    default="denoising",
    type=str,
    help="the objective that is used to train the denoiser",
    choices=["denoising", "classification", "stability", "cosine_sim"],
)
parser.add_argument(
    "--classifier",
    default="",
    type=str,
    help="path to the classifier used with the `classificaiton`"
    "or `stability` objectives of the denoiser.",
)
parser.add_argument(
    "--pretrained-denoiser", default="", type=str, help="path to a pretrained denoiser"
)
parser.add_argument(
    "--optimizer",
    default="Adam",
    type=str,
    help="SGD, Adam, or Adam then SGD",
    choices=["SGD", "Adam", "AdamThenSGD"],
)
parser.add_argument(
    "--start-sgd-epoch",
    default=50,
    type=int,
    help="[Relevent only to AdamThenSGD.] Epoch at which adam switches to SGD",
)
parser.add_argument(
    "--start-sgd-lr",
    default=1e-3,
    type=float,
    help="[Relevent only to AdamThenSGD.] LR at which SGD starts after Adam",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="if true, tries to resume training from an existing checkpoint",
)
parser.add_argument(
    "--azure_datastore_path", type=str, default="", help="Path to imagenet on azure"
)
parser.add_argument(
    "--philly_imagenet_path", type=str, default="", help="Path to imagenet on philly"
)
parser.add_argument(
    "--per_cls_per",
    type=float,
    default=1.00,
    help="per-class sample percent for training data",
)
parser.add_argument("--wandb", help="Log Using WANBD", action="store_true")

parser.add_argument("--aug", help="augmentation", default=None, type=str)
parser.add_argument(
    "--mix_coeff", help="interpolation mixing coefficient", default=0.8, type=float
)

parser.add_argument("--w_ce", type=float, default=1.00)
parser.add_argument("--w_cs", type=float, default=4.00)
parser.add_argument("--w_mmd", type=float, default=4.00)
parser.add_argument("--seed", type=int, default=3)

args = parser.parse_args()

arch = args.classifier.split("/")[-3].split("_")[0]
args.outdir = f"./denoiser_models/{args.per_cls_per:.2f}_{args.aug}_mixcoeff_{args.mix_coeff}_obj_{args.w_ce}*{args.objective}_{args.w_cs}*cosim_pgd_inter_{args.w_mmd}*mmd_pgd_inter_clf_{arch}_{args.outdir}_{args.arch}_{args.noise_sd}"

if args.wandb:
    run = wandb.init(
        project="lim_cert_camera_ready",
        name=args.outdir.split("/")[-1],
        config=args,
        reinit=True,
    )

if args.azure_datastore_path:
    os.environ["IMAGENET_DIR_AZURE"] = os.path.join(
        args.azure_datastore_path, "datasets/imagenet_zipped"
    )
if args.philly_imagenet_path:
    os.environ["IMAGENET_DIR_PHILLY"] = os.path.join(args.philly_imagenet_path, "./")

seed_everything(args.seed)


toPilImage = ToPILImage()


def cosim_loss(x, y):
    return 1 - (F.cosine_similarity(x, y)).mean()


mmd_loss = MMD_Loss().cuda()


def calc_accuracy(loader, model, noise=False, noise_sd=0.25):

    model.eval()

    total, correct = 0, 0

    for data, labels in loader:

        data, labels = data.cuda(), labels.cuda()
        if noise:
            data += torch.randn_like(data, device="cuda") * noise_sd
        outputs, _ = model(data)
        _, pred = torch.max(outputs, 1)
        corr = (pred == labels).float().sum(0).item()
        total += data.size(0)
        correct += corr

    return correct, total


def main():
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    copy_code(args.outdir)

    if args.pretrained_denoiser:
        print(f"Loading pretrained denoiser @ {args.pretrained_denoiser}")
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint["arch"] == args.arch
        denoiser = get_architecture(checkpoint["arch"], args.dataset)
        denoiser.load_state_dict(checkpoint["state_dict"])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    if args.optimizer == "Adam":
        optimizer = Adam(
            denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = SGD(
            denoiser.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamThenSGD":
        optimizer = Adam(
            denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    starting_epoch = 0
    logfilename = os.path.join(args.outdir, "log.txt")

    ## Resume from checkpoint if exists and if resume flag is True
    denoiser_path = os.path.join(args.outdir, "checkpoint.pth.tar")
    if args.resume and os.path.isfile(denoiser_path):
        print("=> loading checkpoint '{}'".format(denoiser_path))
        checkpoint = torch.load(
            denoiser_path, map_location=lambda storage, loc: storage
        )
        assert checkpoint["arch"] == args.arch
        starting_epoch = checkpoint["epoch"]
        denoiser.load_state_dict(checkpoint["state_dict"])
        if (
            starting_epoch >= args.start_sgd_epoch and args.optimizer == "AdamThenSGD "
        ):  # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(
                denoiser.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                denoiser_path, checkpoint["epoch"]
            )
        )
    else:
        if args.resume:
            print("=> no checkpoint found at '{}'".format(args.outdir))

    assert (
        args.classifier != ""
    ), "Please specify a path to the classifier you want to attach the denoiser to."

    if args.classifier in IMAGENET_CLASSIFIERS:
        assert args.dataset == "imagenet"
        # loading pretrained imagenet architectures
        clf = get_architecture(args.classifier, args.dataset, pytorch_pretrained=True)
    else:
        checkpoint = torch.load(args.classifier)
        clf = get_architecture(checkpoint["arch"], "cifar10", with_latent=True)
        clf.load_state_dict(checkpoint["state_dict"])
    clf.cuda().eval()
    requires_grad_(clf, False)

    train_dataset = get_dataset_decrop(
        "train", args.per_cls_per, aug=args.aug, clf=clf, mix_coeff=args.mix_coeff
    )
    test_dataset = get_dataset(args.dataset, "test")
    pin_memory = args.dataset == "imagenet"
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    ## This is used to test the performance of the denoiser attached to a cifar10 classifier
    cifar10_test_loader = DataLoader(
        get_dataset("cifar10", "test"),
        shuffle=False,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    init_logfile(
        logfilename,
        f"Samples Selected in Training Data: {len(train_dataset)}/{len(train_dataset.selects)} | Per-Class Samples: {args.per_cls_per*100}%",
    )
    log(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")

    corr, total = calc_accuracy(
        test_loader, nn.Sequential(denoiser, clf), noise=True, noise_sd=args.noise_sd
    )
    print(f"Pretrained Denoiser Accuracy: {(corr/total)*100.}")

    if args.objective == "cosine_sim":
        print("Using Cosine Similarity Loss")
        criterion = cosim_loss
    else:
        print("Using Cross Entropy Loss")
        criterion = CrossEntropyLoss(
            size_average=None, reduce=None, reduction="mean"
        ).cuda()

    criterion_test = CrossEntropyLoss(
        size_average=None, reduce=None, reduction="mean"
    ).cuda()
    best_acc = 0

    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        train_loss = train(
            train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd, clf
        )
        if args.dataset == "imagenet":
            test_loss, test_acc = test_with_classifier(
                test_loader,
                denoiser,
                criterion_test,
                args.noise_sd,
                args.print_freq,
                clf,
            )
        else:
            # This is needed so that cifar10 denoisers trained using imagenet32 are still evaluated on the cifar10 testset
            test_loss, test_acc = test_with_classifier(
                cifar10_test_loader,
                denoiser,
                criterion_test,
                args.noise_sd,
                args.print_freq,
                clf,
            )

        after = time.time()

        log(
            logfilename,
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, args.lr, train_loss, test_loss, test_acc
            ),
        )

        if args.wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "epoch": epoch,
                }
            )

        scheduler.step()
        args.lr = scheduler.get_last_lr()[0]

        # Switch from Adam to SGD
        if (
            epoch == args.start_sgd_epoch and args.optimizer == "AdamThenSGD "
        ):  # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(
                denoiser.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        torch.save(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.outdir, "checkpoint.pth.tar"),
        )

        if args.objective == "denoising" and test_loss < best_loss:
            best_loss = test_loss
        elif (
            args.objective in ["classification", "stability", "cosine_sim"]
            and test_acc > best_acc
        ):
            best_acc = test_acc
        else:
            continue

        torch.save(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.outdir, "best.pth.tar"),
        )


def train(
    loader: DataLoader,
    denoiser: torch.nn.Module,
    criterion,
    optimizer: Optimizer,
    epoch: int,
    noise_sd: float,
    classifier: torch.nn.Module = None,
):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                (required classifciation/stability objectives), None for denoising objective
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    denoiser.train()
    if classifier:
        classifier.eval()

    for i, (inputs, inputs_adv, inputs_inter, targets) in enumerate(loader):

        if inputs.size(0) == 0:
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs_adv = inputs_adv.cuda()
        inputs_inter = inputs_inter.cuda()

        # augment inputs with noise
        noise = torch.randn_like(inputs, device="cuda") * noise_sd

        optimizer.zero_grad()

        # compute output
        denoised_image = denoiser(inputs + noise)
        if classifier:
            outputs, _ = classifier(denoised_image)

        if isinstance(criterion, MSELoss):
            loss = criterion(denoised_image, inputs)
        elif isinstance(criterion, CrossEntropyLoss):
            if args.objective == "stability":
                with torch.no_grad():
                    targets, _ = classifier(inputs)
                    targets = targets.argmax(1).detach().clone()
            loss = args.w_ce * criterion(outputs, targets)

        ## Add Adversarial Cosim and MMD Loss
        with torch.no_grad():
            targets_logits, features_og = classifier(inputs)
            targets_logits = targets_logits.detach().clone()
            features_og = features_og.detach().clone()

        denoised_adv_logits, features_noisy_adv = classifier(
            denoiser(inputs_adv + noise)
        )
        denoised_inter_logits, features_noisy_inter = classifier(
            denoiser(inputs_inter + noise)
        )

        loss_cosim_adv = args.w_cs * cosim_loss(
            denoised_adv_logits, targets_logits
        )
        loss_cosim_inter = args.w_cs * cosim_loss(
            denoised_inter_logits, targets_logits
        )

        loss_mmd_adv = args.w_mmd * mmd_loss(
            features_og, features_noisy_adv
        )
        loss_mmd_inter = args.w_mmd * mmd_loss(
            features_og, features_noisy_inter
        )

        loss = loss + loss_cosim_adv + loss_cosim_inter + loss_mmd_adv + loss_mmd_inter

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


if __name__ == "__main__":
    main()
    print("=" * 100)
