## Generate Graph for Multiple Per-Class-Percent
import os
from analyze import (
    plot_certified_accuracy_per_sigma_best_model,
    Line,
    ApproximateAccuracy,
)

## 1% Data
paths = [
    (
        "0.01_obj_stability_clf_ResNet110_denoiser_cifar_dncnn_0.25",
        "1% (Baseline)",
    ),
    (
        "0.01_neg_domain_disc_pgd_mixcoeff_0.8_obj_1.0*stability_4.0*cosim_pgd_inter_4.0*mmd_pgd_inter_clf_ResNet110_denoiser_cifar_dncnn_0.25",
        "1% (DE-CROP)",
    ),
]

methods = []
labels = []

for path, label in paths:
    methods.append(
        Line(ApproximateAccuracy(f"./certification_output/{path}"), "$\sigma = 0.25$")
    )
    labels.append(label)

certification_result_without_denoiser = "/media2/inder/i2i_smoothing/denoised-smoothing/data/certify/cifar10/no_denoiser/MODEL_resnet110_90epochs/noise_0.00/test_N10000/sigma_0.25"  # PROVIDED BY Salman et al. 2020

outdir = "cert_graphs"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

outfile = f"./{outdir}/decrop_0.01_per_cls_0.25"
title = "DE-CROP vs Baseline"

plot_certified_accuracy_per_sigma_best_model(
    outfile,
    title,
    1.0,
    methods=methods,
    labels=labels,
    methods_base=[
        Line(
            ApproximateAccuracy(certification_result_without_denoiser),
            "$\sigma = 0.25$",
        )
    ],
    label_base="Without Denoiser",
    sigmas=[0.25],
)
