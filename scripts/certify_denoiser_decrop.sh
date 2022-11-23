pretrained_classifier="path/to/pre-trained_model"
output="certification_output/"

List=( 
    0.01
    )

SIGMAS=(
    0.25
    )

arch='cifar_dncnn'

for size in ${List[*]}; do
    for sigma in ${SIGMAS[*]}; do
        denoiser="./denoiser_models/"$size"_neg_domain_disc_pgd_mixcoeff_0.8_obj_1.0*stability_4.0*cosim_pgd_inter_4.0*mmd_pgd_inter_clf_ResNet110_denoiser_"$arch"_"$sigma"/best.pth.tar"
        python code/certify.py --dataset cifar10 --base_classifier $pretrained_classifier --sigma $sigma --outfile $output --skip 20 --denoiser $denoiser --gpu 2
    done
done