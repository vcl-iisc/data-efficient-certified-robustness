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
        denoiser="./denoiser_models/"$size"_obj_stability_clf_ResNet110_denoiser_"$arch"_"$sigma"/best.pth.tar"
        python code/certify.py --dataset cifar10 --base_classifier $pretrained_classifier --sigma $sigma --outfile $output --skip 20 --denoiser $denoiser --gpu 1
    done
done