pretrained_classifier="path/to/pre-trained_model"
epochs=600
lr=1e-3
lr_step_size=200
gamma=0.1
optimizer='AdamThenSGD'



List=(
    0.01
    )

SIGMAS=(
    0.25
    )

for size in ${List[*]}; do
    for sigma in ${SIGMAS[*]}; do
        python ./code/pretrain_denoiser_decrop.py --dataset cifar10 --arch cifar_dncnn --outdir denoiser --noise $sigma --classifier $pretrained_classifier --objective 'stability' --per_cls_per $size --optimizer $optimizer --gamma $gamma --lr_step_size $lr_step_size --lr $lr --epochs $epochs --gpu 2 --aug 'pgd' --batch 200
    done
done