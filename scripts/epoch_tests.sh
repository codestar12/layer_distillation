for layer_epoch in 1 2 4 8 16 32 ; do
    for model_epoch in 1 2 4 8 16 32 ; do
    python ../iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -md=./models/vgg16_cifar10_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./vgg16_cifar10.h5 \
    -ld=./logs/vgg16_cifar10/layer_"$layer_epoch"_model_"$model_epoch".json
    done
done