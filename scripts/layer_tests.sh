cd ..
layer_epoch=64
 #for layer_epoch in 64 32 16 8 4 2 1; do
for model_epoch in 0 1 4 16; do

#model_epoch=0
    python ./iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -n='prod' \
    -ds='cifar10' \
    -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./cifar-vgg/cifar10vgg_pretrained.h \
    -ld=./logs/vgg16_cifar10_high/one_layer_agg_layer_"$layer_epoch"_model_"$model_epoch".json \
    -bl=1 \
    -bs=64

    mv model.h5 model_1_layer.h5


    python ./iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -n='prod' \
    -ds='cifar10' \
    -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./cifar-vgg/cifar10vgg_pretrained.h \
    -ld=./logs/vgg16_cifar10_high/two_layers_bn_agg_layer_"$layer_epoch"_model_"$model_epoch".json \
    -bl=2 \
    -bs=64

    mv model.h5 model_2_layer.h5


    python ./iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -n='prod' \
    -ds='cifar10' \
    -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./cifar-vgg/cifar10vgg_pretrained.h \
    -ld=./logs/vgg16_cifar10_high/two_layers_no_bn_agg_layer_"$layer_epoch"_model_"$model_epoch".json \
    -bl=2 \
    -bn=False \
    -bs=64

    mv model.h5 model_2_layer_bn.h5


    python ./iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -n='prod' \
    -ds='cifar10' \
    -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./cifar-vgg/cifar10vgg_pretrained.h \
    -ld=./logs/vgg16_cifar10_high/three_layers_bn_agg_layer_"$layer_epoch"_model_"$model_epoch".json \
    -bl=3 \
    -bn=True \
    -bs=64

    mv model.h5 model_3_layer_bn.h5

    python ./iterative_replacement.py \
    -le=$layer_epoch \
    -me=$model_epoch \
    -n='prod' \
    -ds='cifar10' \
    -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
    -rd=./cifar-vgg/cifar10vgg_pretrained.h \
    -ld=./logs/vgg16_cifar10_high/three_layers_no_bn_agg_layer_"$layer_epoch"_model_"$model_epoch".json \
    -bl=3 \
    -bn=False \
    -bs=64

    mv model.h5 model_3_layer.h5

done