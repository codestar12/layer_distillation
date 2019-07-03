cd ..
 #for layer_epoch in 64 32 16 8 4 2 1; do
 #    for model_epoch in 1 2 4 8 16 32 64; do
layer_epoch=128
model_epoch=1
python ./iterative_replacement.py \
-le=$layer_epoch \
-me=$model_epoch \
-n='prod' \
-ds='ds+cifar10' \
-md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
-rd=./cifar-vgg/cifar10vgg_pretrained.h \
-ld=./logs/vgg16_cifar10_high/agg_layer_"$layer_epoch"_model_"$model_epoch".json

# python ./iterative_replacement.py \
# -le=$layer_epoch \
# -me=$model_epoch \
# -n='prod' \
# -ds='ds+cifar10' \ 
# -md=./models/vgg16_cifar10_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
# -rd=./cifar-vgg/cifar10vgg_pretrained.h \
# -ld=./logs/vgg16_cifar10_high/agg_layer_"$layer_epoch"_model_"$model_epoch".json
#    done
# done

#for layer_epoch in 64 32 16 8 4 2 1; do
#    for model_epoch in 1 2 4 8 16 32 64; do
#    python ./iterative_replacement.py \
#    -le=$layer_epoch \
#    -me=$model_epoch \
#    -n='prod'\
#    -ds=cifar100 \
#    -md=./models/vgg16_cifar100_high/vgg_"$layer_epoch"_"$model_epoch".h5 \
#    -rd=./cifar-vgg/cifar100vgg_pretrained.h5 \
#    -ld=./logs/vgg16_cifar100_high/layer_"$layer_epoch"_model_"$model_epoch".json
#    done
#done

