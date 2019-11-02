import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
BATCH_SIZE = 8
VALIDATION_SIZE = 50000
import pathlib
import time
import json
# Add before any TF calls
# Initialize the keras global outside of any tf.functions
temp = tf.zeros([4, 32, 32, 3])  # Or tf.zeros
tf.keras.applications.vgg16.preprocess_input(temp)
mirrored_strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE * mirrored_strategy.num_replicas_in_sync
tf.__version__

tf.executing_eagerly()


AUTOTUNE = tf.data.experimental.AUTOTUNE

from tensorflow.keras.models import load_model



# 

def build_replacement(get_output):
    inputs = tf.keras.Input(shape=get_output.output[0].shape[1::])
    X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}', filters=get_output.output[1].shape[-1], 
                                        kernel_size= (3,3),
                                        padding='Same')(inputs)
    X = tf.keras.layers.BatchNormalization(name=f'batch_norm_{build_replacement.counter}')(X)
    X = tf.keras.layers.ReLU(name=f'relu_{build_replacement.counter}')(X)
    
    build_replacement.counter += 1
    
    X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}', filters=get_output.output[1].shape[-1],
                                        kernel_size=(3,3), 
                                        padding='Same')(X)
    X = tf.keras.layers.BatchNormalization(name=f'batch_norm_{build_replacement.counter}')(X)
    X = tf.keras.layers.ReLU(name=f'relu_{build_replacement.counter}')(X)
    replacement_layers = tf.keras.Model(inputs=inputs, outputs=X)
    
    build_replacement.counter += 1
    
    return replacement_layers

build_replacement.counter = 0

import math
class LayerBatch(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(1281167 // BATCH_SIZE // 5)
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y
    
import math
class LayerTest(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(VALIDATION_SIZE // BATCH_SIZE )
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
#from numba import cuda


from tensorflow.keras.applications.vgg16 import preprocess_input
data_dir = pathlib.Path('/home/cody/datasets/imagenet/train/')


import json
with open('./imagenet_class_index.json') as f:
    CLASS_INDEX = json.load(f)


CLASS_NAMES = np.array([item[0] for key, item in CLASS_INDEX.items()])

def get_label(file_path):
    parts = tf.strings.split(file_path, '/')

    return tf.dtypes.cast(tf.equal(parts[-2], CLASS_NAMES), tf.int32)

def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = preprocess_input(image)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def process_path(file_path):
    label = get_label(file_path)
    image = load_and_preprocess_image(file_path)
    return image, label

def create_ds(data_path, cache='./image-net.tfcache', train=False):
    data_root = pathlib.Path(data_path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    image_count = len(all_image_paths)
    #random.shuffle(all_image_paths)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = path_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #image_label_ds = image_label_ds.cache(cache)
    #if train:
#         #dataset = dataset.cache(cache)
    #    dataset = dataset.shuffle(buffer_size=100)       
    
    dataset = dataset.repeat()
        
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


test_generator = create_ds('/home/cody/datasets/imagenet/val/')
train_generator = create_ds('/home/cody/datasets/imagenet/train/', train=True)

import gc

test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_image)

with mirrored_strategy.scope():
    model = tf.keras.applications.VGG16(weights='imagenet')

    model.save('./model.h5')

    targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']

    targets
    opt = keras.optimizers.RMSprop(lr=0.00005, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

tensorboard_acc = keras.callbacks.TensorBoard(log_dir=f'./logs/train/model_acc/', update_freq='batch')
scores = model.evaluate(test_generator, verbose=2, steps=VALIDATION_SIZE, callbacks=[tensorboard_acc] )
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

all_scores = [{'init':scores}]


start_time = time.time()
while len(targets) > 1:
    
    print(f'targets {targets}')
    print("taking target")
    target = targets[1]
    
    print(f'making output for target layer {target}')
    with mirrored_strategy.scope():
        get_output = tf.keras.Model(inputs=model.input, 
                                    outputs=[model.layers[target - 1].output,
                                            model.layers[target].output])
        optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6)

        loss_object = tf.losses.MeanSquaredError()
        
        get_output.compile(loss=loss_object, optimizer=optimizer)

    get_output.save('./output.h5')
    
    tf.keras.backend.clear_session()

    get_output = tf.keras.models.load_model('./output.h5')
    
    print(f'making replacement layers for target layer {target}')
    
    with mirrored_strategy.scope():
        replacement_layers = build_replacement(get_output)
        
        replacement_len = len(replacement_layers.layers)
        
        optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6)

        loss_object = tf.losses.MeanSquaredError()
        
        replacement_layers.compile(loss=loss_object, optimizer=optimizer)
    
    save = tf.keras.callbacks.ModelCheckpoint('./replacement_layer.h5', 
                                            verbose=0, 
                                            save_weights_only=True,
                                            save_best_only=True)

    layer_train_gen = LayerBatch(get_output, train_generator)
    layer_test_gen = LayerTest(get_output, test_generator)
    
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'./logs/train/layer_{target}')

    print(f'starting fit generator for target layer {target}')
    replacement_layers.fit(x=layer_train_gen, 
                                    epochs=7, 
                                    validation_data=layer_test_gen,
                                    shuffle=False,
                                    validation_steps=VALIDATION_SIZE//5,
                                    verbose=2, callbacks=[save, tensorboard])
    
    print('saving replacement layers to json')
    
    replacement_json = replacement_layers.to_json()
    with open('replacement_layer.json', 'w') as json_file:
        json_file.write(replacement_json)
        
    del replacement_layers
    
    tf.keras.backend.clear_session()

    with open('replacement_layer.json', 'r') as json_file:
        replacement_layers = tf.keras.models.model_from_json(json_file.read())

    print('loading replacement layers weights')
    replacement_layers.load_weights('replacement_layer.h5')
    replacement_layers.compile(loss=loss_object, optimizer=optimizer)
    replacement_layers.evaluate(layer_test_gen)

    model = tf.keras.models.load_model('./model.h5')
    # build top half of model
    print('building top half of model')
    get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[target - 1].output])
    # add in replacement layers
    print('building middle of model with replacement layers')
    new_joint = tf.keras.Model(inputs=get_output.input, outputs=replacement_layers(get_output.output))
    
    #new_joint.summary()
    
    # build bottom of model
    bottom_half = tf.keras.Sequential()
    for layer in model.layers[target + 1::]:
        bottom_half.add(layer)
        
    
    
    print('building bottom of model')
    bottom_half.build(input_shape=new_joint.output.shape)
    #bottom_half.summary()
    print('combining model')
    combined = tf.keras.Model(inputs=new_joint.input, outputs=bottom_half(new_joint.output))
    
    combined.layers[-1].trainable=False
    opt = keras.optimizers.RMSprop(lr=0.00005, decay=1e-6)
    combined.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    
    #combined.summary()
    del bottom_half, new_joint, replacement_layers, model
    gc.collect()
    
#     print('testing combined model')
#     scores = combined.evaluate(x_test, y_test, verbose=1)
#     print('Test loss:', scores[0])
#     print('Test accuracy:', scores[1])


# maybe clear backend here?
    with mirrored_strategy.scope():
        new_combined = tf.keras.Sequential()
        new_layers = []
        new_combined.add(tf.keras.layers.Input(shape=(224,224,3)))
        accum = 0
        print('refactoring model')
        for layer in combined.layers:
            #print(layer.__class__.__name__)
            if hasattr(layer, 'layers'):
                
                for sublayer in layer.layers:
                    #print(sublayer.__class__)
                    if(sublayer.__class__.__name__ != 'InputLayer'): 
                        new_layers.append((sublayer.__class__.__name__, sublayer.get_config(), accum))

                    accum += 1
            elif layer.__class__.__name__ != 'InputLayer':          
                new_layers.append((layer.__class__.__name__, layer.get_config(), accum))
                    
                accum += 1 




        for i, layer in enumerate(new_layers):
            new_combined.add(keras.layers.deserialize(
                                    {'class_name': layer[0], 
                                    'config': layer[1]}))

        new_combined.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    accum = 0
    for i, layer in enumerate(combined.layers):
        if hasattr(layer, 'layers'):

            for sublayer in layer.layers:
                #print(f'{accum} sub is {sublayer} new is {new_combined.layers[accum]}')
                if(sublayer.__class__.__name__ != 'InputLayer'): 
                    #print(sublayer.__class__)
                    new_combined.layers[accum].set_weights(sublayer.get_weights())
                    accum += 1
    #             else:
    #                 accum += 1
            continue
        else:
            #print(layer)
            if(layer.__class__.__name__ != 'InputLayer'):
                print()
                new_combined.layers[accum].set_weights(layer.get_weights())
                accum +=1 
            elif(layer.__class__.__name__ == 'Flatten'):
                accum += 1

#     print('freezing first half of layers')
#     for i in range(target):
#         new_combined.layers[i].trainable = False
    
#     print('freezing last half of layers')
#     for i in range(target +  replacement_len, len(new_combined.layers)):
#         new_combined.layers[i].trainable = False
        
    new_combined.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    del combined
    gc.collect()

    new_save=tf.keras.callbacks.ModelCheckpoint('./model.h5', 
                                                verbose=2, 
                                                save_weights_only=False, 
                                                save_best_only=True)
    print('fine tuning combined model')
    #new_combined.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=5, callbacks=[new_save])
#     new_combined.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=batch_size),
#                         epochs=5,
#                         validation_data=(x_test, y_test),
#                         #workers=5,
#                         callbacks=[new_save])

#     new_combined.fit_generator(generator=train_generator, 
#                                      epochs=5, 
#                                      validation_data=test_generator ,
#                                      verbose=1, callbacks=[save])
    
    print('loading best weights from fine tune')
    #new_combined.load_weights('./refactor_finetune.h5')
    #new_combined.save('.refactor_finetune.h5')
    
    scores = new_combined.evaluate(test_generator, steps=VALIDATION_SIZE, verbose=1, callbacks=[tensorboard_acc])
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    with mirrored_strategy.scope():
        model = tf.keras.Model(inputs=new_combined.input, 
                                    outputs=new_combined.output)
        
        model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    all_scores.append({f'layer {target}': scores})
    #model.summary()
    del new_combined
    gc.collect()
    model.save('./model.h5')
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('./model.h5')
    print('new summary')
    #model.summary()
    targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']
    

    print(f"end time {time.time() - start_time}")
    print(all_scores)
    
print(all_scores)
    

    
    

