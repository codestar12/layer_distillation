import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 4

print(tf.config.experimental_list_devices())


tf.__version__

tf.executing_eagerly()


AUTOTUNE = tf.data.experimental.AUTOTUNE

from tensorflow.keras.models import load_model

model = tf.keras.applications.VGG16(weights='imagenet')

model.save('./baseline_vgg_imagenet.h5')

targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']

targets

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# dataset = dataset.batch(batch_size)
# dataset = dataset.repeat()

# dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# dataset_test = dataset.batch(batch_size)
# dataset_test = dataset.repeat()

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
        return math.ceil(1281167 / BATCH_SIZE / 10)
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y
    
import math
class LayerTest(tf.keras.utils.Sequence):
    
    def __init__(self, input_model, dataset):
        self.input_model = input_model
        self.dataset = dataset.__iter__()
        
    def __len__(self):
        return math.ceil(50000 / BATCH_SIZE)
    
    def __getitem__(self, index):
        X, y = self.input_model(next(self.dataset))
        return X, y

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from numba import cuda


from tensorflow.keras.applications.vgg16 import preprocess_input

# train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
# test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

# train_generator = train_gen.flow_from_directory(
#                 '/home/cody/datasets/imagenet/train/',
#                 target_size=(224, 224),
#                 batch_size=4,
#                 class_mode='categorical'
#                 )

# test_generator = test_gen.flow_from_directory(
#                 '/home/cody/datasets/imagenet/val/',
#                 target_size=(224, 224),
#                 batch_size=4,
#                 class_mode='categorical'
#                 )
import pathlib
import random

def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    
    return preprocess_input(image)

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def create_ds(data_path, cache='./image-net.tfcache'):
    data_root = pathlib.Path(data_path)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    image_count = len(all_image_paths)
    random.shuffle(all_image_paths)
    all_image_labels = [0 for _ in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.cache(cache)
    dataset = image_label_ds.shuffle(buffer_size=8000)  
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

test_generator = create_ds('/home/cody/datasets/imagenet/val/', cache='./image-net-test.tfcache')
train_generator = create_ds('/home/cody/datasets/imagenet/train/', cache='./image-net-train.tfcache')
import gc

opt = keras.optimizers.RMSprop(lr=0.00005, decay=1e-6)
model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])

scores = model.evaluate_generator(test_generator, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
   
while len(targets) > 1:
    
    print(f'targets {targets}')
    print("taking target")
    target = targets[1]
    
    print(f'making output for target layer {target}')
    
    get_output = tf.keras.Model(inputs=model.input, 
                                outputs=[model.layers[target - 1].output,
                                         model.layers[target].output])
    
    print(f'making replacement layers for target layer {target}')
    replacement_layers = build_replacement(get_output)
    
    replacement_len = len(replacement_layers.layers)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)

    loss_object = tf.losses.MeanSquaredError()
    
    replacement_layers.compile(loss=loss_object, optimizer=optimizer)
    
    save = tf.keras.callbacks.ModelCheckpoint('./replacement_layer.h5', 
                                              verbose=0, 
                                              save_weights_only=True,
                                              save_best_only=True)

    layer_train_gen = LayerBatch(get_output, train_generator)
    layer_test_gen = LayerTest(get_output, test_generator)
    
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    print(f'starting fit generator for target layer {target}')
    replacement_layers.fit_generator(generator=layer_train_gen, 
                                     epochs=10, 
                                     validation_data=layer_test_gen ,
                                     verbose=1, callbacks=[save, tensorboard])
    
    print('saving replacement layers to json')
    
    replacement_json = replacement_layers.to_json()
    with open('replacement_layer.json', 'w') as json_file:
        json_file.write(replacement_json)
        
    del replacement_layers
    
    with open('replacement_layer.json', 'r') as json_file:
        replacement_layers = tf.keras.models.model_from_json(json_file.read())

    print('loading replacement layers weights')
    replacement_layers.load_weights('replacement_layer.h5')
    replacement_layers.compile(loss=loss_object, optimizer=optimizer)
    replacement_layers.evaluate_generator(layer_test_gen)
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
    bottom_half.summary()
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

    print('freezing first half of layers')
    for i in range(target):
        new_combined.layers[i].trainable = False
    
    print('freezing last half of layers')
    for i in range(target +  replacement_len, len(new_combined.layers)):
        new_combined.layers[i].trainable = False
        
    new_combined.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    del combined
    gc.collect()

    new_save=tf.keras.callbacks.ModelCheckpoint('./refactor_finetune.h5', 
                                                verbose=1, 
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
    scores = new_combined.evaluate_generator(layer_test_gen.dataset, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    model = tf.keras.Model(inputs=new_combined.input, 
                                outputs=new_combined.output)
    model.summary()
    del new_combined
    gc.collect()
    model.save('./refactor_finetune.h5')
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('./refactor_finetune.h5')
    print('new summary')
    model.summary()
    targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']
    
    

