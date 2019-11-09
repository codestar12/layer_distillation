
# coding: utf-8

# In[1]:


import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import time
import json
import pathlib
import random

# In[2]:


tf.__version__


# In[3]:


tf.executing_eagerly()


# In[4]:


batch_size = 32

#AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def normalize_production(x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', 
                        '--dataset', 
                        help='Which dataset to use (cifar10 or cifar100)',
                        choices=['cifar10', 'cifar100', 'ds+cifar10', 'ds+cifar100'],
                        default='cifar10')
    parser.add_argument('-n',
                        '--norm', 
                        help='How to normalize the images (std or prod)',
                        choices=['std', 'prod'],
                        default='std')

    parser.add_argument('-le',
                        '--layer_train_epochs',
                        help='Number of epochs to retrain layers on real layer activations',
                        type=int,
                        default=50)

    parser.add_argument('-bl',
                    '--block_layers',
                    help='Number of layers in each block',
                    type=int,
                    default=2)

    parser.add_argument('-bn',
                    '--batch_norm',
                    choices=('True', 'False'),
                    default='True')
    
    parser.add_argument('-me',
                        '--model_train_epochs',
                        help='Number of epochs to retrain model for fine tuning',
                        type=int,
                        default=5)                   

    parser.add_argument('-md',
                        '--model_directory',
                        help='File path to save refactored model to',
                        type=str,
                        default='./refactored_model.h5')

    parser.add_argument('-rd',
                        '--replace_directory',
                        help='File path to the model to',
                        type=str,
                        default='./vgg16_cifar10.h5')

    parser.add_argument('-sl',
                        '--save_logs',
                        help='whether to save training logs',
                        type=bool,
                        default=True)

    parser.add_argument('-ld',
                        '--log_dir',
                        help='file path to save logs',
                        default='./logs/refactor_log.json')



    args = parser.parse_args()
    batch_norm = args.batch_norm == 'True'
    model_path, model_name = os.path.split(args.model_directory)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    replace_path, replace_name = os.path.split(args.replace_directory)
    if not os.path.exists(replace_path):
        os.mkdir(replace_path)
    
    log_path = log_name = None
    log = {'model_epocss': args.model_train_epochs,
           'layer_epochs' : args.layer_train_epochs}
    if args.save_logs:
        log_path, log_name = os.path.split(args.log_dir)
        if not os.path.exists(log_path):
            os.mkdir(log_path)



    if args.dataset == 'cifar10' or args.dataset == 'ds+cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    num_predictions = 20


    # The data, split between train and test sets:

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if args.norm == 'std' :
        x_train /= 255
        x_test /= 255
    elif args.norm == 'prod':
        x_train = normalize_production(x_train)
        x_test = normalize_production(x_test)
    else:
        raise("normalize method not recognized use either std or prod")

    
    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model = load_model(replace_path + '/' + replace_name)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])



    # In[ ]:


    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    log['original_acc'] = float(scores[1])
    log['original_loss'] = float(scores[0])

    # In[ ]:


    model.summary()


    # In[ ]:


    targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']


    # In[ ]:
    def preprocess_image(image):
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        if args.norm == 'prod':
            image = normalize_production(image)
        else:
            image /= 255

        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    dataset = None
    if 'ds' not in args.dataset:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        #dataset = dataset.shuffle(buffer)
    else:
        data_root = pathlib.Path('/home/cody/layer-distillation//data/train_32x32/')
        all_image_paths = list(data_root.glob('*'))
        all_image_paths = [str(path) for path in all_image_paths]
        image_count = len(all_image_paths)
        random.shuffle(all_image_paths)
        all_image_labels = [0 for _ in all_image_paths]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        dataset = image_label_ds.shuffle(buffer_size=800000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)


    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    dataset_test = dataset.batch(batch_size)
    dataset_test = dataset.repeat()


    # In[ ]:


    def build_replacement(get_output, layers=2 , batch_norm=True):
        inputs = tf.keras.Input(shape=get_output.output[0].shape[1::])

        #build as many layers as needed
        for i in range(layers-1):
            X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}',
                                                filters=get_output.output[1].shape[-1], 
                                                kernel_size= (3,3),
                                                padding='Same')(inputs)
            if batch_norm:
                X = tf.keras.layers.BatchNormalization(name=f"replacement_batchnorm_{build_replacement.counter}")(X)
            X = tf.keras.layers.ReLU(name=f"replacement_relu_{build_replacement.counter}")(X)
            
            build_replacement.counter += 1

        #at least the last layer must have batch norm
        X = tf.keras.layers.SeparableConv2D(name=f'sep_conv_{build_replacement.counter}',
                                                filters=get_output.output[1].shape[-1], 
                                                kernel_size= (3,3),
                                                padding='Same')(inputs)
        
        X = tf.keras.layers.BatchNormalization(name=f"replacement_batchnorm_{build_replacement.counter}")(X)
        X = tf.keras.layers.ReLU(name=f"replacement_relu_{build_replacement.counter}")(X)
        
        build_replacement.counter += 1
        
        replacement_layers = tf.keras.Model(inputs=inputs, outputs=X)
        
        return replacement_layers

    build_replacement.counter = 0

    # In[ ]:


    import math
    class LayerBatch(tf.keras.utils.Sequence):
        
        def __init__(self, input_model, dataset):
            self.input_model = input_model
            self.dataset = dataset.__iter__()
            
        def __len__(self):

            return math.ceil(50000 / batch_size)
            
        
        def __getitem__(self, index):
            X, y = self.input_model(next(self.dataset))
            return X, y
        
    import math
    class LayerTest(tf.keras.utils.Sequence):
        
        def __init__(self, input_model, dataset):
            self.input_model = input_model
            self.dataset = dataset.__iter__()
            
        def __len__(self):
            return math.ceil(10000 / batch_size)
        
        def __getitem__(self, index):
            X, y = self.input_model(next(self.dataset))
            return X, y


    # In[ ]:


    model.save('./model.h5')

    import gc
    # we ignore the first conv layer 
    num_targets = len(targets) - 1 
    start_time = time.time()
    layer_counter = 1
    log['layer'] = []

    while len(targets) > 1:
        layer_start_time = time.time()
        layer_log = {'layer' : layer_counter}
        
        print(f'targets {targets}')
        print("taking target")
        target = targets[1]
        
        print(f'making output for target layer {target}')
        
        get_output = tf.keras.Model(inputs=model.input, 
                                    outputs=[model.layers[target - 1].output,
                                            model.layers[target].output])
        
        get_output.save('./output.h5')

        tf.keras.backend.clear_session()

        get_output = tf.keras.models.load_model('./output.h5')
        
        print(f'making replacement layers for target layer {target}')
        replacement_layers = build_replacement(get_output, args.block_layers, batch_norm)
        
        replacement_len = len(replacement_layers.layers)
        
        initial_learning_rate = 0.01
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=50000 * (args.layer_train_epochs // 4) // batch_size,
        #     decay_rate=0.2,
        #     staircase=True)
        optimizer= tf.keras.optimizers.RMSprop(learning_rate=initial_learning_rate)
        
#         if 'ds' in args.dataset:
#             learning_rate=.01

        loss_object = tf.losses.MeanSquaredError()
        
        replacement_layers.compile(loss=loss_object, optimizer=optimizer)
        
        save = tf.keras.callbacks.ModelCheckpoint(model_path + '/' + 'replacement_layer.h5', 
                                                verbose=1, 
                                                save_weights_only=True,
                                                save_best_only=True)
        train_gen = LayerBatch(get_output, dataset)
        test_gen = LayerTest(get_output, dataset_test)
        
        ReduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                        patience=5, min_lt=.00001, verbose=1)
        
        earlyStop = tf.keras.callbacks.EarlyStopping(patience=12, verbose=1)

        
        print(f'starting fit generator for target layer {target}')
        replacement_layers.fit_generator(generator=train_gen, 
                                        epochs=args.layer_train_epochs, 
                                        validation_data=test_gen ,
                                        verbose=1, callbacks=[save, ReduceLR,earlyStop])
        


        
        print('saving replacement layers to json')
        
        replacement_json = replacement_layers.to_json()
        with open(model_path + '/' +'replacement_layer.json', 'w') as json_file:
            json_file.write(replacement_json)
            
        del replacement_layers

        tf.keras.backend.clear_session()
        
        with open(model_path + '/' +'replacement_layer.json', 'r') as json_file:
            replacement_layers = tf.keras.models.model_from_json(json_file.read())

        print('loading replacement layers weights')
        replacement_layers.load_weights(model_path + '/' + 'replacement_layer.h5')
        replacement_layers.compile(loss=loss_object, optimizer=optimizer)
        layer_loss = replacement_layers.evaluate_generator(test_gen)
        print(f'layer loss: {layer_loss}')
        
        layer_log['loss'] = float(layer_loss)
        model = tf.keras.models.load_model('./model.h5')
        # build top half of model
        print('building top half of model')
        get_output = tf.keras.Model(inputs=model.input, outputs=[model.layers[target - 1].output])
        # add in replacement layers
        print('building middle of model with replacement layers')
        new_joint = tf.keras.Model(inputs=get_output.input, outputs=replacement_layers(get_output.output))
        
        # build bottom of model
        bottom_half = tf.keras.Sequential()
        for layer in model.layers[target + 1::]:
            bottom_half.add(layer)
        
        print('building bottom of model')
        bottom_half.build(input_shape=new_joint.output.shape)
        print('combining model')
        combined = tf.keras.Model(inputs=new_joint.input, outputs=bottom_half(new_joint.output))
        
        combined.layers[-1].trainable=False
        opt = keras.optimizers.RMSprop(lr=0.00005, decay=1e-6)
        combined.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])


        del bottom_half, new_joint, replacement_layers, model
        
        print('testing combined model')
        scores = combined.evaluate(x_test, y_test, verbose=2)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        layer_log['model_loss'] = float(scores[0])
        layer_log['acc'] = float(scores[1])
        
        old_loss = scores[0]
        
        new_combined = tf.keras.Sequential()
        new_layers = []
        new_combined.add(tf.keras.layers.Input(shape=(32,32,3)))
        accum = 0
        print('refactoring model')
        for layer in combined.layers:
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
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

        new_combined.build()

        accum = 0
        for i, layer in enumerate(combined.layers):
            if hasattr(layer, 'layers'):

                for sublayer in layer.layers:
                    #print(f'{accum} sub is {sublayer} new is {new_combined.layers[accum]}')
                    if(sublayer.__class__.__name__ != 'InputLayer'):              
                        new_combined.layers[accum].set_weights(sublayer.get_weights())
                        accum += 1
        #             else:
        #                 accum += 1
                continue
            else:
                #print(layer)
                if(layer.__class__.__name__ != 'InputLayer'):
                    new_combined.layers[accum].set_weights(layer.get_weights())
                    accum +=1 

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

        new_save=tf.keras.callbacks.ModelCheckpoint(model_path + '/' + model_name, 
                                                    verbose=1, 
                                                    save_weights_only=False, 
                                                    save_best_only=True)
        print('fine tuning combined model')
        
        
        new_combined.save_weights(model_path + '/' + model_name + 'holdout')
# #         new_combined.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=5, callbacks=[new_save])
        if args.model_train_epochs > 0:
            new_combined.fit(
                        dataset,
                        epochs=args.model_train_epochs,
                        validation_data=dataset_test,
                                #workers=5,
                        callbacks=[new_save])
        
            print('loading best weights from fine tune')
            new_combined.load_weights(model_path + '/' + model_name)

            print('testing fine-tuned combined model')
            scores = new_combined.evaluate(x_test, y_test, verbose=2)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
            layer_log['fine_tune_model_loss'] = float(scores[0])
            layer_log['fine_tune_acc'] = float(scores[1]) 
            
            if old_loss < scores[0]:
                print('loading none finetuned weightss')
                new_combined.load_weights(model_path + '/' + model_name + 'holdout')

        layer_end_time = time.time()
        layer_log['train_time'] = float(layer_end_time - layer_start_time)
        log['layer'].append(layer_log)
        
        
            
        
        model = tf.keras.Model(inputs=new_combined.input, outputs=new_combined.output)
        
        del new_combined
        #print('new summary')
        model.save('./model.h5')
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model('./model.h5')
        #model.summary()
        targets = [i for i, layer in enumerate(model.layers) if layer.__class__.__name__ == 'Conv2D']
        layer_counter += 1
    
    end_time = time.time()
    log['train_time'] = float(end_time - start_time)

    model.summary()
    if args.save_logs:
        with open(log_path + '/' + log_name, 'w') as f:
            json.dump(log, f)
        

