# Code to allow swapping between network architectures so that all network parameters 
# can be controlled from the config.json file. Includes multiple external networks in 
# a switch statement depending on the desired architecture

import numpy as np
import keras as K
import tensorflow as tf
import lib.model as model
from tensorflow.keras import applications

def load_swappable_network(config, weights, just_vent=True):
    """
    Possible networks: alexNet, GoogleNet, VGG16, ResNet50, ResNet50V2, ResNet101V2, MobileNet
    """
    if config['model']['architecture'] == 'alexNet':
        classification_model = model.alexNet(config['NCLASSES'])
    elif config['model']['architecture'] == 'googleNet':
        classification_model = model.googleNet(config['NCLASSES'])

    else:
        # Set up network
        if just_vent:
            keras_input = K.layers.Input(shape=(config['IMAGE_SIZE'], config['IMAGE_SIZE'], 1))
            keras_concat = K.layers.Concatenate()([keras_input, keras_input, keras_input])
        else:
            keras_concat = K.layers.Input(shape=(config['IMAGE_SIZE'], config['IMAGE_SIZE'], 3))

        if config['model']['architecture'] == 'VGG16':
            insert_model = applications.VGG16(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] =='ResNet50':
            insert_model = applications.ResNet50(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] == 'ResNet50V2':
                insert_model = applications.ResNet50V2(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
                
        elif config['model']['architecture'] == 'ResNet101V2':
            insert_model = applications.ResNet101V2(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] == 'MobileNet':
            insert_model = applications.MobileNet(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] == 'ResNet152V2':
            insert_model = applications.ResNet152V2(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] == 'InceptionResNetV2':
            insert_model = applications.InceptionResNetV2(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat) 
        elif config['model']['architecture'] == 'NASNetLarge':
            insert_model = applications.NASNetLarge(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
        elif config['model']['architecture'] == 'ConvNeXtXLarge':
            insert_model = applications.ConvNeXtXLarge(include_top=False,
                                                        weights=config['model']['weights'],
                                                        input_shape=(192, 192, 3),
                                                        classes=config['NCLASSES'],
                                                        input_tensor = keras_concat)
            
        # Exploding gradient problems seem to be common for these models, iteratively change regularization
        # in loaded model layers to hopefully eliminate exploding gradients

        insert_model.trainable = False # This keeps batchnorm layers functioning when you unfreeze layers
            
        flat = K.layers.Flatten()(insert_model.output, training=False)
        drop = K.layers.Dropout(0.4)(flat)
        intermediate = K.layers.Dense(512, 
                                    activation='relu',
                                    kernel_initializer=K.initializers.HeNormal(),
                                    bias_initializer=K.initializers.Zeros(),
                                    kernel_regularizer=K.regularizers.L1L2(1e-5, 1e-4,),
                                    bias_regularizer=K.regularizers.L2(1e-4))(drop)
        intermediate._name = "Intermediate"
        predictions = K.layers.Dense(config['NCLASSES'], activation='softmax')(intermediate)
        predictions._name = "Predictions"
        classification_model = K.models.Model(insert_model.input, outputs=predictions)

    regularizer = K.regularizers.L1L2(1e-5, 1e-4,)
    for layer in classification_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = regularizer

    # if config['model']['freeze_layers']>0:
    #     for layer in classification_model.layers[:config['model']['freeze_layers']]:
    #         layer.trainable=False
    #     for layer in classification_model.layers[config['model']['freeze_layers']:]:
    #         layer.trainable=True

    classification_model.layers[-1].weights[1] = np.log(weights) # initialize bias appropriately for imbalanced class

    return classification_model