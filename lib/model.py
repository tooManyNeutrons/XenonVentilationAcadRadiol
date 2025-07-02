# Deep Learning Models
# Alex Matheson 9/12/2023
#
# Functions and classes to call specific models implemented in Keras

import math
import keras as K
import tensorflow as tf
from lib.layer import SpatialTransformer
import numpy as np


### UNET ######################################################################################
# The most common framework for image analysis
# Implemented here in a block format to simply iterate on different UNet implementations

"""
Based on U-Net paper: https://arxiv.org/abs/1505.04597
Impletmented based on keras tutorial by Nikhil Tomar: 
https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/
blob/master/unet-segmentation.ipynb

The network may be subdivided into intermediate-level 'blocks' of individual
layers. Different UNets may be constructed by applying different numbers
of these layers.
"""

#Downsampling blocks
def downSample(x, filters, kernelSize=(3,3), padding='same', strides=1):
    """
    for each downsample block, two convolutions are performed (each reduces tensor
    dimensions by 2) followed by max-pooling (making the image 1/4 the original size
    by taking the max value in each 2x2 cell region)
    """
    con = K.layers.Conv2D(filters, 
                              kernelSize, 
                              padding=padding, 
                              strides=strides, 
                              activation="relu")(x)
    con = K.layers.Conv2D(filters, 
                              kernelSize, 
                              padding=padding, 
                              strides=strides, 
                              activation="relu")(con)
    pool = K.layers.MaxPool2D((2, 2), (2, 2))(con)
    drop = K.layers.Dropout(0.5)(pool)
    
    return con, pool
    
def upSample(x, skip, filters, kernelSize=(3,3), padding='same', strides=1):
    """
    for each upsample block, input is taken from the "lower" layer via upsampling
    and concatenated with the corresponding downsample pooling output. This multi-
    channel input is then convolved twice
    """
    us     = K.layers.UpSampling2D((2,2))(x)
    concat = K.layers.Concatenate()([us, skip])
    drop = K.layers.Dropout(0.5)(concat)
    con = K.layers.Conv2D(filters, 
                              kernelSize, 
                              padding=padding, 
                              strides=strides, 
                              activation="relu")(drop)
    con = K.layers.Conv2D(filters, 
                              kernelSize, 
                              padding=padding, 
                              strides=strides, 
                              activation="relu")(con)
    
    return con

def bottleNeck(x, filters, kernalSize=(3,3), padding='same', strides=1):
    """
    'lowest' block of the UNet. Maxium feature depth is achieved here, and no skipping
    takes place at this block only. Input is the final downsampling block. Connects to
    the first upsampling block
    """
    con = K.layers.Conv2D(filters, 
                              kernalSize, 
                              padding=padding, 
                              strides=strides, 
                              activation='relu')(x)
    con = K.layers.Conv2D(filters, 
                              kernalSize, 
                              padding=padding, 
                              strides=strides, 
                              activation='relu')(con)
    return con
    
def denseFeatureStack(x, filters, kernalSize=(3,3), padding='same', strides=1):
    """
    A single unit for a dense v-net implementation. This block includes series of densely 
    connected convolutional layers, with two outputs. The first is a down-sample, to connect
    to other blocks, the second a convolutional layer that is merged with the convolutions
    at other feature depths.
    """
    con1 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(x)
    
    con1 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(con1)
    concat1 = K.layers.Concatenate()([x, con1])
    
    con2 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(concat1)
    concat2 = K.layers.Concatenate()([concat1, con2])
    
    con3 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(concat2)
    concat3 = K.layers.Concatenate()([concat2, con3])
    
    con4 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(concat3)
    concat4 = K.layers.Concatenate()([concat3, con4])
    
    con5 = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=strides,
                              activation='relu')(concat4)
    down = K.layers.Conv2D(filters,
                              kernalSize,
                              padding=padding,
                              strides=2,
                              activation='relu')(concat4)
    return down, con5
    
def UNet():
    """
    Classical UNet architecture
    https://arxiv.org/abs/1505.04597

    """
    feat = [16, 32, 64, 128, 256]
    imageSize = 128
    inputs = K.layers.Input((imageSize, imageSize, 1))
    
    p0 = inputs
    c1, p1 = downSample(p0, feat[0]) #turns 128x128 to 64x64, block 1
    c2, p2 = downSample(p1, feat[1]) #turns 64x64   to 32x32, block 2
    c3, p3 = downSample(p2, feat[2]) #turns 32x32   to 16x16, block 3
    c4, p4 = downSample(p3, feat[3]) #turns 16x16   to 8x8  , block 4
    
    bot    = bottleNeck(p4, feat[4])
    
    u1     = upSample(bot, c4, feat[3]) #turns 8x8    to 16x16  , block 4
    u2     = upSample(u1 , c3, feat[2]) #turns 16x16  to 32x32  , block 3
    u3     = upSample(u2 , c2, feat[2]) #turns 32x32  to 64x64  , block 2
    u4     = upSample(u3 , c1, feat[2]) #turns 64x64  to 128x128, block 1
    
    outputs = K.layers.Conv2D(3, (1,1), padding="same", activation='softmax')(u4)
    model = K.models.Model(inputs, outputs)
    
    return model

def multiUNet():
    """
    Multi-channel UNet that takes two images (different modality) for feature selection
    Does NOT concatenate into a single image on the first step. Instead, separate downsampling
    paths get individual features for each channel and merge into the upsampling path.
    """
    feat = [16, 32, 64, 128, 256]
    imageSize = 128
    inputProton = K.layers.Input((imageSize, imageSize, 1))
    inputGas    = K.layers.Input((imageSize, imageSize, 1))
    
    p0 = inputProton
    g0 = inputGas
    
    # Proton Downsampling Path
    c1, p1 = downSample(p0, feat[0]) #turns 128x128 to 64x64, block 1
    c2, p2 = downSample(p1, feat[1]) #turns 64x64   to 32x32, block 2
    c3, p3 = downSample(p2, feat[2]) #turns 32x32   to 16x16, block 3
    c4, p4 = downSample(p3, feat[3]) #turns 16x16   to 8x8  , block 4
    
    # Gas Downsampling Path
    d1, g1 = downSample(g0, feat[0]) #turns 128x128 to 64x64, block 1
    d2, g2 = downSample(g1, feat[1]) #turns 64x64   to 32x32, block 2
    d3, g3 = downSample(g2, feat[2]) #turns 32x32   to 16x16, block 3
    d4, g4 = downSample(g3, feat[3]) #turns 16x16   to 8x8  , block 4
    
    # Bridge
    con    = K.layers.Concatenate()([p4, g4])
    bot    = bottleNeck(con, feat[4])
    
    # Concatenate skip paths
    s4     = K.layers.Concatenate()([c4, d4])
    s3     = K.layers.Concatenate()([c3, d3])
    s2     = K.layers.Concatenate()([c2, d2])
    s1     = K.layers.Concatenate()([c1, d1])
    
    # Upsampling Path
    u4     = upSample(bot, s4, feat[3]) #turns 8x8    to 16x16  , block 4
    u3     = upSample(u4 , s3, feat[2]) #turns 16x16  to 32x32  , block 3
    u2     = upSample(u3 , s2, feat[1]) #turns 32x32  to 64x64  , block 2
    u1     = upSample(u2 , s1, feat[0]) #turns 64x64  to 128x128, block 1
    
    outputs = K.layers.Conv2D(3, (1,1), padding="same", activation='softmax')(u1)
    model = K.models.Model([inputProton, inputGas], outputs)
    
    return model

### VNet ##########################################################################
# Precursor to a UNet
# More memory efficient but fewer convolutions on the upsampling path
# https://arxiv.org/abs/1606.04797

def denseVNet():
    """
    VNet with dense connections on each downsampling block
    """
    feat = [4,8,16]
    imageSize = 128
    inputs = K.layers.Input((imageSize, imageSize, 1))
    
    #d - downsampled feature stack
    #c - feature stack for up branch concatenation
    d1, c1 = denseFeatureStack(inputs, feat[0])
    d2, c2 = denseFeatureStack(d1, feat[1])
    d3, c3 = denseFeatureStack(d2, feat[2])
    
    u2 = K.layers.UpSampling2D(size=(2, 2),
                                   data_format=None,
                                   interpolation='bilinear')(c2)
    u3 = K.layers.UpSampling2D(size=(4, 4),
                                   data_format=None,
                                   interpolation='bilinear')(c3)
    concat = K.layers.Concatenate()([c1, u2, u3])
    
    outputs = K.layers.Conv2D(3, 3, padding="same", activation='softmax')(concat)
    model = K.models.Model(inputs, outputs)
    
    return model

def multiVNet():
    """
    VNet with multi-channel support for multimodal imaging
    """
    feat = [4,8,16]
    imageSize = 128
    proInput = K.layers.Input((imageSize, imageSize, 1))
    gasInput = K.layers.Input((imageSize, imageSize, 1))
    
    proD1, proC1 = denseFeatureStack(proInput, feat[0])
    proD2, proC2 = denseFeatureStack(proD1, feat[1])
    proD3, proC3 = denseFeatureStack(proD2, feat[2])
    
    gasD1, gasC1 = denseFeatureStack(gasInput, feat[0])
    gasD2, gasC2 = denseFeatureStack(gasD1, feat[1])
    gasD3, gasC3 = denseFeatureStack(gasD2, feat[2])
    
    proU2 = K.layers.UpSampling2D(size=(2, 2),
                                   data_format=None,
                                   interpolation='bilinear')(proC2)
    proU3 = K.layers.UpSampling2D(size=(4, 4),
                                   data_format=None,
                                   interpolation='bilinear')(proC3)
    gasU2 = K.layers.UpSampling2D(size=(2, 2),
                                   data_format=None,
                                   interpolation='bilinear')(gasC2)
    gasU3 = K.layers.UpSampling2D(size=(4, 4),
                                   data_format=None,
                                   interpolation='bilinear')(gasC3)
    
    concat = K.layers.Concatenate()([proC1, proU2, proU3, gasC1, gasU2, gasU3])
    
    outputs = K.layers.Conv2D(3, 3, padding='same', activation='softmax')(concat)
    model = K.models.Model([proInput, gasInput], outputs)
    
    return model


### Registration Networks #############################################################

def params_to_tform(input_tensor):
    """
    Turns numbers on the range [0, 1] to a transform matrix. Current function is hard
    coded for specific transform ranges and needs to be re-written for general transforms.
    Generates forward and inverse transforms for debugging and general utilies.
    """
    batch_size = tf.shape(input_tensor)[0]
    
    # Matrices necessary for working in integer coordinates
    t_pos = tf.constant([[1., 0., 64.], [0., 1., 64.], [0., 0., 1.]])
    t_pos_tile = tf.tile(tf.expand_dims(t_pos,0), [batch_size, 1, 1])
    t_neg = tf.constant([[1., 0., -64.], [0., 1., -64.], [0., 0., 1.]])
    t_neg_tile = tf.tile(tf.expand_dims(t_neg,0), [batch_size, 1, 1])
    
    # Matrices necessary for matrix augmentation
    bot = tf.constant([[[0., 0., 1.]]])
    bot_tile = tf.tile(bot, [batch_size, 1, 1])
    
    # Convert the 0->1 scaled parameters to their real values
    angle_element = tf.slice(input_tensor, [0,0], [batch_size, 1]) * (15. * math.pi / 180.)
    scale_factor_element = tf.slice(input_tensor, [0,1], [batch_size, 1])
    scale_element = scale_factor_element*0.1 + tf.ones_like(scale_factor_element, dtype=tf.float32)
    trans_element = tf.slice(input_tensor, [0,2], [batch_size, 2]) * 12.8
    
    # Create rotation, scale, and translation matrix elements
    cos = tf.reshape(tf.math.cos(angle_element), [batch_size, 1, 1])
    sin = tf.reshape(tf.math.sin(angle_element), [batch_size, 1, 1])
    rrow0 = tf.concat([cos, -1*sin], 2)
    rrow1 = tf.concat([sin, cos], 2)
    rotate = tf.concat([rrow0, rrow1], 1)
    
    scale = tf.multiply(tf.tile(tf.expand_dims(scale_element, -1), [1, 2, 2]), tf.eye(2, batch_shape=[batch_size]))
    
    translate = tf.expand_dims(trans_element,-1)
    
    # Find linear map A 
    A = tf.matmul(rotate, scale)
    
    # Forward transform
    aug_forward = tf.concat([tf.concat([A, translate], 2), bot_tile], 1)
    forward = tf.matmul(tf.matmul(t_pos_tile, aug_forward), t_neg_tile)
    forward_6 = tf.slice(forward, [0,0,0], [batch_size, 2, 3])
    
    # Inverse transform
    A_inv = tf.linalg.inv(A)
    trans_inv = tf.matmul(-1.*A_inv, translate)
    aug_inv = tf.concat([tf.concat([A_inv, trans_inv], 2), bot_tile], 1)
    inverse = tf.matmul(tf.matmul(t_pos_tile, aug_inv), t_neg_tile)
    inverse_6 = tf.slice(inverse, [0,0,0], [batch_size, 2, 3])
    
    return forward_6

def fcn_spatial_transformer(image_size=128):
    """
    Model to register multimodal images in 2D. Network includes a convolutional path to extract image
    features. Convolutional information is concatenated and fed into a series of dense layers like
    a traditional neural network. Dense layers output predicted numbers on the range [0, 1] representing
    how much scaling/rotation/translation is required to register. Params_to_tform turns these 
    numbers into a a transform matrix. A spatial transformer layer applies the transform to the 
    input image. This architecture reduces the black box nature of the neural network as the 
    transform itself can be recorded. This preserves image information. Including a spatial transformer
    allows the network to be a one-stop registration tool and allows image comparison loss functions.
    """
    input_proton = K.layers.Input((image_size, image_size, 1))
    input_gas = K.layers.Input((image_size, image_size, 1))

    v1Pro =  K.layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_proton)
    v2Pro =  K.layers.Conv2D(16, (3,3), activation='relu', padding='same')(v1Pro)
    v1Gas =  K.layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_gas)
    v2Gas =  K.layers.Conv2D(16, (3,3), activation='relu', padding='same')(v1Gas)
    
    concat = K.layers.Concatenate()([v1Pro, v2Pro, v1Gas, v2Gas])
    drop   = K.layers.Dropout(0.5)(concat)
    f1     = K.layers.Flatten()(drop)
    
    d1 = K.layers.Dense(64, activation='relu')(f1)
    d2 = K.layers.Dense(64, activation='relu')(d1)
    d3 = K.layers.Dense(64, activation='relu')(d2)
    params = K.layers.Dense(3, activation='linear', name='params')(d3)
    
    tforms = K.layers.Lambda(params_to_tform, name='tforms')(params)
    
    registered_image = SpatialTransformer((128,128), name="registered_image")([input_proton, tforms])
    model = K.models.Model([input_proton, input_gas], [params, registered_image])
    
    return model

##### Dense Networks ############################################################################
# Networks that condense information from convolutional layers to dense layers for classification

def alexNet(n_classes, image_size=192, include_top=True):
    """
    AlexNET implementation
    A Krizhevsky 2012

    Current padding assumes a 192x192 image
    """
    if include_top:
        input = K.layers.Input((image_size, image_size, 1))
    input = K.layers.Resizing(192, 192, interpolation='nearest')(input)
    zp    = K.layers.ZeroPadding2D(3)(input) # need to pad to 195 for pooling math
    c1    = K.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='valid')(zp)
    p1    = K.layers.MaxPool2D((3, 3), (2, 2))(c1)
    c2    = K.layers.Conv2D(256, (5,5), activation='relu', padding='valid')(p1)
    p2    = K.layers.MaxPool2D((3, 3), (2, 2))(c2)
    c3    = K.layers.Conv2D(384, (3,3), activation='relu', padding='same')(p2)
    c4    = K.layers.Conv2D(384, (3,3), activation='relu', padding='same')(c3)
    c5    = K.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    p3    = K.layers.MaxPool2D((3,3), (2,2))(c5)

    flat  = K.layers.Flatten()(p3)
    d1    = K.layers.Dense(4096, activation='relu')(flat)
    d2    = K.layers.Dense(4096, activation='relu')(d1)
    out   = K.layers.Dense(n_classes, activation='softmax')(d2)
    model = K.models.Model(input, out)

    return model

def inception_module(x, filter_size):
    """
    GoogleNet networks are also known as inception networks. This is a fundamental inception
    block.
    """
    c1  = K.layers.Conv2D(filter_size[0], (1,1), activation='relu', padding='same')(x)
    c3  = K.layers.Conv2D(filter_size[1], (3,3), activation='relu', padding='same')(x)
    c3  = K.layers.Conv2D(filter_size[2], (1,1), activation='relu', padding='same')(c3)
    c5  = K.layers.Conv2D(filter_size[3], (5,5), activation='relu', padding='same')(x)
    c5  = K.layers.Conv2D(filter_size[4], (1,1), activation='relu', padding='same')(c5)
    m1  = K.layers.MaxPool2D((3,3), (1,1), padding='same')(x)
    m1  = K.layers.Conv2D(filter_size[5], (1,1), activation='relu', padding='same')(m1)
    cat = K.layers.Concatenate()([c1, c3, c5, m1])

    return cat 

def auxillary_module(x, n_classes):
    """
    Auxillary branches act as a skip-connection between the loss function and earlier portions
    of the network. Helps reduce vanishing gradient descent. This implementation is currently
    specific to googleNet. May be able to generalize in future for application to multiple
    network architectures.
    Arguments:
    x - previous layer in neural network
    Returns: 
    out - the final layer in the branch for network compilation
    """

    ap1   = K.layers.AveragePooling2D((5,5), strides=(3,3))(x)
    c1    = K.layers.Conv2D(128, 1, padding='same', activation='relu')(ap1)
    f1    = K.layers.Flatten()(c1)
    d1    = K.layers.Dense(1024, activation='relu')(f1)
    dr    = K.layers.Dropout(0.7)(d1)
    d2    = K.layers.Dense(n_classes, activation='softmax')(dr)

    return d2

def googleNet(n_classes, image_size=192, include_top=True):
    """
    Based on googleNet architecture
    Paper: Szegedy et al. Going deeper with convolutions. Arxiv 2014.
    """
    input = K.layers.Input((image_size, image_size, 1))
    input = K.layers.Resizing(192, 192, interpolation='nearest')(input)
    c1    = K.layers.Conv2D(64, 6, (2,2), activation='relu', padding='same')(input)
    m1    = K.layers.MaxPool2D((3,3), (2,2))(c1)
    c2    = K.layers.Conv2D(64, (3,3), activation='relu', padding='same')(m1)
    c3    = K.layers.Conv2D(192, (3,3), activation='relu', padding='same')(c2)
    m2    = K.layers.MaxPool2D((2,2), (2,2))(c3)

    i1    = inception_module(m2, [64, 96, 128, 16, 32, 32]) #still don't understand these numbers
    i2    = inception_module(i1, [128, 128, 192, 32, 96, 64])
    m3    = K.layers.MaxPool2D((3,3), (2,2))(i2)
    i3    = inception_module(m3, [128, 128, 192, 32, 96, 64])

    aux1  = auxillary_module(i3, n_classes)

    i4    = inception_module(i3, [160, 112, 224, 24, 64, 64])
    i5    = inception_module(i4, [128, 128, 256, 24, 64, 64])
    i6    = inception_module(i5, [112, 144, 288, 32, 64, 64])

    aux2  = auxillary_module(i6, n_classes)

    i7    = inception_module(i6, [256, 160, 320, 32, 128, 128])
    mp    = K.layers.MaxPool2D((3,3), (2,2))(i7)
    i8    = inception_module(mp, [256, 160, 320, 2, 128, 128])
    i9    = inception_module(i8, [384, 192, 384, 48, 128, 128])

    gap   = K.layers.GlobalAveragePooling2D()(i9)
    dr    = K.layers.Dropout(0.4)(gap)

    out   = K.layers.Dense(n_classes, activation='softmax', name="Output")(dr)
    model = K.models.Model(input, [out, aux1, aux2])

    return model


def vggNet(n_classes, image_size=192):
    input = K.layers.Input((image_size, image_size, 1))
    input = K.layers.Resizing(192, 192, interpolation='nearest')(input)
    c1    = K.layers.Conv2D(64, (3,3), activation='relu', padding='same')(input)
    c2    = K.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    
    p1    = K.layers.MaxPool2D((2,2), (2,2))(c2)
    c3    = K.layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c4    = K.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)

    p2    = K.layers.MaxPool2D((2,2), (2,2))(c4)
    c5    = K.layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c6    = K.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c5)
    c7    = K.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c6)

    p3    = K.layers.MaxPool2D((2,2), (2,2))(c7)
    c8    = K.layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c9    = K.layers.Conv2D(512, (3,3), activation='relu', padding='same')(c8)
    c10   = K.layers.Conv2D(512, (3,3), activation='relu', padding='same')(c9)

    p4    = K.layers.MaxPool2D((2,2), (2,2))(c10)
    flat  = K.layers.Flatten()(p4)
    d1    = K.layers.Dense(4096, activation='relu')(flat)
    d2    = K.layers.Dense(4096, activation='relu')(d1)
    out   = K.layers.Dense(n_classes, activation='softmax')(d2)

    model = K.models.Model(input, out)

    return model

### For RESNET use Keras built-in networks??

def resNet(n_classes, image_size=128, initial_features=32):
    """
    WORK IN PROGRESS - NOT FINISHED
    Alex Matheson Nov 17 2023
    """
    def residual_block(x, n_filters):
        c1 = K.layers.Conv2D(n_filters, (3,3), (1,1), activation='relu', padding='same')(x)
        b1 = K.layers.BatchNormalization()(c1)
        c2 = K.layers.Conv2D(n_filters, (3,3), (1,1), activation='relu', padding='same')(b1)
        b2 = K.layers.BatchNormalization()(c2)

        cat  = K.layers.Concatenate()([b2, x])
        relu = K.layers.ReLU()(cat)

        return relu
    
    def stack_blocks(x, blocks=16):
        for block in range(blocks):
            filter_size = 10
    
    input  = K.layers.Input(image_size)
    c1     = K.layers.Conv2D(initial_features, (3,3), (1,1), activation='relu', padding='same')
    b1     = K.layers.BatchNormalization()(c1)
    r1     = stack_blocks(b1)
    gp     = K.layers.GlobalAveragePooling2D()(r1)
    flat   = K.layers.Flatten()(gp)
    output = K.layers.Dense(n_classes, activation='softmax')(flat)

    model = K.models.Model(input, output)


def denseClassCNN(n_classes, image_size=128): #Still need to compute layer size
    """
    Uses a combination of densely connected CNN and dense layers for classification 
    Based on an architecture used in Cui et al. Comp Med Imag Graphics 2019.
    """
    def denseBlock(x, filter_size):
        c1    = K.layers.Conv2D(filter_size, (3,3), activation='relu', padding='same')(x)
        b1    = K.layers.BatchNormalization()(c1)
        c2    = K.layers.Conv2D(filter_size, (3,3), strides=(2,2), activation='relu')(b1)
        c3    = K.layers.Conv2D(filter_size, (3,3), activation='relu', padding='same')(c2)
        cat1  = K.layers.Concatenate()([c2, c3])
        b2    = K.layers.BatchNormalization()(cat1)

        return b2
    
    input = K.layers.Input((image_size, image_size, 1))
    
    conv_block_1 = denseBlock(input, 32)
    conv_block_2 = denseBlock(conv_block_1, 64)
    conv_block_3 = denseBlock(conv_block_2, 128)

    final_conv_1 = K.layers.Conv2D(64, (2,2), activation='relu', padding='valid')(conv_block_3)
    final_conv_2 = K.layers.Conv2D(32, (2,2), activation='relu', padding='valid')(final_conv_1)
    drop         = K.layers.Dropout(0.5)(final_conv_2)
    flat         = K.layers.Flatten()(drop)
    d1           = K.layers.Dense(512)(flat)
    d2           = K.layers.Dense(218)(d1)
    out          = K.layers.Dense(n_classes, activation='softmax')(d2)

    model = K.models.Model(input, out)

    return model

##### Transformer Networks ############################################################################
# Swin transformers - more recent network architectures than mid-2000s CNNs
# Below functions are adapted from keras online tutorial for swin-transformers

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = K.ops.reshape(
        x,
        (
            -1,
            patch_num_y,
            window_size,
            patch_num_x,
            window_size,
            channels,
        ),
    )
    x = K.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = K.ops.reshape(x, (-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = K.ops.reshape(
        windows,
        (
            -1,
            patch_num_y,
            patch_num_x,
            window_size,
            window_size,
            channels,
        ),
    )
    x = K.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = K.ops.reshape(x, (-1, height, width, channels))
    return x

class WindowAttention(K.layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = K.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = K.layers.Dropout(dropout_rate)
        self.proj = K.layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=K.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = K.Variable(
            initializer=relative_position_index,
            shape=relative_position_index.shape,
            dtype="int",
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = K.ops.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = K.ops.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = K.ops.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = K.ops.reshape(self.relative_position_index, (-1,))
        relative_position_bias = K.ops.take(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,
        )
        relative_position_bias = K.ops.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = K.ops.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + K.ops.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = ops.cast(
                K.ops.expand_dims(K.ops.expand_dims(mask, axis=1), axis=0),
                "float32",
            )
            attn = K.ops.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = K.ops.reshape(attn, (-1, self.num_heads, size, size))
            attn = K.activations.softmax(attn, axis=-1)
        else:
            attn = K.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = K.ops.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = K.ops.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(K.layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = K.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = K.layers.Dropout(dropout_rate)
        self.norm2 = K.layers.LayerNormalization(epsilon=1e-5)

        self.mlp = K.Sequential(
            [
                K.layers.Dense(num_mlp),
                K.layers.Activation(K.activations.gelu),
                K.layers.Dropout(dropout_rate),
                K.layers.Dense(dim),
                K.layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = K.ops.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = K.ops.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = K.ops.expand_dims(mask_windows, axis=1) - K.ops.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = K.ops.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = K.ops.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = K.Variable(
                initializer=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = K.ops.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = K.ops.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = K.ops.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = K.ops.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = K.ops.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = K.ops.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

# Using tf ops since it is only used in tf.data.
def patch_extract(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, patch_size[0], patch_size[1], 1),
        strides=(1, patch_size[0], patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(K.layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = K.layers.Dense(embed_dim)
        self.pos_embed = K.layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = K.ops.arange(start=0, stop=self.num_patch)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(K.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = K.layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = K.ops.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = K.ops.concatenate((x0, x1, x2, x3), axis=-1)
        x = K.ops.reshape(x, (-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)





