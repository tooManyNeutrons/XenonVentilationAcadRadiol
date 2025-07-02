# Deep Learning Loss Functions
# Alex Matheson 9/12/2023
#
# Custom loss functions for neural networks
# Loss functions follow Tensorflow conventions:
# true - the ground truth provided during training
# pred - network output to be tested against the prediction
#
# Loss functions are listed alphabetically

import math
import keras as K
import tensorflow as tf
import numpy as np
from sklearn.utils.extmath import cartesian

def categorical_crossentropy_custom(epsilon=0.0000001):
    def loss(y_true, y_pred):
        y_pred = tf.math.divide(y_pred, tf.math.reduce_sum(y_pred, axis=-1, keepdims=True))
        return tf.math.reduce_sum(tf.math.multiply(y_true, -1*tf.math.log(y_pred+epsilon)), keepdims=False)
    
    return loss

def categorical_focal_loss(gamma=2., alpha=4., smoothing=0.):
    def smooth_labels(labels, smoothing=0.):
        labels = labels * (1. - tf.cast(smoothing, tf.float32)*tf.cast(tf.shape(labels)[-1], tf.float32))
        labels = labels + (smoothing)
        return labels
    
    def loss (y_true, y_pred):
        y_true = smooth_labels(y_true, smoothing=smoothing)
        model_out = tf.math.add(y_pred, K.backend.epsilon())
        ce = tf.math.multiply (y_true, -tf.math.log(model_out))
        weight = tf.math.multiply(y_true, tf.math.pow(tf.math.subtract(1., model_out), gamma))
        f1 = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
        reduced_f1 = tf.reduce_max(f1, axis=1)
        return tf.reduce_mean(reduced_f1)
    
    return loss

def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.

    For example:

    A = [[1, 2],
         [3, 4]]

    B = [[1, 2],
         [3, 4]]

    should return:

        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D

def correlationLoss(y_true, y_pred):
    """
    Cross-correlation based loss 

    Conventions:
    y_true  - the source gas image (previously registered by observer)
    y_pred  - the transformation matrix calculated by the neural network
    trans   - an 8 element tensor defining the mapping in tensorflow
    gas_reg - the attempted registration for the gas image
    """
    
    def normalizedCrossCorrelation(inputs):
        """
        Performs image-wise zero-normalized cross correlation
        img1  - 1st image, zero-mean
        img2  - 2nd image (should be same shape as img 1, no checks currently), zero-mean
        n     - number of pixels in img
        denom - product of the variance in each image
        """
        img1 = inputs[0] - K.mean(inputs[0]) / (K.std(inputs[0]) * inputs[0].shape[0] * inputs[0].shape[1])
        img2 = inputs[1] - K.mean(inputs[1]) / (K.std(inputs[1]))
        pad = img2.shape[1] - 1 # Determine padding needed for NCC, tf default uses wrong cropping
        num = tf.nn.conv2d(tf.expand_dims(img1, 0),  # H,W,C -> 1,H,W,C
                           tf.expand_dims(img2, 2),  # H,W,C -> H,W,C,1
                           strides=[1,1,1,1],
                           #padding="SAME")
                           padding=[[0,0],[pad, pad],[pad, pad],[0,0]])  # Result of conv is 1,H,W,1
        return num
    
    output = tf.map_fn(
        lambda inputs : normalizedCrossCorrelation(inputs),
        elems=[y_true, y_pred],
        dtype=tf.float32,
     )
    # The best possible correlation is autocorrelation with the true label, use this to normalize
    norm = tf.map_fn(
        lambda inputs : normalizedCrossCorrelation(inputs),
        elems=[y_true, y_true],
        dtype=tf.float32)
    final_output = output[:, 0, :, :, 0]
    final_norm   = norm[:,0,:,:,0] # B,1,H,W,1 -> B,H,W
    print(final_output)
    print(final_norm)
    print(np.sum(final_output), np.sum(final_norm))
    norm_correlation = K.sum(final_output,(1,2)) / K.sum(final_norm, (1,2))
    
    return tf.math.reduce_mean(norm_correlation)

def dice(y_true, y_pred, eps=0.001): #TESTED SUCCESSFULLY
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    #eps required to prevent errors in zero-filled slices
    dsc = ((2 * K.sum(K.abs(y_true * y_pred)) + eps) / 
            (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + eps))
    return dsc

def diceAndCrossEntropy(y_true, y_pred):
    """
    Combines generalized Dice and cross-entropy for multi-class segmentation
    """
    cce = K.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred) + generalizedDiceLoss(y_true, y_pred)

def diceLoss(y_true, y_pred): #TESTED SUCCESSFULLY
    return 1-dice(y_true, y_pred)

# Below versions of DICE included to support IkeML Code
# Below versions will crash for empty data sets

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dyn_weighted_bincrossentropy(y_true, y_pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    num_pred = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) + K.sum(y_true)
    
    # get weight of values in 'pos' category
    zero_weight =  K.sum(y_true)/ num_pred +  K.backend.epsilon() 
    
    # get weight of values in 'false' category
    one_weight = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) / num_pred +  K.backend.epsilon()
    
    # calculate the weight vector
    weights =  (1.0 - y_true) * zero_weight +  y_true * one_weight 
    
    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return K.mean(weighted_bin_crossentropy)

def f1_weighted(y_true, y_pred):
    ground_positives = tf.math.reduce_sum(y_true, axis=0) + K.backend.epsilon()
    pred_positives = tf.math.reduce_sum(y_pred, axis=0) + K.backend.epsilon()
    true_positives = tf.math.reduce_sum(y_true * y_pred, axis=0) + K.backend.epsilon()

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.backend.epsilon())

    weighted_f1 = f1 * ground_positives / tf.math.reduce_sum(ground_positives)
    weighted_f1 = tf.math.reduce_sum(weighted_f1)

    return 1 - weighted_f1


def generalizedDice(y_true, y_pred):
    """
    Generalized version of DICE for multiclass segmentation
    """
    numClasses = y_true.shape[-1]
    dims = [i for i in range(K.ndim(y_true)-1)]
    w = 1 / (K.square(K.sum(y_true, dims) + K.backend.epsilon()))
    num = K.sum(w * K.sum( y_true * y_pred, dims))
    denom = K.sum(w * K.sum( y_true + y_pred, dims) ) 

    gd = 2*num/denom
    return gd
    
def generalizedDiceLoss(y_true, y_pred):
    return 1 - K.sum(generalizedDice(y_true, y_pred))

def mind_loss(stride, patch_size):
    """
    Based on MIND by Heinrich: https://www.sciencedirect.com/science/article/pii/S1361841512000643#b0060
    Implemented in tensorflow using notes by Courtney Guo
    """
    def distance(tensor):
        filt = tf.constant(1./patch_size**2, shape=(patch_size, patch_size, 1, 1))
        return tf.nn.conv2d(tensor, filt, strides=1, padding='SAME')
        
    def MIND(tensor):
        # Compute distance terms
        tensor_up    = tf.roll(tensor, -1*stride, axis=1) - tensor
        tensor_down  = tf.roll(tensor, stride, axis=1) - tensor
        tensor_left  = tf.roll(tensor, -1*stride, axis=2) - tensor
        tensor_right = tf.roll(tensor, stride, axis=2) - tensor
        
        tensor_up_sq    = distance(tensor_up)
        tensor_down_sq  = distance(tensor_down)
        tensor_left_sq  = distance(tensor_left)
        tensor_right_sq = distance(tensor_right)
        
        #Compute 6-neighborhood variance
        var = (1./6)*tf.reduce_sum(tensor_up_sq+tensor_down_sq+tensor_left_sq+tensor_right_sq, axis=[1,2], keepdims=True)
        
        scale = (1./tf.cast(tf.shape(tensor)[1]*tf.shape(tensor)[2], tf.float32))
        tensor_up_gauss = scale*tf.math.exp(tf.math.divide(-1.*tensor_up_sq, var))
        tensor_down_gauss = scale*tf.math.exp(tf.math.divide(-1.*tensor_down_sq, var))
        tensor_left_gauss = scale*tf.math.exp(tf.math.divide(-1.*tensor_left_sq, var))
        tensor_right_gauss = scale*tf.math.exp(tf.math.divide(-1.*tensor_right_sq, var))
        
        return tf.concat([tensor_up_gauss, tensor_down_gauss, tensor_left_gauss, tensor_right_gauss],axis=3)
    
    def loss(y_true, y_pred):
        y_true_MIND = MIND(y_true)
        y_pred_MIND = MIND(y_pred)
        return tf.reduce_mean(y_true_MIND - y_pred_MIND)
    return loss

def mutual_information(n_bins, eps=0.00001, bandwidth=0.000017):
    """
    Based on MI deep learning algorithm by Courtney Guo
    in thesis: Multi-Modal Image Registration with Unsupervised Deep Learning
    """
    def make_histogram_bins(n_bins, y_true, y_pred):
        max_val = tf.reduce_max(tf.concat([y_true, y_pred], 0))
        true_linspace = tf.linspace(0., max_val, n_bins)
        pred_linspace = tf.linspace(0., max_val, n_bins)
        
        # make batch-wise tiled
        b, w, h, c = tf.shape(y_true)
        rs = tf.constant([1, 1, n_bins])
        tile = tf.convert_to_tensor([b, w*h, 1])
        
        true_linspace_rs    = tf.reshape(true_linspace, rs)
        true_linspace_tiled = tf.tile(true_linspace_rs, tile)
        pred_linspace_rs    = tf.reshape(pred_linspace, rs)
        pred_linspace_tiled = tf.tile(pred_linspace_rs, tile)
        
        return true_linspace_tiled, pred_linspace_tiled
    
    def loss(y_true, y_pred):
        # Flatten and tile images batch-wise
        batch_flat = K.layers.Flatten()
        y_true_flat = batch_flat(y_true)
        y_pred_flat = batch_flat(y_pred)
        y_true_tile = tf.tile(tf.expand_dims(y_true_flat,-1), tf.constant([1, 1, n_bins]))
        y_pred_tile = tf.tile(tf.expand_dims(y_pred_flat,-1), tf.constant([1, 1, n_bins]))

        # Determine voxel distributions via Parzen Windowing
        bins_true, bins_pred, = make_histogram_bins(n_bins, y_true, y_pred)
        I_true = tf.math.exp(-1*tf.math.square(y_true_tile - bins_true)/bandwidth)
        I_pred = tf.math.exp(-1*tf.math.square(y_pred_tile - bins_pred)/bandwidth)

        # Normalize distributions
        denom_true = tf.math.reduce_sum(I_true, axis=-1)
        denom_pred = tf.math.reduce_sum(I_pred, axis=-1)
        denom_true_tile = tf.tile(tf.expand_dims(denom_true,-1), tf.constant([1,1,n_bins]))
        denom_pred_tile = tf.tile(tf.expand_dims(denom_pred,-1), tf.constant([1,1,n_bins]))
        I_true_norm = tf.divide(I_true, denom_true_tile)
        I_pred_norm = tf.divide(I_true, denom_pred_tile)
        tf.print(tf.shape(I_true_norm))

        # Determine probability distributions
        P_true = tf.math.reduce_mean(I_true_norm, axis=1)
        P_pred = tf.math.reduce_mean(I_pred_norm, axis=1)

        # Determine joint distribution and distribution products
        P_joint = tf.divide(tf.reduce_sum(tf.multiply(I_true, I_pred), axis=(1,2)),
                            tf.cast(tf.shape(y_true)[2]*tf.shape(y_true)[2], tf.float32))
        P_true_pred =tf.reduce_sum(tf.multiply(P_true,P_pred), axis=1)

        MI = tf.math.reduce_sum(tf.math.multiply(P_joint, tf.math.log(tf.math.divide(P_joint, P_true_pred)))+ eps)
        MI = tf.divide(MI, tf.reduce_sum(P_true))
        tf.print(P_true_pred)
        return MI
    return loss

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def sobelLoss(y_true, y_pred):
    """
    Calculates loss based on sobel filters, a.k.a. edge detection. This
    determines the difference between the edges detected in the true
    and predicted images. This is similar, but not equivalent, to taking the
    difference of edge lengths. In the binary case, this should be equivalent
    but I need to check. Used from: https://stackoverflow.com/questions/
    47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras
    since there appear to be buds in my custom funtion using K.conv_2d
    
    This version calculates the difference in the X and Y filters
    before computing the edge magnitude
    """
    sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])
    
    # Calculate sobel for each channel separately: replicate the S.F for
    # each channel
    _channels = K.reshape(K.ones_like(y_true[0,0,0,:]),(1,1,-1,1))
    filt = sobelFilter * _channels
    
    sobelTrue = K.depthwise_conv2d(y_true, filt)
    sobelPred = K.depthwise_conv2d(y_pred, filt)
    
    return K.mean(K.square(sobelTrue - sobelPred))
    
def squareSobelLoss(y_true, y_pred):
    """
    Calculates loss based on sobel filters, a.k.a. edge detection. This
    determines the difference between the edges detected in the true
    and predicted images. This is similar, but not equivalent, to taking the
    difference of edge lengths. In the binary case, this should be equivalent
    but I need to check. Used from: https://stackoverflow.com/questions/
    47346615/how-to-implement-custom-sobel-filter-based-loss-function-using-keras
    since K.conv2d implementations currently buggy
    """
    sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])
    
    _channels = K.reshape(K.ones_like(y_true[0,0,0,:], dtype='float32'),(1,1,-1,1))
    filt = sobelFilter * _channels

    # Calculate the squared sobel components for each tensor
    sobelTrue = K.square(tf.nn.depthwise_conv2d(y_true, filt, (1,1,1,1), padding='SAME'))
    sobelPred = K.square(tf.nn.depthwise_conv2d(y_pred, filt, (1,1,1,1), padding='SAME'))
    
    # reshape the matrices to merge the x2 and y2 channels
    newShape = K.shape(sobelTrue)
    newShape = K.concatenate([newShape[:-1],
                            newShape[-1:]//2,
                            K.variable([2], dtype='int32')])
    squareSobelTrue = K.sum(K.reshape(sobelTrue,newShape),axis=-1)
    squareSobelPred = K.sum(K.reshape(sobelPred,newShape),axis=-1)
    
    return K.mean(K.abs(squareSobelTrue - squareSobelPred))

def sobelAndCrossEntropy(y_true, y_pred):
    return K.losses.CategoricalCrossentropy(y_true, y_pred, sample_weight=0.8) + \
           0.2*squareSobelLoss(y_true, y_pred)

def ssim_loss(y_true, y_pred):
    """
    Structural similarity index
    SSIM is supported by tensorflow
    Currently uses default image parameters
    Assumes images are normalized on the range [0, 1]

    See en.wikipedia.org/wiki/Structural_similarity for more info
    """
    return 1.-tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

def static_weighted_bincrossentropy(y_true, y_pred):
    weights = (y_true * 1.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def SWBCE_DICE_loss(y_true, y_pred):
    return static_weighted_bincrossentropy(y_true, y_pred)+dice_coef_loss(y_true, y_pred)

def weighted_hausdorff_distance(w, h, alpha):
    """
    Hausdorff distance loss function implemented in tensorflow
    Source: https://arxiv.org/abs/1806.07564

    This version requires the loss function to be initialized in the main code block, then that loss function
    can be passed to the network

    Current implementation is memory intensive. May only be feasible as a way to calculate HD post-hoc and 
    not during gradient descent
    """
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            d_matrix = tf.cond(tf.equal(tf.shape(d_matrix)[1],0),
                              lambda: tf.zeros([w*h, 1]),
                              lambda: d_matrix)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)
            term_2 = tf.where(tf.math.is_nan(term_2), eps, term_2)

            return term_1 + term_2

        back_true = tf.squeeze(tf.slice(y_true, [0,0,0,1], [-1,-1,-1,1]))
        back_pred = tf.squeeze(tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1]))
        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (back_true, back_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss