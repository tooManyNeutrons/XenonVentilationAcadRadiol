# Utilities for neural networks
# Alex Matheson 9/13/2023
#
# Quality of life tools and common functions for neural networks

import random
import tensorflow as tf
import numpy as np
from itertools import zip_longest
import skimage.transform as TR
from sklearn.model_selection import KFold
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import affine_transform
from cv2 import  warpAffine, INTER_LINEAR, BORDER_REPLICATE
import cv2


def tf_repeat(tensor, repeats):
    """
    Args:
    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor

def train_split(allIDs, nfolds):
    """
    Takes file names from a directory list and randomly splits them into groups for cross-fold validation
    allIDs - list of subject names to sort
    nfolds - number of folds for cross-validation
    trainIDs - nfolds x N/nfolds list of file names for each fold
    testIDs - list of testing file names for each fold
    """

    #Generate cross validation

    # Note: shuffle maintains original order within each fold
    seed = 7
    folds = KFold(n_splits=nfolds, random_state=seed, shuffle=True)

    trainIDs = {}
    testIDs  = {}
    fold     = 0
    for tr, ts in folds.split(allIDs):
        trainIDs[fold] = [allIDs[i] for i in tr]
        testIDs[fold]  = [allIDs[i] for i in ts]
        fold = fold+1

    return trainIDs, testIDs

def train_split_classification(allIDs, nfolds):
    """
    Takes file names from a directory list and randomly splits them into groups for cross-fold validation
    allIDs - list of subject names to sort
    nfolds - number of folds for cross-validation
    trainIDs - nfolds x N/nfolds list of file names for each fold
    testIDs - list of testing file names for each fold
    """

    #Generate cross validation

    # Note: shuffle maintains original order within each fold
    seed = 7
    folds = KFold(n_splits=nfolds, random_state=seed, shuffle=True)

    trainIDs = {}
    testIDs  = {}
    fold     = 0
    for tr, ts in folds.split(allIDs):
        trainIDs[fold] = [allIDs[i] for i in tr]
        testIDs[fold]  = [allIDs[i] for i in ts]
        # temp_train = list(zip([allIDs[i] for i in tr], [labels[i] for i in tr]))
        # temp_test  = list(zip([allIDs[i] for i in ts], [labels[i] for i in ts]))
        # random.shuffle(temp_train)
        # random.shuffle(temp_test)
        # trainIDs[fold], train_labels[fold] = zip(*temp_train)
        # testIDs[fold], test_labels[fold] = zip(*temp_test)
        # trainIDs[fold], train_labels[fold] = list(trainIDs[fold]), list(train_labels[fold])
        # testIDs[fold], test_labels[fold] = list(testIDs[fold]), list(test_labels[fold])
        fold = fold+1


    return trainIDs, testIDs

def three_group_k_fold(items, n_folds):
    n_groups = [[]]*n_folds

    avail = items
    for i in range(n_folds):
        n = len(items)//n_folds
        if len(avail)%n_folds != 0:
            n+=1
        n_groups[i] = avail[:n]
        avail = avail[n:]

    train_group = [[]]*n_folds
    valid_group = [[]]*n_folds
    test_group  = [[]]*n_folds
    for i in range(n_folds):
        test_group[i] = n_groups[-1]
        valid_group[i] = n_groups[0][:len(n_groups[0])//2]

        _tmp = n_groups[0][len(n_groups[0])//2:]
        for j in range(1,n_folds-1):
            _tmp = _tmp + n_groups[j]

        train_group[i] = _tmp

        # Shift all groups by 1 position
        _reorder = n_groups[1:]
        _reorder.append(n_groups[0])
        n_groups = _reorder

    return train_group, valid_group, test_group

# Note: The below augmentation tools are for 3D from the old IkeML code. Keeping for legacy purposes for now

def rot_x(angle,ptx,pty):  #useful eqn for RotationSAG
    return np.cos(angle)*ptx + np.sin(angle)*pty

def rot_y(angle,ptx,pty):  #useful eqn for RotationSAG
    return -np.sin(angle)*ptx + np.cos(angle)*pty

def RotationSAG(X,thetaval):
    '''
    This function rotates the 3D image around the z-axis (normal to saggital 
    slice) only.  Reasoning: kids lay on their backs, but may be slightly tilted
    about this axis.
    
    Beware hard-coded values in this version.
    '''
    x = X[:,0,0].size
    y = X[0,:,0].size
    numslice = X[0,0,:].size;
    mxx = X.max()
    mnx = X.min()
    X = (X-mnx)/mxx
    output = np.empty([x,y,numslice],dtype=np.float32)
    for nn in range(numslice):
        output[nn,:,:] = TR.rotate(np.squeeze(X[nn,:,:]),thetaval)
    return output

def RotationAXI(X,thetaval):
    '''
    This function rotates the 3D image around the z-axis (normal to saggital 
    slice) only.  Reasoning: kids lay on their backs, but may be slightly tilted
    about this axis.
    
    Beware hard-coded values in this version.
    '''
    x = X[:,0,0].size
    y = X[0,:,0].size
    numslice = X[0,0,:].size;
    mxx = X.max()
    mnx = X.min()
    X = (X-mnx)/mxx
    output = np.empty([x,y,numslice],dtype=np.float32)
    for nn in range(numslice):
        output[:,nn,:] = TR.rotate(np.squeeze(X[:,nn,:]),thetaval)
    return output


def RotationCOR(X,thetaval):
    '''
    This function rotates the 3D image around the z-axis (normal to coronal 
    slice) only.  Reasoning: kids lay on their backs, but may be slightly tilted
    about this axis.
    
    Beware hard-coded values in this version.
    '''
    x = X[:,0,0].size
    y = X[0,:,0].size
    numslice = X[0,0,:].size;
    mxx = X.max()
    mnx = X.min()
    X = (X-mnx)/mxx
    output = np.empty([x,y,numslice],dtype=np.float32)
    for nn in range(numslice):
        output[:,:,nn] = TR.rotate(np.squeeze(X[:,:,nn]),thetaval)
    return output

def Zoom(X,zoomvals):
    zoomx = zoomvals[0]
    zoomy = zoomvals[1] 
    zoomz = zoomvals[2]
    xsc = 1+zoomx;
    ysc = 1+zoomy;
    zsc = 1+zoomz;
    cx = 112//2;
    cy = 112//2;
    cz = 112//2;
    
    #x and y direction 
    xytrans = TR.AffineTransform(matrix = np.array([[xsc, 0, cx-cx*xsc],
                                                    [0, ysc, cy-cy*ysc],
                                                    [0,   0,         1]]))
    #z-transform looks like an x tranform.  Axes are gonna be swapped around first, it's FINE.
    ztrans  = TR.AffineTransform(matrix = np.array([[zsc, 0, cz-cz*zsc],
                                                    [0,   1,         0],
                                                    [0,   0,         1]]))
    X = TR.warp(X,xytrans,order=1,preserve_range=True)
    X = np.swapaxes(X,1,2)
    X = TR.warp(X,ztrans, order=1,preserve_range=True)
    X = np.swapaxes(X,1,2)
    return X
    
def Shear(X,shearvals):
    sxyx = shearvals[0]
    sxyy = shearvals[1]
    sxzx = shearvals[2]
    sxzz = shearvals[3]
    syzy = shearvals[4]
    syzz = shearvals[5]
    cx = 112//2;
    cy = 112//2;
    cz = 112//2;
    
    xyshear =TR.AffineTransform(matrix = np.array([[1, sxyx, -sxyx*cy],
                                                   [sxyy, 1, -sxyy*cx],
                                                   [0,    0,        1]]))
    xzshear =TR.AffineTransform(matrix = np.array([[1, sxzx, -sxzx*cz],
                                                   [sxzz, 1, -sxzz*cx],
                                                   [0,    0,        1]]))
    yzshear =TR.AffineTransform(matrix = np.array([[1, syzy, -syzy*cz],
                                                   [syzz, 1, -syzz*cy],
                                                   [0,    0,        1]]))
    
    Xo = TR.warp(X,xyshear,order=1,preserve_range=True)
    Xo = np.swapaxes(X,1,2)
    Xo = TR.warp(X,xzshear,order=1,preserve_range=True)
    Xo = np.swapaxes(X,1,2)
    Xo = np.swapaxes(X,0,2)
    Xo = TR.warp(X,yzshear,order=1,preserve_range=True)
    Xo = np.swapaxes(X,0,2)
    return X

def Translate(X,tr):  #Not included yet.
    X = X
    return X

def data_augmentation(X,TH_COR,TH_AXI,TH_SAG,Z,SH,TR):
    X = Zoom(X,Z)
    X = Shear(X,SH)
    X = RotationCOR(X,TH_COR)
    X = RotationSAG(X,TH_SAG)
    X = Translate(X,TR)
    return X

# New data augmentation framework below. A more simple version is possible in Keras, but issues
# in version 2.10 cause a massive performance drop. Augmentation layers may be fixed in future 
# but for now the below code can be used instead, which is compatible with numpy, not with the
# later tensors.

class Augmentation:
    """
    Class to perform data augmentation before data is passed to a neural network. Based on numpy
    and scikit implementations of transforms. Currently supports 2D affine transformations and
    elasic transformations, either separately or combined
    Arguments:
    dx    > x-translation
    dy    > y-translation
    sc    > scale
    rot   > rotation
    flip  > flip about the x-, y- or both axes
    noise > variance of Gaussian noise to add
    """
    def __init__(self, dx=0, dy=0, sc=0, rot=0, flip_x=False, flip_y=True, noise=0.01,
                 elastic_alpha=1, elastic_sigma=0):
        self.dx            = dx
        self.dy            = dy
        self.sc            = sc
        self.rot           = rot
        self.fx            = flip_x
        self.fy            = flip_y
        self.noise         = noise
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def translate(self):
        return np.array([
            [1, 0, self.dx*np.random.uniform(-1,1)],
            [0, 1, self.dy*np.random.uniform(-1,1)],
            [0, 0, 1                              ]
        ])
    
    def rotate(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [np.cos(self.rot*u), -np.sin(self.rot*u), 0],
            [np.sin(self.rot*u),  np.cos(self.rot*u), 0],
            [0,                   0,                  1]
        ])

    def scale(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [self.sc*u+1, 0,           0],
            [0,           self.sc*u+1, 0],
            [0,           0,           1]
        ])
    
    def flip_y(self):
        return np.array([
            [1,  0,                          0],
            [0, -1+2*np.random.randint(0,2), 0],
            [0,  0,                          1]
        ])
    
    def flip_x(self):
        return np.array([
            [-1+2*np.random.randint(0,2), 0, 0],
            [ 0,                          1, 0],
            [ 0,                          0, 1]
        ])
    
    def elastic(self, img):
        """
        Deformable warping image on a smoothed grid. Code implemented based on implementation
        of Simard technique from www.kaggle.com/code/bguberfain/elastic-transform-for-data-
        augmentation/notebook

        img          > image to warp
        alpha        > "height" or intensity of warp
        sigma        > "width" or spacing of warp
        alpha_affine > initial "noise" that initializes the warping

        returns an elastically warped image

        """

        shape = img.shape
        #shape_size = shape[:2]

        # center_square = np.float32(shape_size) // 2
        # square_size = min(shape_size) // 3
        # pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size],
        #                    center_square-square_size])
        # pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape)
        # M = cv2.getAffineTransform(pts1, pts2)
        # image = cv2.warpAffine(image, M, shape_size(::-1), borderMode = cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha

        x,y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        return map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

    
    def pos_size(self, img):
        w, h = img.shape
        return np.array([
            [1, 0, w/2.],
            [0, 1, h/2.],
            [0, 0, 1   ]])
    
    def neg_size(self, img):
        w, h = img.shape
        return np.array([
            [1, 0, -w/2.],
            [0, 1, -h/2.],
            [0, 0, 1    ]])
    
    def normalize(self, dat):
        # Normalize data
        dat = dat - np.amin(dat) # Set min to 0
        if np.amax(dat) > 0.01:
            dat = dat/np.amax(dat)

        return dat
    
    def gaussian_noise(self, dat, stdev):
        gnoise = np.random.normal(0, stdev, size=dat.shape)
        dat = dat + gnoise
        return dat

    def apply(self, img):  
        flip_mat = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        
        if self.fx:
            flip_mat = self.flip_x() @ flip_mat
        if self.fy:
            flip_mat = self.flip_y() @ flip_mat
              
        tform = self.pos_size(img) @ self.scale() @ self.rotate() @ self.translate()  @ flip_mat @ self.neg_size(img)
        img =  warpAffine(img, 
                          tform[:2,:], 
                          (img.shape[0], img.shape[1]), 
                          flags=INTER_LINEAR, 
                          borderMode=BORDER_REPLICATE, 
                          borderValue=None)
        #img = self.elastic(img)
        img = self.normalize(img)
        img = self.gaussian_noise(img, self.noise) # This is no longer normalized on 0-1 but is still normalized
        img = self.normalize(img)

        return img



def Normalizer(X,Bigmax,Bigmin):

    mxx = X.max()
    #mnx = X.min()
    if mxx >0:  #Keep here in case edge is zeros only. (If kept, "zero" matrix would be offset by -Bigmin/Bigmax)
        X = (X-Bigmin)/Bigmax
    return X

class Augmentation3D:
    """
    Class to perform data augmentation before data is passed to a neural network. Based on numpy
    and scikit implementations of transforms. Currently supports 3D affine transformations
    Arguments:
    dx    > x-translation
    dy    > y-translation
    sc    > scale
    rot   > rotation
    flip  > flip about the x-, y- or both axes
    noise > variance of Gaussian noise to add
    """
    def __init__(self, dx=0, dy=0, dz=0, sc=0, theta=0, phi=0, psi=0, flip_x=False, flip_y=True, noise=0,
                 elastic_alpha=1, elastic_sigma=1, elastic_z=True):
        self.dx            = dx
        self.dy            = dy
        self.dz            = dz
        self.sc            = sc
        self.theta         = theta
        self.phi           = phi
        self.psi           = psi
        self.fx            = flip_x
        self.fy            = flip_y
        self.noise         = noise
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        if elastic_z == True:
            self.elastic_z = elastic_alpha
        else:
            self.elastic_z = 0

    def translate(self):
        return np.array([
            [1, 0, 0, self.dx*np.random.uniform(-1,1)],
            [0, 1, 0, self.dy*np.random.uniform(-1,1)],
            [0, 0, 1, self.dz*np.random.uniform(-1,1)],
            [0, 0, 0, 1                              ]
        ])
    
    def r_theta(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [1,  0,                    0,                    0],
            [0,  np.cos(self.theta*u), np.sin(self.theta*u), 0],
            [0, -np.sin(self.theta*u), np.cos(self.theta*u), 0],
            [0,  0,                    0,                    1]
        ])
    
    def r_phi(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [np.cos(self.phi*u), 0, -np.sin(self.phi*u), 0],
            [0,                  1,                  0,  0],
            [np.sin(self.phi*u), 0, np.cos(self.phi*u),  0],
            [0,                  0,                  0,  1]
        ])
    
    def r_psi(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [np.cos(self.psi*u), -np.sin(self.psi*u), 0, 0],
            [np.sin(self.psi*u),  np.cos(self.psi*u), 0, 0],
            [0,                   0,                  1, 0],
            [0,                   0,                  0, 1]
        ])

    def scale(self):
        u = np.random.uniform(-1,1)
        return np.array([
            [self.sc*u+1, 0,           0,           0],
            [0,           self.sc*u+1, 0,           0],
            [0,           0,           self.sc*u+1, 0],
            [0,           0,           0,           1]
        ])
    
    def flip_y(self):
        return np.array([
            [1,  0,                          0, 0],
            [0, -1+2*np.random.randint(0,2), 0, 0],
            [0,  0,                          1, 0],
            [0,  0,                          0, 1]
        ])
    
    def flip_x(self):
        return np.array([
            [-1+2*np.random.randint(0,2), 0, 0, 0],        #shape_size = shape[:2]
            [ 0,                          1, 0, 0],
            [ 0,                          0, 1, 0],
            [ 0,                          0, 0, 1]
        ])
    
    def elastic(self, img):
        """
        Deformable warping image on a smoothed grid. Code implemented based on implementation
        of Simard technique from www.kaggle.com/code/bguberfain/elastic-transform-for-data-
        augmentation/notebook

        img          > image to warp
        alpha        > "height" or intensity of warp
        sigma        > "width" or spacing of warp
        alpha_affine > initial "noise" that initializes the warping

        returns an elastically warped image

        """

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_z

        x,y,z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1,1))

        return map_coordinates(img, indices, order=1, mode='constant').reshape(shape)

    
    def pos_size(self, img):
        w, h, t = img.shape
        return np.array([
            [1, 0, 0, w/2.],
            [0, 1, 0, h/2.],
            [0, 0, 1, t/2.],
            [0, 0, 0, 1]])
    
    def neg_size(self, img):
        w, h, t = img.shape
        return np.array([
            [1, 0, 0, -w/2.],
            [0, 1, 0, -h/2.],
            [0, 0, 1, -t/2.],
            [0, 0, 0, 1]])
    
    def normalize(self, dat):
        # Normalize data
        if np.amax(dat) > 0.01:
            dat = dat/np.amax(dat)

        return dat
    
    def gaussian_noise(self, dat, stdev):
        gnoise = np.random.normal(0, stdev, size=dat.shape)
        dat = dat + gnoise
        return dat

    def apply(self, img):  
        flip_mat = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        
        if self.fx:
            flip_mat = self.flip_x() @ flip_mat
        if self.fy:
            flip_mat = self.flip_y() @ flip_mat
              
        tform = self.pos_size(img) @ self.scale() @ self.r_theta() @ self.r_phi() @ self.r_psi() @ self.translate() @ flip_mat @ self.neg_size(img)
        img =  affine_transform(img, tform, mode='reflect')
        #img = self.elastic(img) # Elastic before noise in case any null regions from outside of noisy volume warp i

        img = self.normalize(img)
        img = self.gaussian_noise(img, self.noise)

        return img



def Normalizer(X,Bigmax,Bigmin):

    mxx = X.max()
    #mnx = X.min()
    if mxx >0:  #Keep here in case edge is zeros only. (If kept, "zero" matrix would be offset by -Bigmin/Bigmax)
        X = (X-Bigmin)/Bigmax
    return X

class AugmentationNonRandom:
    """
    Augmentation for handling multi-channel data so that each channel is transformed 
    identically. Seed should be uniform on [-1, 1] for the first 4 numbers, random 0
    or 1 for the last two seeds.
    Arguments:
    dx    > x-translation
    dy    > y-translation
    sc    > scale
    rot   > rotation
    flip  > flip about the x-, y- or both axes
    noise > variance of Gaussian noise to add
    seed  > six random numbers on [-1, 1] corresponding to dx, dy, rot, scale, flipx, flipy
    """
    def __init__(self, dx=0, dy=0, sc=0, rot=0, flip_x=False, flip_y=True, noise=0.01,
                 elastic_alpha=1, elastic_sigma=0, seed=[0, 0, 0, 0, 0, 0], noiseless_channels=[False]):
        self.dx                 = dx
        self.dy                 = dy
        self.sc                 = sc
        self.rot                = rot
        self.fx                 = flip_x
        self.fy                 = flip_y
        self.noise              = noise
        self.elastic_alpha      = elastic_alpha
        self.elastic_sigma      = elastic_sigma
        self.seed               = seed
        self.noiseless_channels = noiseless_channels

    def translate(self):
        return np.array([
            [1, 0, self.dx*self.seed[0]],
            [0, 1, self.dy*self.seed[1]],
            [0, 0, 1                   ]
        ])
    
    def rotate(self):
        u = self.seed[2]
        return np.array([
            [np.cos(self.rot*u), -np.sin(self.rot*u), 0],
            [np.sin(self.rot*u),  np.cos(self.rot*u), 0],
            [0,                   0,                  1]
        ])

    def scale(self):
        u = self.seed[3]
        return np.array([
            [self.sc*u+1, 0,           0],
            [0,           self.sc*u+1, 0],
            [0,           0,           1]
        ])
    
    def flip_y(self):
        return np.array([
            [1,  0,                0],
            [0, -1+2*self.seed[4], 0],
            [0,  0,                1]
        ])
    
    def flip_x(self):
        return np.array([
            [-1+2*self.seed[5], 0, 0],
            [ 0,                1, 0],
            [ 0,                0, 1]
        ])
    
    def elastic(self, img):
        """
        Deformable warping image on a smoothed grid. Code implemented based on implementation
        of Simard technique from www.kaggle.com/code/bguberfain/elastic-transform-for-data-
        augmentation/notebook

        img          > image to warp
        alpha        > "height" or intensity of warp
        sigma        > "width" or spacing of warp
        alpha_affine > initial "noise" that initializes the warping

        returns an elastically warped image

        """

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 -1), self.elastic_sigma) * self.elastic_alpha

        x,y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        return map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

    
    def pos_size(self, img):
        w, h = img.shape[:2]
        return np.array([
            [1, 0, w/2.],
            [0, 1, h/2.],
            [0, 0, 1   ]])
    
    def neg_size(self, img):
        w, h = img.shape[:2]
        return np.array([
            [1, 0, -w/2.],
            [0, 1, -h/2.],
            [0, 0, 1    ]])
    
    def normalize(self, dat):
        # Normalize data
        dat = dat - np.amin(dat) # Set min to 0
        if np.amax(dat) > 0.01:
            dat = dat/np.amax(dat)

        return dat
    
    def gaussian_noise(self, dat, stdev):
        gnoise = np.random.normal(0, stdev, size=dat.shape)
        dat = dat + gnoise
        return dat

    def apply(self, img):  
        flip_mat = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        
        if self.fx:
            flip_mat = self.flip_x() @ flip_mat
        if self.fy:
            flip_mat = self.flip_y() @ flip_mat
              
        tform = self.pos_size(img) @ self.scale() @ self.rotate() @ self.translate()  @ flip_mat @ self.neg_size(img)
        img =  warpAffine(img, 
                          tform[:2,:], 
                          (img.shape[0], img.shape[1]), 
                          flags=INTER_LINEAR, 
                          borderMode=BORDER_REPLICATE, 
                          borderValue=None)
        #img = self.normalize(img) # turned this off since the channels need to be normalized differently
        for i in range(img.shape[-1]):
            if not self.noiseless_channels[i]:
                img[:,:,i] = self.gaussian_noise(img[:,:,i], self.noise) # This is no longer normalized on 0-1 but is still normalized
                img[:,:,i] = self.normalize(img[:,:,i])

        return img