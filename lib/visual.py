# Visualization Tools
# Alex Matheson 9/13/2023
# 
# Functions for visualizing batched data used in neural networks and for comparing ground-truth images
# alonside network predictions.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def noTicks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def trainShow(im, lab):
    """
    shows paired proton images and associated label maps
    """
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im, cmap="Greys")
    noTicks(ax)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(lab, cmap="Greys")
    noTicks(ax)
    
def testShow(im, lab, test):
    """
    shows the test results of a segmentation branch test against
    the training proton and "true" labelmap
    """
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(im, cmap="Greys")
    noTicks(ax)
    ax.fig.add_subplot(1, 3, 2)
    ax.imshow(lab, cmap="Greys")
    noTicks(ax)
    ax.fig.add_subplot(1, 3, 3)
    ax.imshow(test, cmap="Greys")
    noTicks(ax)

def testOverlap(ax, lab, test, cbar=False):
    """
    Shows overlap between two different binary images. 
    Input: 
    ax   > the axis to plot image on. Provides subplot support through 
           overlapBatchShow
    lab  > ground-truth binary label map
    test > predicted binary label map
    cbar > applies a legend for the color-scheme
    """
    _w, _h  = lab.shape
    overHot = np.zeros((_w, _h, 4))

    overHot[:,:,0] = np.logical_and( np.logical_not(lab), np.logical_not(test) )
    overHot[:,:,1] = np.logical_and(test, np.logical_not(lab) )
    overHot[:,:,2] = np.logical_and(np.logical_not(test), lab )
    overHot[:,:,3] = np.logical_and(test, lab)
    
    overlap = np.argmax(overHot, axis=-1)
    
    #colormap for different 
    colours = [(0,0,0), (0.25,1,0), (1,0,0.75), (1,1,1)]
    cm = LinearSegmentedColormap.from_list('overlap', colours, N=4)
    ax.imshow(overlap, cmap=cm, vmin=0, vmax=3)
    if cbar==True:
        cb = plt.colorbar()
        cb.ax.get_yaxis().set_ticks(['True Negative', 'False Negative', 
                                     'False Positive', 'True Positive'])
    noTicks(ax)

def overlapBatchShow(lab, test, cbar=False):
    numBatches = lab.shape[0]
    numCols    = 4
    numRows    = int(np.ceil(numBatches/numCols))
    
    fig=plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(numRows): #iterate over batches
        for j in range(numCols):
            ax = fig.add_subplot(numRows, 
                                 numCols, 
                                 i*numCols + j + 1)
            labSlice  = np.argmax(lab[i*numCols + j, :, :, :],
                                  axis=-1)>0
            testSlice = np.argmax(test[i*numCols + j, :, :, :],
                                  axis=-1)>0
            testOverlap(ax, labSlice, testSlice, cbar)
            noTicks(ax)
    return fig
    
def protonBatchShow(tensor):
    """
    Displays all images in a single modality batch and dynamically
    allocates subplots.
    Inputs:
    tensor > tensor of images. Should be in the format (image#, [dims])
    """
    
    fig=plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    numEntries = tensor.shape[0]
    numCols = 4
    numRows = int(np.ceil(numEntries/numCols))
    for i in range(numRows):
        for j in range(numCols):
            ax = fig.add_subplot(numRows, numCols, i*numCols + j + 1)
            ax.imshow(tensor[i*numCols + j, :, :, 0], cmap='Greys_r')
            noTicks(ax)
    return fig
            
def labelBatchShow(tensor):
    """
    Displays a label map for ground-truth data when processed as a 
    batch. Labels are generated in tensors of the form [batchsize, imageDim0,
    imagedim1, {imagedim2}, numClasses]. This code merges the final dimension 
    to show a labelmap as opposed to the individual binary classes.
    Inpurs:
    tensor > tensor of binary classes
    """
    
    fig=plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    numEntries = tensor.shape[0]
    numCols = 4
    numRows = int(np.ceil(numEntries/numCols))
    
    _img = np.argmax(tensor, axis=-1)
    for i in range(numRows):
        for j in range(numCols):
            ax = fig.add_subplot(numRows, numCols, i*numCols + j + 1)
            ax.imshow(_img[i*numCols + j, :, :], vmin=0, vmax=2)
            noTicks(ax)
    return fig

def classContour(img, seg):
    """
    Displays contours of network segmentations on top of source image being segmented
    Inputs:
    img > image being segmented
    seg > multi-class segmentation results. Expected layers: background (0), right (1), left (2)
    Outputs:
    fig > the generated figure for saving / further manipulation
    """
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(img, cmap='gray')
    ax.contour(seg[:,:,1], colors='cyan')
    ax.contour(seg[:,:,2], colors='red')
    noTicks(ax)
    return fig

def classification_visualize(imgs, results, labels, files, dict, z_offset=0):
    """
    Shows batches of images, their ground-truth disease status, the predicted value from the network
    and the probability of that result overlaid on the image
    """
    fig=plt.figure(figsize=(15,9))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    numEntries = imgs.shape[0]
    numCols = 8
    numRows = int(np.ceil(numEntries/numCols))
    for i in range(numRows):
        for j in range(numCols):
            ax = fig.add_subplot(numRows, numCols, i*numCols + j + 1)
            ax.imshow(imgs[i*numCols + j, :, :, z_offset], cmap='Greys_r')
            # Get the ground truth disease classification and use dictionary to convert back to text
            indx = np.argmax(results[i*numCols+j,:]) # Most likely classification
            ax.text(10, 
                    20, 
                    "Disease:{}".format(list(dict.keys())[list(dict.values()).index(labels[i*numCols+j])]),
                    color='lime',
                    fontsize=7)
            ax.text(10, 
                    170, 
                    "Predicted:{} ({})".format(list(dict.keys())[list(dict.values()).index(indx)],
                                                       "{:.2f}".format(results[i*numCols+j,indx])), 
                    color='lime',
                    fontsize=7)
            ax.text(10,
                    185,
                    files[i*numCols + j].parts[-1],
                    color='lime',
                    fontsize=7)
            noTicks(ax)
    return fig

def classification_batch_show(imgs, labels, files, dict):
    """
    Shows batches of images, their ground-truth disease status, the predicted value from the network
    and the probability of that result overlaid on the image
    """
    fig=plt.figure(figsize=(15,9))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    numEntries = imgs.shape[0]
    numCols = 8
    numRows = int(np.ceil(numEntries/numCols))
    for i in range(numRows):
        for j in range(numCols):
            ax = fig.add_subplot(numRows, numCols, i*numCols + j + 1)
            ax.imshow(imgs[i*numCols + j, :, :, 0], cmap='Greys_r')
            # Get the ground truth disease classification and use dictionary to convert back to text
            ax.text(10, 
                    20, 
                    "Disease:{}".format(list(dict.keys())[list(dict.values()).index(labels[i*numCols+j])]),
                    color='lime',
                    fontsize=7)
            ax.text(10, 
                    170, 
                    "Label Number:{}".format(labels[i*numCols+j]), 
                    color='lime',
                    fontsize=7)
            ax.text(10,
                    185,
                    files[i*numCols + j].parts[-1],
                    color='lime',
                    fontsize=7)
            noTicks(ax)
    return fig

def export_classification_images(out_array, test_data, key, diseases, savedir):
    """
    Function to take output from a classification network, plots the image, 
    shows the classification, and shows a pie plot of the probability of 
    each class
    """
    def limit_pct(pct):
        return ('%.1f' % pct) if pct>10 else ''

    def set_pie_labels(vals, labels):
        label_list = []
        for i in range(np.size(vals)):
            if vals[i] > .02:
                label_list.append(list(labels)[i])
            else:
                label_list.append('')

        return label_list

    print(out_array.shape, len(key))
    for i in range(out_array.shape[0]):
        fig = plt.figure()
        ax1 = plt.subplot(1,2,1)
        img2show, label2show = test_data.__load__(key[i], i)
        ax1.imshow(img2show, cmap='Greys_r')
        indx = np.argmax(out_array[i,:])
        ax1.text(10, 
                    20, 
                    "Disease:{}".format(list(diseases.keys())[list(diseases.values()).index(np.argmax(label2show))]),
                    color='lime',
                    fontsize=12)
        ax1.text(10, 
                    170, 
                    "Predicted:{} ({})".format(list(diseases.keys())[list(diseases.values()).index(indx)],
                                                        "{:.2f}".format(out_array[i,indx])), 
                    color='lime',
                    fontsize=12)
        ax1.text(10,
                    185,
                    key[i].parts[-1],
                    color='lime',
                    fontsize=12)
        plt.xticks([])
        plt.yticks([])

        ax2 = plt.subplot(1,2,2)
        ax2.pie(out_array[i,:], labels=set_pie_labels(out_array[i,:], diseases.keys()),
                autopct=limit_pct, radius=0.75, textprops={'color': 'w'})
        fig.set_facecolor((0, 0, 0))

        plt.savefig(savedir + '/Images/{}.png'.format(key[i].parts[-1]), bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close()