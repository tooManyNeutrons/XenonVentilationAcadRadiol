# Input/output tools for deep learning
# Alex Matheson 9/13/2023
#
# Tools for importing, translating and outputing common imaging file formats to python formats
# for use in neural networks

import ants
import numpy as np
import nibabel as nib
import skimage.transform as TR

## Import

def get_nifti(filepath,img_size,feature_scale):
    '''
    This function is handy to grab nifti files using the ANTS package. A more 
    common method, Nibabel, clashes with older versions of tensorflow/keras/numpy, 
    so this makes do.
    
    This outputs an MxNxO-sized, 3D numpy array with the 3D data.
    '''
    filename = filepath;
    img = ants.image_read(filename);
    if img.shape != img_size:
        img = ants.resample_image(img, resample_params = (img_size), use_voxels=True, interp_type=4);
    if feature_scale == True:
        img = ants.iMath(img, 'Normalize');
    array = img.numpy();
    array = np.flip(array,0)  #Trying to get data to match ITK-Snap layout. 
    return array  

def get_NIBnifti(filepath,img_size,feature_scale):
    '''
    This function is handy to grab nifti files using the Nibabel package.
    
    This outputs an MxNxO-sized, 3D numpy array with the 3D data.
    '''
    filename = filepath;
    img = nib.load(filename)
    img = np.array(img.dataobj)
    if img.shape != img_size:
        img = TR.resize(img, (img_size), anti_aliasing=True, order=3,mode='constant');
    array = img#.numpy();
    #array = np.flip(array,0)  #Trying to get data to match ITK-Snap layout. 
    return array

###########################################################################################################

## Export


def export_png_predictions(result, generator, class_dict):
for i in range(result.shape[0]):
    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    img2show, label2show = generator.__load__(test_ids[fold][i], i)
    ax1.imshow(img2show, cmap='Greys_r')
    indx = np.argmax(result[i,:])
    ax1.text(10, 
                20, 
                "Disease:{}".format(list(class_dict.keys())[list(class_dict.values()).index(np.argmax(label2show))]),
                color='lime',
                fontsize=12)
    ax1.text(10, 
                170, 
                "Predicted:{} ({})".format(list(class_dict.keys())[list(class_dict.values()).index(indx)],
                                                    "{:.2f}".format(result[i,indx])), 
                color='lime',
                fontsize=12)
    ax1.text(10,
                185,
                test_ids[fold][i].parts[-1],
                color='lime',
                fontsize=12)
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot(1,2,2)
    ax2.pie(result[i,:], labels=class_dict.keys(), autopct='%1.1f%%', radius=0.75, textprops={'color': 'w'})
    fig.set_facecolor((0, 0, 0))

    plt.savefig('./Output/Fold1/{}.png'.format(test_ids[0][i].parts[-1]), bbox_inches='tight', pad_inches=0.0)
    plt.close()