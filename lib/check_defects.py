# Checks a slice to see if there are ventilation defects before passing labelled data to a neural network
#
#
import os
import pathlib
import ants
import numpy as np
from scipy.ndimage import binary_erosion

def check_defects(path_list, threshold=2):
    """
    returns a binary list representing if the associated ID has defects

    path_list = list of paths to individual ventilation slices
    threshold = minimum VDP to be classified as diseased
    """

    defect_mask_list = np.ones(len(path_list))

    for i, path in enumerate(path_list):
        defect_path = pathlib.PosixPath(path.parents[1].joinpath('Defect', 
                                                                 path.parts[-1][:-11]+"Defect_"+
                                                                 path.parts[-1][-6:]))
        mask_path   = pathlib.PosixPath(path.parents[1].joinpath('Mask', 
                                                                 path.parts[-1][:-11]+"Mask_"+
                                                                 path.parts[-1][-6:]))

        #skip iteration if files missing
        if (not os.path.isfile(defect_path) or not os.path.isfile(mask_path)):
            #print("File Missing", i, defect_path.parts[-1], defect_mask_list[i])
            continue


        defect_img = ants.image_read(str(defect_path)).numpy()
        mask_img   = ants.image_read(str(mask_path)).numpy()

        #some issues with old files not converting/scaling correctly
        if np.shape(defect_img) != np.shape(mask_img):
            #print("shape mismatch", i, defect_path.parts[-1])
            continue

        thoracic_volume = np.sum(mask_img==1)
        defect_volume   = np.sum((defect_img<4) & (defect_img>0))
        if thoracic_volume>50:
            vdp = 100*defect_volume/thoracic_volume
        else:
            continue

        if vdp<threshold:
            defect_mask_list[i]=0

    return list(defect_mask_list)

def check_snr(path_list, threshold=8):
    """
    Returns a binar list representing if the associated ID has SNR greater than the threshold

    path_list = list of paths to individual ventilation slices
    threshold = maximum acceptable SNR
    """

    mask_list = np.ones(len(path_list))

    for i, path in enumerate(path_list):
        img_path = pathlib.PosixPath(path.parents[1].joinpath('Vent', 
                                                                 path.parts[-1][:-11]+"Vent_"+
                                                                 path.parts[-1][-6:]))
        mask_path   = pathlib.PosixPath(path.parents[1].joinpath('Mask', 
                                                                 path.parts[-1][:-11]+"Mask_"+
                                                                 path.parts[-1][-6:]))
        #skip iteration if files missing
        if (not os.path.isfile(img_path) or not os.path.isfile(mask_path)):
            #print("File Missing", i, path.parts[-1], mask_list[i])
            continue

        vent_img = ants.image_read(str(img_path)).numpy()
        mask_img   = ants.image_read(str(mask_path)).numpy()

        # Erode the ROIs so that blurring, etc doesn't affect values
        st_element = np.ones((5,5), dtype=bool)
        vent_roi = binary_erosion(mask_img==1, st_element)
        noise_roi = binary_erosion(mask_img==0, st_element)

        # Mask doesn't include the trachea... cut out the middle region of the noise
        noise_roi[np.int32(noise_roi.shape[0]*0.3):np.int32(noise_roi.shape[0]*0.7),
                  np.int32(noise_roi.shape[1]*0.3):np.int32(noise_roi.shape[1]*0.7)] = 0

        #some issues with old files not converting/scaling correctly
        if np.shape(vent_img) != np.shape(mask_img):
            #print("shape mismatch", i, path.parts[-1])
            continue

        signal = np.mean(vent_img[vent_roi])
        noise  = np.std(vent_img[noise_roi])
        snr    = signal/noise

        if snr<threshold:
            mask_list[i]=0

    return mask_list