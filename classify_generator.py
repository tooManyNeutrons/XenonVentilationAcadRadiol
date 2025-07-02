# Generator for hyperpolarized gas classification networks
# Alex Matheson 9/14/2023

import os.path
import ants
import keras as K
import numpy as np
import lib.utils as utils
import lib.model as model
from skimage.transform import resize
from scipy.ndimage import affine_transform, binary_dilation
import pathlib 

class ClassifyGenerator(K.utils.Sequence):
    def __init__(self, ids, labels, path, nclasses, batchSize=32, imageSize=192, training=True, 
                 x_translate=0, y_translate=0, scale=0, rotate=0, flip_x=False, flip_y=False, noise=0.,
                 elastic_alpha=1, elastic_sigma=20, invert=False):
        """
        Input variables:
        path      > the path that data subfolders are located in
        batchSize > how many images are processed before the model updates
                    batch is filled by random slices
        imageSize > the size of a single dimension. Assumes a square slice
        training  > sets the data to training mode vs testing mode. In training mode, ground-truth labels are
                    provided to the network, which are not needed for testing
        """
        super().__init__()
        self.ids  = ids
        self.labels = labels
        self.path = path
        self.nclasses = nclasses
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.on_epoch_end()
        self.training = training

        self.x_translate   = x_translate
        self.y_translate   = y_translate
        self.scale         = scale
        self.rotate        = rotate
        self.flip_x        = flip_x
        self.flip_y        = flip_y
        self.noise         = noise
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.invert        = invert

        self.aug = utils.Augmentation(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma)
        
    def update_transform(self):
        self.aug = utils.Augmentation(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma)
        
    def __load__(self, id, num): ### Will probably need to update these paths to ensure they match
        """
        loads a single set of matched data for the provided file name. Each file should be on
        a separate path. 
        """
        # Load files
        img = ants.image_read(id.as_posix())
        label = self.labels[num]
        
        # Convert to numpy arrays
        dat = img.numpy()

        # Dec 18 2024: test to see if inverting the images produces the same result
        if self.invert:
            mask_id = pathlib.Path(str(id).replace('Vent', 'Mask'))
            mask   = ants.image_read(mask_id.as_posix())
            mask   = mask.numpy()
            mask   = resize(mask, dat.shape)

            dat = dat - np.min(dat.flatten())
            dat = np.max(dat.flatten()) - dat
            dat = np.multiply(dat, mask)

        # else:
        #     mask_id = pathlib.Path(str(id).replace('Vent', 'Mask'))
        #     mask   = ants.image_read(mask_id.as_posix())
        #     mask   = mask.numpy()
        #     mask   = resize(mask, dat.shape)
        #     strel  = np.ones([5,5])
        #     mask   = binary_dilation(mask, strel)

        #     dat = np.multiply(dat, mask)

        # Resize

        dat = resize(dat, (self.imageSize, self.imageSize), order=0)

        return dat, label
    
        
    def __getitem__(self, indx):
        """
        Retrieves a single batch of data to feed into the network. Data and
        slices are randomly generated. Intended to retrieve taining data.
        Inputs:
        
        Outputs:
        proton > a tensor of random proton slices
        xenon  > a tensor of xenon slices matching proton slices
        label  > a tensor of combined, registered 
        trans  > a tensor of transformation parameters matching the proton and
                 xenon images
        """
        fileBatch = self.ids[indx*self.batchSize : (indx+1)*self.batchSize]
         
        #initialize
        img = np.empty([self.batchSize, self.imageSize, self.imageSize, 1])
        label = np.empty([self.batchSize, self.nclasses])
        
        if self.training:
        #get a single random slice from each random image specified. 
            for i, name in enumerate(fileBatch):
                img[i,:,:,0], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                img[i,:,:,0] = self.aug.apply(img[i,:,:,0])
                # if np.amax(img[i,:,:,0]) > 0.01:
                #      img[i,:,:,0] = img[i,:,:,0]/np.amax(img[i,:,:,0])

            if (np.isinf(img)).any():
                print('Infinity error')
            if (np.isnan(img)).any():
                print('Nan error')

            return img, label
        else:
            for i, name in enumerate(fileBatch):
                img[i,:,:,0], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                if np.amax(img[i,:,:,0]) > 0.01:
                    img[i,:,:,0] = img[i,:,:,0]/np.amax(img[i,:,:,0])
            return img
    
    def onEpochEnd(self):
        order = np.linspace(len(self.ids))
        np.random.shuffle(order)
        self.ids = self.ids[order]
        self.labels = self.labels[order]
        pass
    
    def __len__(self):
        l = int(np.ceil(len(self.ids))/float(self.batchSize))
        if l*self.batchSize < len(self.ids):
            l += 1
        return l
    
class ClassifyGenerator3D(K.utils.Sequence):
    def __init__(self, ids, labels, path, nclasses, batchSize=32, imageSize=192, thickness=64, training=True, 
                 x_translate=0, y_translate=0, z_translate=0, scale=0, rotate_theta=0, rotate_phi=0, rotate_psi=0,
                 flip_x=False, flip_y=False, noise=1, elastic_alpha=1, elastic_sigma=20):
        """
        Input variables:
        path      > the path that data subfolders are located in
        batchSize > how many images are processed before the model updates
                    batch is filled by random slices
        imageSize > the size of a single dimension. Assumes a square slice
        training  > sets the data to training mode vs testing mode. In training mode, ground-truth labels are
                    provided to the network, which are not needed for testing
        """
        self.ids  = ids
        self.labels = labels
        self.path = path
        self.nclasses = nclasses
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.thickness = thickness
        self.on_epoch_end()
        self.training = training

        self.aug = utils.Augmentation3D(x_translate, y_translate, z_translate, scale, rotate_theta, rotate_phi,
                                      rotate_psi, flip_x, flip_y, noise, elastic_alpha, elastic_sigma)
        
    def __load__(self, id, num): ### Will probably need to update these paths to ensure they match
        """
        loads a single set of matched data for the provided file name. Each file should be on
        a separate path. 
        """
        # Load files
        img = ants.image_read(id.as_posix())
        label = self.labels[num]
        
        # Convert to numpy arrays
        dat = img.numpy()

        # Resize
        dat = affine_transform(dat, self.aug, output_shape=(self.imageSize, self.imageSize, self.thickness))

        return dat, label
    
        
    def __getitem__(self, indx):
        """
        Retrieves a single batch of data to feed into the network. Data and
        slices are randomly generated. Intended to retrieve taining data.
        Inputs:
        
        Outputs:
        proton > a tensor of random proton slices
        xenon  > a tensor of xenon slices matching proton slices
        label  > a tensor of combined, registered 
        trans  > a tensor of transformation parameters matching the proton and
                 xenon images
        """
        fileBatch = self.ids[indx*self.batchSize : (indx+1)*self.batchSize]
         
        #initialize
        img = np.empty([self.batchSize, self.imageSize, self.imageSize, self.thickness])
        label = np.empty([self.batchSize, self.nclasses])
        
        if self.training:
        #get a single random slice from each random image specified. 
            for i, name in enumerate(fileBatch):
                img[i,:,:,:], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                img[i,:,:,:] = self.aug.apply(img[i,:,:,:])

            if (np.isinf(img)).any():
                print('Infinity error')
            if (np.isnan(img)).any():
                print('Nan error')

            return img, label
        else:
            for i, name in enumerate(fileBatch):
                img[i,:,:,:], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                if np.amax(img[i,:,:,:]) > 0.01:
                    img[i,:,:,:] = img[i,:,:,:]/np.amax(img[i,:,:,:])
            return img
    
    def onEpochEnd(self):
        pass
    
    def __len__(self):
        l = int(np.ceil(len(self.ids))/float(self.batchSize))
        if l*self.batchSize < len(self.ids):
            l += 1
        return l
    
class ClassifyGenerator25D(K.utils.Sequence):
    def __init__(self, ids, labels, path, nclasses, batchSize=32, imageSize=192, number_z_slices=3, training=True, 
                 x_translate=0, y_translate=0, scale=0, theta=0, phi=0, psi=0,
                 flip_x=False, flip_y=False, noise=0.02, elastic_alpha=0, elastic_sigma=20, elastic_z=True):
        """
        This generator is for 2.5 dimensional data: somewhere in between 2 and 3 dimensions, that is, 3D slabs sampled
        from a 3D volume. This allows multiple data 'slabs' from a single 3D volume. Needs to use 3D augmentation,
        but z-related variables (z_slices, phi, psi) should not be changed from 0.
        Input variables:
        path            > the path that data subfolders are located in
        batchSize       > how many images are processed before the model updates
                          batch is filled by random slices
        imageSize       > the size of a single dimension. Assumes a square slice
        training        > sets the data to training mode vs testing mode. In training mode, ground-truth labels are
                          provided to the network, which are not needed for testing
        number_z_slices > total number of slices in a slab. Ideally an odd number so that each side is symmetric. If
                          there are not enough slices on either side, will shift the slab "window" so it has the same
                          number of slices as the rest
        """
        self.ids             = ids
        self.labels          = labels
        self.path            = path
        self.nclasses        = nclasses
        self.batchSize       = batchSize
        self.imageSize       = imageSize
        self.on_epoch_end()
        self.training        = training
        self.number_z_slices = number_z_slices
        self.elastic_z       = elastic_z

        self.x_translate      = x_translate
        self.y_translate      = y_translate
        self.scale            = scale
        self.theta            = theta
        self.phi              = phi
        self.psi              = psi
        self.flip_x           = flip_x
        self.flip_y           = flip_y
        self.noise            = noise
        self.elastic_alpha    = elastic_alpha
        self.elastic_sigma    = elastic_sigma
        
        self.z_translate = 0 # should not be z-shifting if already using cropped slabs

        self.aug = utils.Augmentation3D(self.x_translate, self.y_translate, self.z_translate, self.scale, self.theta, 
                                        self.phi, self.psi, self.flip_x, self.flip_y, self.noise, self.elastic_alpha, 
                                        self.elastic_sigma, self.elastic_z)
        
    def __load__(self, id, num): ### Will probably need to update these paths to ensure they match
        """
        loads a single set of matched data for the provided file name. For a given slice, it will pull a 
        number of adjacent slices to make a slab. For this code to work, the slices provided must not cause
        the slab to attempt to pull slices from outside the lung. Code calling the generator should only
        provide a list of elligible files, because the generator cannot dynamically exclude files at batch
        creation without causing issues with a fixed number of training steps. 
        """
        # Determine slice number and adjacent slices
        id_name     = id.parts[-1][:-6]
        slice_num   = int(id.parts[-1][-6:-4])
        start_slice = slice_num - (self.number_z_slices-1)//2
        stop_slice  = slice_num + (self.number_z_slices-1)//2

        # Load files
        dat = np.zeros((self.imageSize, self.imageSize, self.number_z_slices))
        for i, slice in enumerate(range(start_slice, stop_slice+1)):
            temp_name  = id_name + "{:02d}".format(slice) + ".nii"
            temp_name  = id.parent / temp_name
            temp_img   = ants.image_read(temp_name.as_posix())
            # Resize
            temp_img = resize(temp_img.numpy(), (self.imageSize, self.imageSize), order=0)

            dat[:,:,i] = temp_img
        label = self.labels[num]

        return dat, label
    
        
    def __getitem__(self, indx):
        """
        Retrieves a single batch of data to feed into the network. Data and
        slices are randomly generated. Intended to retrieve taining data.
        Inputs:
        
        Outputs:
        proton > a tensor of random proton slices
        xenon  > a tensor of xenon slices matching proton slices
        label  > a tensor of combined, registered 
        trans  > a tensor of transformation parameters matching the proton and
                 xenon images
        """
        fileBatch = self.ids[indx*self.batchSize : (indx+1)*self.batchSize]
         
        #initialize
        img = np.empty([self.batchSize, self.imageSize, self.imageSize, self.number_z_slices])
        label = np.empty([self.batchSize, self.nclasses])
        
        if self.training:
        #get a single random slice from each random image specified. 
            for i, name in enumerate(fileBatch):
                img[i,:,:,:], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                img[i,:,:,:] = self.aug.apply(img[i,:,:,:])

            if (np.isinf(img)).any():
                print('Infinity error')
            if (np.isnan(img)).any():
                print('Nan error')

            return img, label
        else:
            for i, name in enumerate(fileBatch):
                img[i,:,:,:], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                if np.amax(img[i,:,:,:]) > 0.01:
                    img[i,:,:,:] = img[i,:,:,:]/np.amax(img[i,:,:,:])
            return img
    
    def onEpochEnd(self):
        pass
    
    def __len__(self):
        l = int(np.ceil(len(self.ids))/float(self.batchSize))
        if l*self.batchSize < len(self.ids):
            l += 1
        return l
    
    def init_augmentation(self):
        self.aug = utils.Augmentation3D(self.x_translate, self.y_translate, self.z_translate, self.scale,
                                        self.theta, self.phi, self.psi, self.flip_x, self.flip_y, 
                                        self.noise, self.elastic_alpha, self.elastic_sigma, self.elastic_z)
        

class ClassifyGeneratorPatches(K.utils.Sequence):
    def __init__(self, ids, labels, path, nclasses, batchSize=32, imageSize=192, training=True, patch_size=(2,2),
                 x_translate=0, y_translate=0, scale=0, rotate=0, flip_x=False, flip_y=False, noise=1,
                 elastic_alpha=1, elastic_sigma=20):
        """
        Input variables:
        path      > the path that data subfolders are located in
        batchSize > how many images are processed before the model updates
                    batch is filled by random slices
        imageSize > the size of a single dimension. Assumes a square slice
        training  > sets the data to training mode vs testing mode. In training mode, ground-truth labels are
                    provided to the network, which are not needed for testing
        """
        super().__init__()
        self.ids  = ids
        self.labels = labels
        self.path = path
        self.nclasses = nclasses
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.on_epoch_end()
        self.training = training
        self.patch_size = patch_size

        self.x_translate   = x_translate
        self.y_translate   = y_translate
        self.scale         = scale
        self.rotate        = rotate
        self.flip_x        = flip_x
        self.flip_y        = flip_y
        self.noise         = noise
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.aug = utils.Augmentation(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma)
        
    def update_transform(self):
        self.aug = utils.Augmentation(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma)
        
    def __load__(self, id, num): ### Will probably need to update these paths to ensure they match
        """
        loads a single set of matched data for the provided file name. Each file should be on
        a separate path. 
        """
        # Load files
        img = ants.image_read(id.as_posix())
        label = self.labels[num]
        
        # Convert to numpy arrays
        dat = img.numpy()

        # Resize
        dat = resize(dat, (self.imageSize, self.imageSize), order=0)

        return dat, label
    
        
    def __getitem__(self, indx):
        """
        Retrieves a single batch of data to feed into the network. Data and
        slices are randomly generated. Intended to retrieve taining data.
        Inputs:
        
        Outputs:
        proton > a tensor of random proton slices
        xenon  > a tensor of xenon slices matching proton slices
        label  > a tensor of combined, registered 
        trans  > a tensor of transformation parameters matching the proton and
                 xenon images
        """
        fileBatch = self.ids[indx*self.batchSize : (indx+1)*self.batchSize]
         
        #initialize
        img = np.empty([self.batchSize, self.imageSize, self.imageSize, 1])
        label = np.empty([self.batchSize, self.nclasses])
        
        if self.training:
        #get a single random slice from each random image specified. 
            for i, name in enumerate(fileBatch):
                img[i,:,:,0], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                img[i,:,:,0] = self.aug.apply(img[i,:,:,0])
                if np.amax(img[i,:,:,0]) > 0.01:
                     img[i,:,:,0] = img[i,:,:,0]/np.amax(img[i,:,:,0])

            if (np.isinf(img)).any():
                print('Infinity error')
            if (np.isnan(img)).any():
                print('Nan error')

            #img = np.concatenate((img, img, img), axis=-1) # To make it 'color' like model expects
            img = model.patch_extract(img, self.patch_size)

            return img, label
        else:
            for i, name in enumerate(fileBatch):
                img[i,:,:,0], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                if np.amax(img[i,:,:,0]) > 0.01:
                    img[i,:,:,0] = img[i,:,:,0]/np.amax(img[i,:,:,0])

            #img = np.concatenate((img, img, img), axis=-1)
            img = model.patch_extract(img, self.patch_size)
            return (img, )
    
    def onEpochEnd(self):
        pass
    
    def __len__(self):
        l = int(np.ceil(len(self.ids))/float(self.batchSize))
        if l*self.batchSize < len(self.ids):
            l += 1
        return l
    
class ClassifyGeneratorMultiChannel(K.utils.Sequence):
    def __init__(self, ids, labels, path, nclasses, batchSize=32, imageSize=192, training=True, 
                 x_translate=0, y_translate=0, scale=0, rotate=0, flip_x=False, flip_y=False, noise=0.,
                 elastic_alpha=1, elastic_sigma=20, noiseless_channels=[False, False, False]):
        """
        Input variables:
        path      > the path that data subfolders are located in
        batchSize > how many images are processed before the model updates
                    batch is filled by random slices
        imageSize > the size of a single dimension. Assumes a square slice
        training  > sets the data to training mode vs testing mode. In training mode, ground-truth labels are
                    provided to the network, which are not needed for testing
        """
        super().__init__()
        self.ids  = ids
        self.labels = labels
        self.path = path
        self.nclasses = nclasses
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.on_epoch_end()
        self.training = training

        self.x_translate        = x_translate
        self.y_translate        = y_translate
        self.scale              = scale
        self.rotate             = rotate
        self.flip_x             = flip_x
        self.flip_y             = flip_y
        self.noise              = noise
        self.elastic_alpha      = elastic_alpha
        self.elastic_sigma      = elastic_sigma
        self.noiseless_channels = noiseless_channels

        self.aug = utils.AugmentationNonRandom(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma)
        
    def update_transform(self, seed):
        self.aug = utils.AugmentationNonRandom(self.x_translate, self.y_translate, self.scale, self.rotate, self.flip_x, 
                                      self.flip_y, self.noise, self.elastic_alpha, self.elastic_sigma, seed, self.noiseless_channels)
        
    def __load__(self, id, num): ### Will probably need to update these paths to ensure they match
        """
        loads a single set of matched data for the provided file name. Each file should be on
        a separate path. 
        """
        # Load files
        vent = ants.image_read(id.as_posix())
        label = self.labels[num]

        def_id = pathlib.Path(str(id).replace('Vent', 'Defect'))
        mask_id = pathlib.Path(str(id).replace('Vent', 'Mask'))

        defect = ants.image_read(def_id.as_posix())
        mask   = ants.image_read(mask_id.as_posix())
        
        # Convert to numpy arrays
        ch0 = vent.numpy()
        ch1 = defect.numpy()
        ch2 = mask.numpy()

        # Normalized channels SEPARATELY
        if np.amax(ch0) > 0.01:
            ch0 = ch0/np.amax(ch0)
        if np.amax(ch2) > 0.01:
            ch1 = ch1/np.amax(ch1)
        if np.amax(ch2) > 0.01:
            ch2 = ch2/np.amax(ch2)

        # Resize image
        ch0 = resize(ch0, (self.imageSize, self.imageSize), order=0)
        ch1 = resize(ch1, (self.imageSize, self.imageSize), order=0)
        ch2 = resize(ch2, (self.imageSize, self.imageSize), order=0)

        dat = np.stack((ch0, ch1, ch2), axis=-1)


        return dat, label
    
        
    def __getitem__(self, indx):
        """
        Retrieves a single batch of data to feed into the network. Data and
        slices are randomly generated. Intended to retrieve taining data.
        Inputs:
        
        Outputs:
        proton > a tensor of random proton slices
        xenon  > a tensor of xenon slices matching proton slices
        label  > a tensor of combined, registered 
        trans  > a tensor of transformation parameters matching the proton and
                 xenon images
        """
        fileBatch = self.ids[indx*self.batchSize : (indx+1)*self.batchSize]
         
        #initialize
        img = np.empty([self.batchSize, self.imageSize, self.imageSize, 3])
        label = np.empty([self.batchSize, self.nclasses])
        
        if self.training:
        #get a single random slice from each random image specified. 
            for i, name in enumerate(fileBatch):
                rand0     = np.random.uniform(0,1,4)
                rand1     = np.random.randint(0,2,2)
                seed = np.concatenate((rand0, rand1), axis=0)
                self.update_transform(seed)
                img[i,...], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                img[i,...] = self.aug.apply(img[i,...])


            if (np.isinf(img)).any():
                print('Infinity error')
            if (np.isnan(img)).any():
                print('Nan error')

            return img, label
        else:
            for i, name in enumerate(fileBatch):
                img[i,...], label[i,:] = self.__load__(name, indx*self.batchSize + i)
                if np.amax(img[i,...]) > 0.01:
                    img[i,...] = img[i,...]/np.amax(img[i,...])
            return img
    
    def onEpochEnd(self):
        order = np.linspace(len(self.ids))
        np.random.shuffle(order)
        self.ids = self.ids[order]
        self.labels = self.labels[order]
        pass
    
    def __len__(self):
        l = int(np.ceil(len(self.ids))/float(self.batchSize))
        if l*self.batchSize < len(self.ids):
            l += 1
        return l