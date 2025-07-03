# Xenon Ventilation Classification Code
Version 1.0, Published
Alex Matheson, July 2025

This repository provides the code used in "Disease Classificatoin of Pulmonary Xenon Ventilation MRI Using Artificial Intelligence" by Matheson et al. 2025. 

---------------------------------------------
Prerequisites:
- A GPU
- CUDA Drivers

Python 3.10.8:
- antspyx
- cudnn==8.9.2.26
- opencv==4.6.0
- keras==2.10.0
- nibabel==4.0.2
- numpy==1.26.4
- tensorflow==2.10.0
- matplotlib==3.6.2
- pandas==1.5.1
- scipy==1.12.0
- sklearn==1.1.3

----------------------------------------------

Experiment design:
This code is designed to be called and run as individual experiments, using different architectures, data, or hyperparameters between experiments. These experiments are specified in a .json file, located in ./config/. What to specify in the config file:


    "session_name": "A name for this experiemnt"
    "session_date": "The Date",
    "session_description":"Notes for tracking experiment",
    
    "IMAGE_SIZE":Input size in pixels (non-compliant images will be scaled to this size),
    "NCLASSES": number of classes to classify,
    "BATCH_SIZE": number of images to process at a time,
    "NEPOCHS": number of times to repeat training (recommended >200),
    "NFOLDS": number of folds used for cross-validation,
    "epoch_save_step": how frequently to save the model (outdated),
    "snr_threshold": images with SNR below this threshold are excluded,
    "random_seed": debugging tool to ensure that training runs can be replicated,
    "Relabel_defects": toggle to re-label slices with low VDP as "normal",
    "Relabel_exceptions": toggle to re-label based on non-VDP abnormalities,
    "image_type": what sub-folder to check in the input directory, useful if there are multiple image types being stored for different experiments,
    "DATA_PATH":path to the training data,
    "PathOutput": where to store the model for this training set,

    "model": {
        "architecture": which architecture to use, see /lib/models.py for full list,
        "weights": manual weighting during training,
        "freeze_layers": freeze pre-trained layers so that only the fully connected layers are trained,
        "learning_rate": sets the starting learning rate for adam optimized training, 
        "clipvalue": clips gradient values to prevent exploding gradients, 
        "clipnorm": prevents the norm of gradients from exceeding this value, prevents exploding gradients,
        "smoothing": label-smoothing value to use to prevent divide-by-zero errors for 'perfect predictions',
        "gamma": value used to tune focal loss


    _This dictionary supplies the different possible diseases to classify and their numeric label for the network_
    "diseases":{
            "Control":0,
            "Asthma":1,
            "BOS":2,
            "BPD":3,
            "CF":4,
            "LAM":5
    },

    _Data augmentation maximum parameters, self-explanitory_
    "transform_parameters":{
        "x_translate": maximum x-translation, 
        "y_translate":15, 
        "scale":0.2, 
        "rotate":0.3, 
        "flip_x":true, 
        "noise":0.03, 
        "elastic_alpha":800,
        "elastic_sigma":35
    }
}

Once the .json file is configured, a single experiment can be run through the command line by calling `python cli_run_experiment.py config.json` a shell script has been provided to run multiple experiments, one after another, by including a list of config files to run in order.

## Logging

All terminal output is stored in a log file in /logs/ for training and validation, with the generated filename using the experiment name and date provided in the .json.

## Expected file structure
The following file structure is recommended, and is currently hard coded.

```
├──XenonVentilationAcadRadiol
│   ├── config
│   ├── lib
│   ├── logs
│       ├── train
│       ├── validation
│   ├── Output
│       ├── Trial Name
│           ├── Fold 1
│               ├── Images
│               ├── Weights
│           ├── Fold 2
```

  --------------------------------------------------------

  Please contact the author if you need more assistance setting up the environment and running the code
                
