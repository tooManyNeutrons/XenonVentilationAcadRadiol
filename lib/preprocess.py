import csv
import keras as K
import numpy as np
import pathlib
import random
import os.path

import lib.check_defects as chd
import lib.utils as utils

def prepare_files(config):
    """
    Prepares data to pass to a neural network. Gets lists of valid data, removes data with poor SNR,
    and re-labels slices based on defect / observer labels. Finally, sorts data into n-fold cross 
    validation data sets.
    """

    print('Preprocessing files...')

    id_list      = []
    subject_list = []
    disease_list = []
    for path in pathlib.Path(config['DATA_PATH']).rglob("*.nii"):
        if path.parts[-2] == config['image_type']:
            id_list.append(path)
            subject_list.append(path.parts[-3])
            disease_list.append(path.parts[-4])

    ## Get the labels
    labels = [-1]*len(disease_list)
    for i, item in enumerate(disease_list):
        if item in config['diseases']:
            labels[i] = config['diseases'][item]

    all_subjects     = [subject_list[i] for i in range(len(subject_list)) if labels[i]>=0] 
    all_labels       = [labels[i] for i in range(len(subject_list)) if labels[i]>=0]
    all_ids          = [id_list[i] for i in range(len(subject_list)) if labels[i]>=0]

    # OPTIONAL: Some slices have very few defects. Set their label to 'Control' if there are little to no defects
    if config['Relabel_defects']:
        has_defects  = chd.check_defects(all_ids, threshold=1)
        all_labels = [all_labels[i] if x else 0 for i, x in enumerate(has_defects)]

    # OPTIONAL: Some images have poor SNR which may make defects difficult to discern / incorrect texture features
    # Remove images with SNR below threshold
    snr_good = chd.check_snr(all_ids, config['snr_threshold'])
    all_subjects = [all_subjects[i] for i, x in enumerate(snr_good) if x==1]
    all_labels   = [all_labels[i]  for i, x in enumerate(snr_good) if x==1]
    all_ids      = [all_ids[i] for i, x in enumerate(snr_good) if x==1 ]

    print("{} slices excluded for poor SNR".format(np.sum(np.asarray(snr_good)==0)))
    print("{} slices excluded for poor SNR".format(np.sum(np.asarray(has_defects)==0)))

    # Manual overrides: some files are not correctly being re-classified according to defects, etc.
    # Load overriding labels from a csv file and over-write for the corresponding labels
    if config['Relabel_exceptions']:
        with open('exceptions.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                subject_id = row[0]
                condition  = row[1]

                for i, id in enumerate(all_ids):
                    if subject_id in str(id):
                        if condition in config['diseases']:
                            all_labels[i] = config['diseases'][condition]

    categorical_labels = K.utils.to_categorical(all_labels, num_classes=config['NCLASSES'])

    # Get a list of unique subjects
    unique_subjects = []
    [unique_subjects.append(x) for x in all_subjects if x not in unique_subjects]

    # Split data for testing/training/validation
    train_subj, test_subj = utils.train_split_classification(unique_subjects, config['NFOLDS'])
    valid_subj   = [0]*config['NFOLDS']
    train_ids    = [0]*config['NFOLDS']
    train_labels = [0]*config['NFOLDS']
    valid_ids    = [0]*config['NFOLDS']
    valid_labels = [0]*config['NFOLDS']
    test_ids     = [0]*config['NFOLDS']
    test_labels  = [0]*config['NFOLDS']
    for foldi in range(config['NFOLDS']):
        shuffled_order      = list(range(len(train_subj[foldi])))
        random.shuffle(shuffled_order)
        train_subj[foldi] = [train_subj[foldi][x] for x in shuffled_order]
        train_subj[foldi] = train_subj[foldi][:int((6/7)*len(train_subj[foldi]))]
        valid_subj[foldi] = train_subj[foldi][int((6/7)*len(train_subj[foldi])):]

        temp_order          = [x for x in range(len(all_ids)) if all_ids[x].parts[-3] in train_subj[foldi]]
        train_ids[foldi]    = [all_ids[x] for x in temp_order]
        train_labels[foldi] = [categorical_labels[x] for x in temp_order]
        shuffled_order      = list(range(len(train_ids[foldi])))
        random.shuffle(shuffled_order)
        train_ids[foldi]    = [train_ids[foldi][x] for x in shuffled_order]
        train_labels[foldi] = [train_labels[foldi][x] for x in shuffled_order]

        temp_order          = [x for x in range(len(all_ids)) if all_ids[x].parts[-3] in valid_subj[foldi]]
        valid_ids[foldi]    = [all_ids[x] for x in temp_order]
        valid_labels[foldi] = [categorical_labels[x] for x in temp_order]
        shuffled_order      = list(range(len(valid_ids[foldi])))
        random.shuffle(shuffled_order)
        valid_ids[foldi]    = [valid_ids[foldi][x] for x in shuffled_order]
        valid_labels[foldi] = [valid_labels[foldi][x] for x in shuffled_order]

        temp_order          = [x for x in range(len(all_ids)) if all_ids[x].parts[-3] in test_subj[foldi]]
        test_ids[foldi]     = [all_ids[x] for x in temp_order]
        test_labels[foldi]  = [categorical_labels[x] for x in temp_order]
        shuffled_order      = list(range(len(test_ids[foldi])))
        random.shuffle(shuffled_order)
        test_ids[foldi]     = [test_ids[foldi][x] for x in shuffled_order]
        test_labels[foldi]  = [test_labels[foldi][x] for x in shuffled_order]

    return train_ids, valid_ids, test_ids, train_labels, valid_labels, test_labels


# Could be problems with above code, I don't think I trust it. Write custom cross-validation code

def train_valid_test_split(config):
    id_list      = []
    subject_list = []
    disease_list = []
    for path in pathlib.Path(config['DATA_PATH']).rglob("*.nii"):
        if path.parts[-2] == config['image_type']:
            id_list.append(path)
            subject_list.append(path.parts[-3])
            disease_list.append(path.parts[-4])

    ## Get the labels
    labels = [-1]*len(disease_list)
    for i, item in enumerate(disease_list):
        if item in config['diseases']:
            labels[i] = config['diseases'][item]

    all_subjects     = [subject_list[i] for i in range(len(subject_list)) if labels[i]>=0] 
    all_labels       = [labels[i] for i in range(len(subject_list)) if labels[i]>=0]
    all_ids          = [id_list[i] for i in range(len(subject_list)) if labels[i]>=0]

    # OPTIONAL: Some slices have very few defects. Set their label to 'Control' if there are little to no defects
    if config['Relabel_defects']:
        has_defects  = chd.check_defects(all_ids, threshold=1)
        all_labels = [all_labels[i] if x else 0 for i, x in enumerate(has_defects)]

    # OPTIONAL: Some images have poor SNR which may make defects difficult to discern / incorrect texture features
    # Remove images with SNR below threshold
    snr_good = chd.check_snr(all_ids, config['snr_threshold'])
    all_subjects = [all_subjects[i] for i, x in enumerate(snr_good) if x==1]
    all_labels   = [all_labels[i]  for i, x in enumerate(snr_good) if x==1]
    all_ids      = [all_ids[i] for i, x in enumerate(snr_good) if x==1 ]

    print("{} slices excluded for poor SNR".format(np.sum(np.asarray(snr_good)==0)))

    # Manual overrides: some files are not correctly being re-classified according to defects, etc.
    # Load overriding labels from a csv file and over-write for the corresponding labels
    if config['Relabel_exceptions']:
            with open('exceptions.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    subject_id = row[0]
                    condition  = row[1]

                    for i, id in enumerate(all_ids):
                        if subject_id in str(id):
                            if condition in config['diseases']:
                                all_labels[i] = config['diseases'][condition]

    # Optional: only select datasets with ventilation, defect, and mask images
    temp_all_ids    = []
    temp_all_labels = []
    
    if 'Has_defect' in config.keys():
        print("Checking if defects exist")
        for i, id in enumerate(all_ids):
            defect_id = pathlib.Path(str(id).replace('Vent', 'Defect'))
            if os.path.exists(defect_id):
                temp_all_ids.append(id)
                temp_all_labels.append(all_labels[i])
        all_ids    = temp_all_ids
        all_labels = temp_all_labels
        

    temp_all_ids    = []
    temp_all_labels = []
    
    if 'Has_mask' in config.keys():
        print("Finished Checking Defects, Checking if mask exists.")
        for i, id in enumerate(all_ids):
            mask_id = pathlib.Path(str(id).replace('Vent', 'Mask'))
            if os.path.exists(mask_id):
                temp_all_ids.append(id)
                temp_all_labels.append(all_labels[i])
        all_ids    = temp_all_ids
        all_labels = temp_all_labels
        print("Finished Checking Mask.")

    categorical_labels = K.utils.to_categorical(all_labels, num_classes=config['NCLASSES'])

    # Get a list of unique subjects
    unique_subjects = []
    [unique_subjects.append(x) for x in all_subjects if x not in unique_subjects]

    # Split data for testing/training/validation
    # Randomize unique subject order
    np.random.seed(config['random_seed'])
    np.random.shuffle(unique_subjects)

    # Split into train/testing/valid based on ratios
    train_subj, valid_subj, test_subj = utils.three_group_k_fold(unique_subjects, config['NFOLDS'])

    train_ids    = [0]*config['NFOLDS']
    train_labels = [0]*config['NFOLDS']
    valid_ids    = [0]*config['NFOLDS']
    valid_labels = [0]*config['NFOLDS']
    test_ids     = [0]*config['NFOLDS']
    test_labels  = [0]*config['NFOLDS']
    for foldi in range(config['NFOLDS']):
        train_ids[foldi]    = [x for x in all_ids if x.parts[-3] in train_subj[foldi]]
        np.random.shuffle(train_ids[foldi])
        train_labels[foldi] = [categorical_labels[all_ids.index(x)] for x in train_ids[foldi]]

        valid_ids[foldi]    = [x for x in all_ids if x.parts[-3] in valid_subj[foldi]]
        np.random.shuffle(valid_ids[foldi])
        valid_labels[foldi] = [categorical_labels[all_ids.index(x)]for x in valid_ids[foldi]]

        test_ids[foldi]    = [x for x in all_ids if x.parts[-3] in test_subj[foldi]]
        np.random.shuffle(test_ids[foldi])
        test_labels[foldi] = [categorical_labels[all_ids.index(x)] for x in test_ids[foldi]]

    return train_ids, valid_ids, test_ids, train_labels, valid_labels, test_labels