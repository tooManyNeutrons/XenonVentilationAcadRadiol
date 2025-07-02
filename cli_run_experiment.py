# Performing classification with built-in keras models
import sys

import csv
import json
import keras as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
import sklearn.metrics as skl
import matplotlib.pyplot as plt

import classify_generator as cg
import lib.preprocess as preprocess
import lib.evaluate as evaluate
import lib.visual as visual
import model_swap
from tensorflow.keras import applications

class Logger(object):
    """
    Duplicates terminal output to a log file for later troubleshooting / debugging
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__():
        self.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.terminal != None:
            sys.stdout = self.terminal
            self.stdout = None

        if self.log != None:
            self.log.close()
            self.log = None

# load variables from config file for a single experiment
def run(json_file):
    with open(json_file, "r") as f:
        cfg = json.load(f)

    # Get the data

    train_ids, valid_ids, test_ids, train_labels, valid_labels, test_labels = preprocess.train_valid_test_split(cfg)
    loss_history = []

    # Set up data generator
    for fold in range(cfg["NFOLDS"]):
        print("Model: " + cfg['model']['architecture'] + " Fold {}".format(fold))
        print()
        savedir = cfg['PathOutput'] + "/Fold{}".format(fold+1)
        with open(savedir + "/train_ids.csv", "w") as outfile:
            wr = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            for row in train_ids[fold]:
                wr.writerow([row.parts[-1]])

        with open(savedir + "/valid_ids.csv", "w") as outfile:
            wr = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            for row in valid_ids[fold]:
                wr.writerow([row.parts[-1]])

        with open(savedir + "/test_ids.csv", "w") as outfile:
            wr = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            for row in test_ids[fold]:
                wr.writerow([row.parts[-1]])

        train_data = cg.ClassifyGenerator(train_ids[fold], train_labels[fold], cfg["DATA_PATH"], cfg["NCLASSES"], cfg["BATCH_SIZE"],
                                        cfg["IMAGE_SIZE"], training=True, **cfg["transform_parameters"])
        valid_data = cg.ClassifyGenerator(valid_ids[fold], valid_labels[fold],cfg["DATA_PATH"], cfg["NCLASSES"], cfg["BATCH_SIZE"],
                                        cfg["IMAGE_SIZE"], training=True)
        test_data  = cg.ClassifyGenerator(test_ids[fold], test_labels[fold], cfg["DATA_PATH"], cfg["NCLASSES"], cfg["BATCH_SIZE"],
                                        cfg["IMAGE_SIZE"], training=False)

        # Save terminal output for later reference
        log_file = Logger(savedir + "/log_file.log")
        sys.stdout = log_file
        print("Beginning logging, experiment date:", datetime.today().strftime('%Y-%m-%d'))

        # Get sample weights for class balancing during training
        class_weight = {}
        weight_list = []
        gt_labels = np.stack(test_labels[fold][:], 0)
        for key in cfg["diseases"].keys():
            class_weight[cfg["diseases"][key]] = np.sum(gt_labels[:,cfg["diseases"][key]])/gt_labels.shape[0]
            weight_list.append(np.sum(gt_labels[:,cfg["diseases"][key]])/gt_labels.shape[0])

        classification_model = model_swap.load_swappable_network(cfg, weight_list)
        
        classification_model.compile(optimizer=K.optimizers.Adam(cfg['model']['learning_rate'],
                                                            clipvalue=cfg['model']['clipvalue']), #.0001 used previously
                            #loss=loss.categorical_focal_loss(cfg['model']['smoothing'],
                            #                                 cfg['model']['gamma']),
                            loss=K.losses.CategoricalFocalCrossentropy(label_smoothing=cfg['model']['smoothing'],
                                                                    gamma=cfg['model']['gamma'],
                                                                    alpha=weight_list),
                            metrics=[K.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                                        K.metrics.TopKCategoricalAccuracy(2),
                                        K.metrics.Recall(),
                                        K.metrics.AUC()])
            

        # Perform training
        mc = K.callbacks.ModelCheckpoint(savedir +  "/Weights/best_weight.weights.h5", 
                                            save_weights_only=True,
                                            save_best_only=True, 
                                            verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_logs/", histogram_freq=1,)
        metric_callback = K.callbacks.CSVLogger(savedir + '/metrics.csv', separator=',', append=True)

        train_steps = len(train_ids[fold])//cfg['BATCH_SIZE']# -1 in case smaller batches are causing NaN errors
        test_steps  = len(test_ids[fold])//cfg['BATCH_SIZE']

        classification_model.fit(train_data,
                                    validation_data=valid_data, 
                                    #steps_per_epoch=train_steps,
                                    epochs=cfg['NEPOCHS'],
                                    #class_weight=class_weight, 
                                    callbacks=[mc, tensorboard_callback, metric_callback], 
                                    verbose=2)
    
        # Now train all layers / fine tune
    
        for layer in classification_model.layers:
            if not isinstance(layer, K.layers.BatchNormalization):
                layer.trainable=True

        classification_model.compile(optimizer=K.optimizers.Adam(cfg['model']['learning_rate'],
                                                        clipvalue=cfg['model']['clipvalue']), #.0001 used previously
                        #loss=loss.categorical_focal_loss(cfg['model']['smoothing'],
                        #                                 cfg['model']['gamma']),
                        loss=K.losses.CategoricalFocalCrossentropy(label_smoothing=cfg['model']['smoothing'],
                                                                gamma=cfg['model']['gamma'],
                                                                alpha=weight_list),
                        metrics=[K.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                                    K.metrics.TopKCategoricalAccuracy(2),
                                    K.metrics.Recall(),
                                    K.metrics.AUC()])
    
        classification_model.fit(train_data,
                                validation_data=valid_data, 
                                #steps_per_epoch=train_steps,
                                epochs=cfg['NEPOCHS'],
                                #class_weight=class_weight, 
                                callbacks=[mc, tensorboard_callback, metric_callback], 
                                verbose=2)

        # Perform testing
        result = classification_model.predict(test_data,
                                                batch_size=cfg['BATCH_SIZE'],
                                                steps=test_steps,
                                                verbose=2)

        # Visualize / check results
        metric_df = pd.read_csv(savedir + '/metrics.csv')
        metric_df.plot(kind='line', y=['categorical_accuracy', 'val_categorical_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(savedir +  "/acc.png")
        plt.close()

        metric_df.plot(kind='line', y=['loss', 'val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(savedir +  "/loss.png")
        plt.close()

        # Run test data

        test_steps = len(test_ids[fold]) // cfg['BATCH_SIZE']

        classification_model.load_weights(savedir +  "/Weights/best_weight.weights.h5")
        result = classification_model.predict(test_data,
                                                batch_size=cfg['BATCH_SIZE'],
                                                verbose=2)
        

        # Calculate statistics on results
        n_ids = min(len(test_ids[fold]), result.shape[0]) # Need to find a way to fix this
        result = result[:n_ids,:]

        visual.export_classification_images(result, test_data, test_ids[fold], cfg['diseases'], savedir)

        result_ids = test_ids[fold][:test_steps*cfg['BATCH_SIZE']]
        result_labels = np.stack(test_labels[fold][:], 0)
        result_labels = result_labels[:test_steps*cfg['BATCH_SIZE'],:]
        evaluate.evaluate_network(result, result_ids, result_labels, cfg, savedir)

        log_file.close()

if __name__ == '__main__':
    # load a list of different experiments to run, each with a separate json file
    json_list = sys.argv[1:]

    for file in json_list:
        run(file)