# Debugging the standalone python file
# September 6 2024

# Temporary Changes Sept 18 for two-step training

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
import model_swap
from tensorflow.keras import applications

class Logger(object):
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

# load variables from config file
with open("config/config_18092024_VGG16Transfer.json", "r") as f:
    cfg = json.load(f)

# Get the data

train_ids, valid_ids, test_ids, train_labels, valid_labels, test_labels = preprocess.train_valid_test_split(cfg)
loss_history = []

# Set up data generator
for fold in range(1):
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
    mc = K.callbacks.ModelCheckpoint(savedir +  "/Weights/best_weight_temp.weights.h5", 
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
                                epochs=50,
                                #class_weight=class_weight, 
                                callbacks=[mc, tensorboard_callback, metric_callback], 
                                verbose=2)
    
    # Now train all layers / fine tune
    
    for layer in classification_model.layers:
        if not isinstance(layer, K.layers.BatchNormalization):
            layer.trainable=True

    classification_model.compile(optimizer=K.optimizers.Adam(cfg['model']['learning_rate']*.1,
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
                                epochs=300,
                                #class_weight=class_weight, 
                                callbacks=[mc, tensorboard_callback, metric_callback], 
                                verbose=2)
    
    # Perform testing
    result = classification_model.predict(test_data,
                                            batch_size=cfg['BATCH_SIZE'],
                                            steps=test_steps,
                                            verbose=2)

    # Visualize / check results
    # loss_loss = np.asarray(classification_model.history['loss'])
    # loss_loss[loss_loss>10] = np.median(loss_loss)
    # plt.plot(loss_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss (CCE)')
    # val_loss = np.asarray(classification_model.history['val_loss'])
    # val_loss[val_loss>20] = np.median(val_loss)
    # plt.plot(val_loss)
    # plt.savefig(savedir +  "/loss.png")
    # plt.close()
    #json.dump(classification_model.history, open(savedir + "/training_history.json", 'w' ))

    # plt.plot(classification_model.history['categorical_accuracy'])
    # plt.plot(classification_model.history['val_categorical_accuracy'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig(savedir + "/acc.png")
    # plt.close()

    # Run test data

    test_steps = len(test_ids[fold]) // cfg['BATCH_SIZE']

    classification_model.load_weights(savedir +  "/Weights/best_weight.weights.h5")
    result = classification_model.predict(test_data,
                                            batch_size=cfg['BATCH_SIZE'],
                                            verbose=2)
    
    columns = ["fold", "subject", "slice", "file location", "disease",
        "true_0", "true_1", "true_2", "true_3", "true_4", "true_5",
        "pred_0", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]


    # Calculate statistics on results
    n_ids = min(len(test_ids[fold]), result.shape[0]) # Need to find a way to fix this
    result = result[:n_ids,:]

    export_data = [[]]
    for i in range(fold, fold+1): # Careful if you are't doing all folds
        for j, item in enumerate(test_ids[i][:n_ids]):
            line = np.concatenate([np.array([i], dtype='uint8'), 
                                np.array([item.parts[-1][:-7]]), 
                                np.array([item.parts[-1][-6:-4]]),
                                np.array([item]),
                                np.array([item.parts[-4]]), 
                                test_labels[i][j],
                                result[j,:]], 
                                0)
            export_data.append(line)

    df = pd.DataFrame(data = export_data, columns=columns)
    #df.to_csv('./Output/Vent_result_Feb_29.csv')

    # Calculate accuracy
    acc = np.sum(np.logical_and(test_labels[fold][:n_ids], 
                                K.utils.to_categorical(np.argmax(result,-1), 
                                num_classes=cfg['NCLASSES'])))/result.shape[0]
    top_2_indices = np.argsort(result,-1)[:,-2:]
    top_2_indices = np.sum(K.utils.to_categorical(top_2_indices, num_classes=cfg['NCLASSES']),1, dtype='uint8')
    acc_top_2 = np.sum(np.any(np.logical_and(test_labels[fold][:n_ids], top_2_indices),-1))/result.shape[0]
    print("Total accuracy: {}".format(acc))
    print("Total top 2 accuracy: {}".format(acc_top_2))

    # Calculate disease-wise accuracy
    acc_disease      = []
    acc_top_disease  = []
    class_weight     = []
    class_weight_all = []
    cce_disease_all  = []
    rec_disease_all  = []
    pre_disease_all  = []
    f1_disease_all   = []
    auc_disease_all  = []

    cce = K.losses.CategoricalCrossentropy()
    rec = K.metrics.Recall()
    pre = K.metrics.Precision()

    gt_labels = np.stack(test_labels[fold][:n_ids], 0)
    bin_result = tf.one_hot(np.argmax(result, axis=-1), cfg['NCLASSES'], axis=-1).numpy()

    cce_total = cce(gt_labels, result)
    recall_total = rec(gt_labels, result)
    precision_total = pre(gt_labels, result)

    for i in range(cfg['NCLASSES']):
        acc_disease.append(np.sum(np.argmax(result[gt_labels[:,i]==1,:],1)==i)/np.sum(gt_labels[:,i]))
        acc_top_disease.append(np.sum((gt_labels[:,i]==1) & np.any(np.argsort(result,-1)[:,-2:]==i,-1))/np.sum(gt_labels[:,i]))
        cce_disease_all.append(cce(gt_labels[gt_labels[:,i]==1], result[gt_labels[:,i]==1]).numpy())
        rec_disease_all.append(rec(gt_labels[gt_labels[:,i]==1], result[gt_labels[:,i]==1]).numpy())
        pre_disease_all.append(pre(gt_labels[gt_labels[:,i]==1], result[gt_labels[:,i]==1]).numpy())
        f1_disease_all.append(skl.f1_score(gt_labels[:,i], bin_result[:,i]))

    f1_total = skl.f1_score(np.argmax(gt_labels, -1), np.argmax(result, -1), average='micro')
    auc_total = skl.roc_auc_score(gt_labels, result, average='micro', multi_class='ovr')

    # Careful, alpha might change fold by fold
    loss=K.losses.CategoricalFocalCrossentropy(label_smoothing=cfg['model']['smoothing'],
                                                                        gamma=cfg['model']['gamma'],
                                                                        alpha=weight_list)
    focal_total = loss(gt_labels, result)

    print("Total CCE: {}".format(cce_total))
    print("Total recall: {}".format(recall_total))
    print("Total precision: {}".format(precision_total))
    print("Total F1 score: {}".format(f1_total))
    print("Total Focal Loss: {}".format(focal_total))
    print("Micro-average AUC: {}".format(auc_total))
    print("Disease accuracy: {}".format(acc_disease))
    print("Top 2 accuracy: {}".format(acc_top_disease))
    print("Disease CCE: {}".format(cce_disease_all))
    print("Disease recall: {}".format(rec_disease_all))
    print("Disease precision: {}".format(pre_disease_all))
    print("F1 by class: {}".format(f1_disease_all))
    #print("AUC by glass: {}".format(auc_disease_all))

    gt_labels = np.stack(test_labels[fold])
    for i in range(cfg['NCLASSES']):
        class_weight_all.append(np.sum(gt_labels[:,i])/gt_labels.shape[0])

    print("Class weight all participants: {}".format(class_weight_all))

    # Calculate confusion matrix

    gt_labels = np.stack(test_labels[fold][:n_ids], 0)
    gt_names = [list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(x)] for x in iter(np.argmax(gt_labels, -1))]
    result_names = [list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(x)] for x in iter(np.argmax(result, -1))]

    plt.figure(figsize=(20,20))
    confusion_matrix = skl.confusion_matrix(gt_names, result_names, labels=list(cfg['diseases'].keys()), normalize='true')
    conf_plot = skl.ConfusionMatrixDisplay(confusion_matrix, display_labels=list(cfg['diseases'].keys()))
    conf_plot.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90)
    plt.rcParams.update({'font.size':16})
    plt.savefig(savedir +  "/confusion.png")
    plt.close()

    # Calculate one-versus-all AUCs
    fig, ax = plt.subplots(figsize=(6,6))
    result_onehot = np.zeros_like(result, dtype=bool)
    result_onehot[np.arange(result.shape[0]), np.argmax(result, -1)] = 1
    for class_id in range(cfg['NCLASSES']):
        skl.RocCurveDisplay.from_predictions(
            gt_labels[:,class_id],
            result[:,class_id],
            name='ROC curve for {}'.format(list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(class_id)]),
            ax=ax,
            plot_chance_level=(class_id == 5)
        )
    plt.rc('legend',fontsize=6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class-wise One-versus-all AUC')
    plt.savefig(savedir +  "/roc.png")
    plt.close()

    log_file.close()


