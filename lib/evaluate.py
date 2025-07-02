# Code to evaluate statistics on neural network classification results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as K
import tensorflow as tf
import sklearn.metrics as skl

def evaluate_network(result, ids, labels, cfg, savedir):
    # Set up some default info for rest of code
    plt.rc('legend', fontsize=12)
    plt.rcParams.update({'font.size':16})

    columns = ["subject", "slice", "file location", "disease",
        "true_0", "true_1", "true_2", "true_3", "true_4", "true_5",
        "pred_0", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]
    
    # Calculate statistics on results
    n_ids = min(len(ids), result.shape[0]) # Need to find a way to fix this
    result = result[:n_ids,:]

    # Export results to a csv
    export_data = [[]]
    for j, item in enumerate(ids):
        line = np.concatenate(
                            [np.array([item.parts[-1][:-7]]), 
                            np.array([item.parts[-1][-6:-4]]),
                            np.array([item]),
                            np.array([item.parts[-4]]), 
                            labels[j],
                            result[j,:]], 
                            0)
        export_data.append(line)

    df = pd.DataFrame(data = export_data, columns=columns)
    df.to_csv(cfg['PathOutput'] + '/results.csv')

    # Calculate accuracy
    acc = np.sum(np.logical_and(labels[:n_ids], 
                                K.utils.to_categorical(np.argmax(result,-1), 
                                num_classes=cfg['NCLASSES'])))/result.shape[0]
    top_2_indices = np.argsort(result,-1)[:,-2:]
    top_2_indices = np.sum(K.utils.to_categorical(top_2_indices, num_classes=cfg['NCLASSES']),1, dtype='uint8')
    acc_top_2 = np.sum(np.any(np.logical_and(labels[:n_ids], top_2_indices),-1))/result.shape[0]
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

    gt_labels = np.stack(labels[:n_ids], 0)
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
    print("Total CCE: {}".format(cce_total))
    print("Total recall: {}".format(recall_total))
    print("Total precision: {}".format(precision_total))
    print("Micro-average AUC: {}".format(auc_total))
    print("Disease accuracy: {}".format(acc_disease))
    print("Top 2 accuracy: {}".format(acc_top_disease))
    print("Disease CCE: {}".format(cce_disease_all))
    print("Disease recall: {}".format(rec_disease_all))
    print("Disease precision: {}".format(pre_disease_all))
    print("F1 by class: {}".format(f1_disease_all))
    #print("AUC by glass: {}".format(auc_disease_all))

    gt_labels = np.stack(labels)
    for i in range(cfg['NCLASSES']):
        class_weight_all.append(np.sum(gt_labels[:,i])/gt_labels.shape[0])

    print("Class weight all participants: {}".format(class_weight_all))

    # Calculate confusion matrix

    gt_labels = np.stack(labels[:n_ids], 0)
    gt_names = [list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(x)] for x in iter(np.argmax(gt_labels, -1))]
    result_names = [list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(x)] for x in iter(np.argmax(result, -1))]

    plt.figure(figsize=(20,20))
    confusion_matrix = skl.confusion_matrix(gt_names, result_names, labels=list(cfg['diseases'].keys()), normalize='true')
    conf_plot = skl.ConfusionMatrixDisplay(confusion_matrix, display_labels=list(cfg['diseases'].keys()))
    conf_plot.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90)
    plt.savefig(savedir +  "/confusion.png")
    plt.close()

    # Custom top-2 confusion matrix
    conf2 = np.zeros([cfg['NCLASSES'], cfg['NCLASSES']])
    top_2_indices = np.argsort(result,-1)[:,-2]
    gt_args = np.argmax(labels, -1)
    result_args = np.argmax(result, -1)
    for i in range(cfg['NCLASSES']):
        for j in range(cfg['NCLASSES']):
            conf2[i, j] = np.sum((gt_args==i)&(top_2_indices==j))/np.sum(gt_args==i)
    conf_plot = skl.ConfusionMatrixDisplay(conf2, display_labels=list(cfg['diseases'].keys()))
    conf_plot.plot(cmap=plt.cm.Greens, values_format='.2f')
    plt.xticks(rotation=90)
    plt.savefig(savedir +  "/confusion_2nd_only.png")
    plt.close()

    conf_plot = skl.ConfusionMatrixDisplay(confusion_matrix + conf2, display_labels=list(cfg['diseases'].keys()))
    conf_plot.plot(cmap=plt.cm.Reds, values_format='.2f')
    plt.xticks(rotation=90)
    plt.savefig(savedir +  "/confusion_top_2.png")
    plt.close()



    # Calculate one-versus-all AUCs
    fig, ax = plt.subplots(figsize=(6,6))
    result_onehot = np.zeros_like(result, dtype=bool)
    result_onehot[np.arange(result.shape[0]), np.argmax(result, -1)] = 1
    
    rate_list = []
    for class_id in range(cfg['NCLASSES']):
        skl.RocCurveDisplay.from_predictions(
            gt_labels[:,class_id],
            result[:,class_id],
            name='ROC curve for {}'.format(list(cfg['diseases'].keys())[list(cfg['diseases'].values()).index(class_id)]),
            ax=ax,
            plot_chance_level=(class_id == 5)
        )
        # Get ROC TPF FNF 
        fpr, tpr, thresholds = skl.roc_curve(
            gt_labels[:, class_id],
            result[:,class_id]
        )
        rates = np.stack((fpr, tpr, thresholds), axis=1)
        rate_list.append(rates)
        
    # Making a .csv of tpr thresholds for each class
    max_len = np.max([x.shape[0] for x in rate_list])
    rates = np.array([])
    for i in range(cfg['NCLASSES']):
        rate_list[i] = np.pad(rate_list[i], ((0, max_len - rate_list[i].shape[0]), (0,0)), constant_values=-1)
        rates = np.concatenate((rates,rate_list[i])) if rates.size else rate_list[i]
    
    rates[rates==-1] = None
    roc_df = pd.DataFrame(data=rates)
    roc_df.to_csv(savedir + 'roc_thresholds.csv')

    # Plotting the ROC
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class-wise One-versus-all AUC')
    plt.savefig(savedir +  "/roc.png")
    plt.close()

    

