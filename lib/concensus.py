from keras.metrics import Recall, Precision
from keras.utils import to_categorical
import pandas as pd
import sklearn.metrics as skl
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np

def load_from_csv(path):
    result_df = pd.read_csv(path)
    result_df.reset_index()
    result_df = result_df.drop(0)

    columns = result_df.columns.to_list()
    subject_list = {}
    for index, row in result_df.iterrows():
        if row['subject'] not in subject_list.keys():
            subject_list[row['subject']] = {}
        subject_list[row['subject']]['True'] = row['disease']
        if 'Slices' not in subject_list[row['subject']].keys():
            subject_list[row['subject']]['Slices'] = {}
        subject_list[row['subject']]['Slices'][int(row['slice'])] = [row[x] for x in columns if 'pred' in x]

    return subject_list

def concensus(path, disease_dict, nclasses):
    subject_list = {}

    #Get a new structure for each individual subject to collect the slice-wise predictions
    subject_list = load_from_csv(path)
        
    # Find the most selected answer 
    gt_labels = np.zeros([len(subject_list.keys()), nclasses])
    preds     = np.zeros_like(gt_labels)
    for i, subject in enumerate(subject_list.keys()):
        temp_pred = []
        for slice_number in subject_list[subject]['Slices'].keys():
            temp_pred.append(subject_list[subject]['Slices'][slice_number])
            print(subject, subject_list[subject]['True'], subject_list[subject]['Slices'][slice_number])

        # Determine the frequency of each guess for a given subject
        slice_pred = np.argmax(np.asanyarray(temp_pred), 1)
        n_slice_pred = np.zeros(nclasses)
        for j in range(nclasses): # Determine how many slices were classified as each disease
            n_slice_pred[j] = np.sum(slice_pred==j)
        order_of_likelihood = np.argsort(n_slice_pred) # Get ordered list of most disease classifications
        n_slice_pred_sorted = np.sort(n_slice_pred) # Find out frequency of each classification

        # # Shouldn't label control if any slices are classified as disease
        # if (order_of_likelihood[-1]==0) and (n_slice_pred_sorted[-1] < 0.3*int(len(subject_list[subject]['Slices'].keys()))):
        #     subject_list[subject]['Prediction Modified Mode'] = \
        #     list(disease_dict.keys())[list(disease_dict.values()).index(order_of_likelihood[-2])]
        # else:
        #     subject_list[subject]['Prediction Modified Mode'] = \
        #     list(disease_dict.keys())[list(disease_dict.values()).index(order_of_likelihood[-1])]

        # # Try another version of this using MODE, but let the 3 least likely classes override if n>2
        # if np.any(n_slice_pred[1:4]>1):
        #     subject_list[subject]['Prediction Small Classes'] = \
        #     list(disease_dict.keys())[list(disease_dict.values()).index(np.argmax(n_slice_pred[1:4]+1))]
        # else:
        #     temp_mode = mode(np.argmax(np.asarray(temp_pred), 1))[0]
        #     subject_list[subject]['Prediction Small Classes'] = \
        #     list(disease_dict.keys())[list(disease_dict.values()).index(temp_mode)]

        # Store the top-2 choice
        subject_list[subject]['2nd Choice'] = list(disease_dict.keys())[list(disease_dict.values()).index(order_of_likelihood[-2])]

        # Other methods by standard statistics
        temp_mode = mode(np.argmax(np.asarray(temp_pred), 1))[0]
        subject_list[subject]['Prediction Mode'] = list(disease_dict.keys())[list(disease_dict.values()).index(temp_mode)]
        temp_mean = np.argmax(np.sum(np.asarray(temp_pred), 0))
        subject_list[subject]['Prediction Mean'] = list(disease_dict.keys())[list(disease_dict.values()).index(temp_mean)]

        gt_labels[i] = to_categorical(disease_dict[subject_list[subject]['True']], nclasses)
        preds[i]     = np.mean(np.asarray(temp_pred), 0)

        print(subject_list[subject]['Prediction Mean'], np.sum(np.asarray(temp_pred), 0))

    rec = Recall()
    pre = Precision()
    rec_result = rec(gt_labels, preds)
    pre_result = pre(gt_labels, preds)

    # Compute Total Accuracy
    total_n_disease = [0, 0, 0, 0, 0, 0, 0, 0]
    total_correct   = [0, 0, 0, 0, 0, 0, 0, 0]
    top2_correct    = [0, 0, 0, 0, 0, 0, 0, 0]
    for subject in subject_list.keys():
        total_n_disease[disease_dict[subject_list[subject]['True']]] += 1
        if subject_list[subject]['True'] == subject_list[subject]['Prediction Mean']:
            total_correct[disease_dict[subject_list[subject]['True']]] += 1
        if (subject_list[subject]['True'] == subject_list[subject]['Prediction Mean']) | (subject_list[subject]['True'] == subject_list[subject]['2nd Choice']):
            top2_correct[disease_dict[subject_list[subject]['True']]] += 1

    auc_total = skl.roc_auc_score(gt_labels, preds, average='micro', multi_class='ovr')

    print("Accuracy: ", np.sum(total_correct)/np.sum(total_n_disease))
    print("By disease: ", np.divide(total_correct, total_n_disease))
    print("Top-2 Accuracy: ", np.sum(top2_correct)/np.sum(total_n_disease))
    print("By disease: ", np.divide(top2_correct, total_n_disease))
    print("Recall: ", rec_result)
    print("Precision: ", pre_result)
    print("Micro-average AUC: ", auc_total)

    plt.figure(figsize=(20,20))
    confusion_matrix = skl.confusion_matrix([subject_list[x]['True'] for x in subject_list.keys()],
                                            [subject_list[x]['Prediction Mean'] for x in subject_list.keys()],
                                            labels=list(disease_dict.keys()), 
                                            normalize='true')
    conf_plot = skl.ConfusionMatrixDisplay(confusion_matrix, display_labels=list(disease_dict.keys()))
    conf_plot.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.show()

    # Make ROC curves for each class this way
    fig, ax = plt.subplots(figsize=(6,6))
    result_onehot = np.zeros_like(preds, dtype=bool)
    result_onehot[np.arange(preds.shape[0]), np.argmax(preds, -1)] = 1
    
    for class_id in range(nclasses):
        skl.RocCurveDisplay.from_predictions(
            gt_labels[:,class_id],
            preds[:,class_id],
            name='ROC curve for {}'.format(list(disease_dict.keys())[list(disease_dict.values()).index(class_id)]),
            ax=ax,
            plot_chance_level=(class_id == 5)
        )
