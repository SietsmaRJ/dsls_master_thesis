import os
import pandas as pd
import numpy as np
from sklearn import metrics

# Change working directory to the data files containing folder
os.chdir(
    "/home/sietsmarj/Documents/School/Master_DSLS/Final_Thesis/Initial_Data_exploration")

# Load in training data
var_train_res = pd.read_csv("train_results.txt", sep='\t', header=0)
var_train_res.columns = ["chr", "pos", "ref", "alt", "gene", "csq1", "cadd",
                         "capice", "pred", "combpred"]
var_train_lable = pd.read_csv("train.txt", sep='\t', header=0)
var_train_lable.columns = ["chr", "pos", "ref", "alt", "csq2", "maf", "label"]

# Merging and deleting
capice_train = pd.merge(var_train_res, var_train_lable)
capice_train_nrows = capice_train.shape[0]
print("Train dataset NaN counting.")
for nan in capice_train.columns:
    try:
        print(f"Column {nan}: {capice_train[nan].isna().sum()}"
              f" of the {capice_train_nrows}."
              f" Which is: {capice_train[nan].isna().sum() / capice_train_nrows * 100} %")
    except TypeError:
        continue
del (var_train_lable, var_train_res)

# Labeling
capice_train['src'] = 'CAPICE_train'

# Loading in the test data
var_test = pd.read_csv('test_results.txt', sep='\t', header=0)
var_test.columns = ["chr", "pos", "ref", "alt", "maf", "csq3", "label", "revel",
                    "clinpred", "sift", "provean", "cadd", "fathmm", "capice",
                    "ponp2"]
vart_test_nrows = var_test.shape[0]
print("Test dataset NaN counting.")
for nan in var_test.columns:
    print(f"Column {nan}: {var_test[nan].isna().sum()}"
          f" of the {vart_test_nrows}."
          f" Which is: {var_test[nan].isna().sum() / vart_test_nrows * 100} %")
var_test['src'] = 'CAPICE_test'

capice = capice_train[
    ["chr", "pos", "ref", "alt", "src", "label", "capice"]].append(
    var_test[["chr", "pos", "ref", "alt", "src", "label", "capice"]])
del (capice_train, var_test)

vkgl_labels = pd.read_csv("VKGL_14nov2019.txt", sep='\t', header=0)
vkgl_labels.columns = ["variant", "chr", "pos", "ref", "alt", "cdna", "protein",
                       "transcript", "hgvs", "gene", "label", "nrlabs"]
vkgl_res = pd.read_csv("VKGL_14nov2019_capice.txt", sep='\t', header=0)
vkgl_res.columns = ["chr", "pos", "ref", "alt", "csq4", "capice"]
vkgl = pd.merge(vkgl_res, vkgl_labels)
vkgl_nrows = vkgl.shape[0]
print("VKGL dataset NaN counting.")
for nan in vkgl.columns:
    print(f"Column {nan}: {vkgl[nan].isna().sum()}"
          f" of the {vkgl_nrows}."
          f" Which is: {vkgl[nan].isna().sum() / vkgl_nrows * 100} %")
vkgl['src'] = 'VKGL_14nov2019'

del (vkgl_labels, vkgl_res)

overlap_capice = pd.merge(vkgl, capice, on=['chr', 'pos', 'ref', 'alt'])[
    ['chr', 'pos', 'ref', 'alt']]
temp_not_overlap = pd.merge(vkgl, overlap_capice,
                            on=['chr', 'pos', 'ref', 'alt'], how='outer',
                            indicator=True)
vkgl1NIC = temp_not_overlap[temp_not_overlap['_merge'] == 'left_only']
vkgl1IC = pd.merge(vkgl, overlap_capice, on=['chr', 'pos', 'ref', 'alt'])

del (overlap_capice, temp_not_overlap)

all = capice[["chr", "pos", "ref", "alt", "src", "label", "capice"]].append(
    vkgl1NIC[["chr", "pos", "ref", "alt", "src", "label", "capice"]]
)
del (vkgl1NIC, vkgl1IC)

exons = pd.read_csv("Ensembl_GRCh37_ExonRegions.txt", sep='\t', header=0)

var_res = pd.read_csv("train_results.txt", header=0, sep='\t')
var_inp = pd.read_csv("train.txt", header=0, sep='\t')

# print(var_res, '\n', "______________", '\n', var_inp)

var_res = var_res[['chr', 'pos', 'ref', 'alt', 'GeneName', 'prediction']]
var_inp = var_inp[['#Chrom', 'Pos', 'Ref', 'Alt', 'label']]

var_comb = pd.merge(var_res, var_inp, left_on=['chr', 'pos', 'ref', 'alt'],
                    right_on=['#Chrom', 'Pos', 'Ref', 'Alt'])

var_comb = var_comb.drop(['#Chrom', 'Pos', 'Ref', 'Alt'], axis=1)

var_comb['prediction'] = var_comb['prediction'].replace({"Pathogenic":1,"Neutral":0})

var_comb['label'] = var_comb['label'].replace({"LP/P":1,"LB/B":0})

overview = pd.DataFrame(0, np.arange(1), columns=['Gene', 'AUC', 'FPR', 'Precision',
                                 'Recall/TPR', 'F-score', 'N_benign',
                                                  'N_malign'])


for genename in var_comb['GeneName'].unique():
    subset_data = var_comb[var_comb['GeneName'] == genename]
    y_true = np.array(subset_data['label'])
    y_pred = np.array(subset_data['prediction'])
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        auc = metrics.roc_auc_score(y_true, y_pred)
    else:
        auc = np.NaN
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    fpr = 1-recall
    f_score = metrics.f1_score(y_true, y_pred)
    n_benign = subset_data[subset_data['label'] == 0]['label'].count()
    n_malign = subset_data[subset_data['label'] == 1]['label'].count()
    # //TODO: add has_m_cat as metric (0=no, 1=yes)

    row = pd.DataFrame(index=np.arange(1),
                       data={'Gene': genename, 'AUC': auc, 'FPR': fpr,
                        'Precision': precision, 'Recall/TPR': recall,
                        'F-score': f_score, 'N_benign': n_benign,
                        'N_malign': n_malign})
    overview = overview.append(row, ignore_index=True)
print(overview)
overview = overview.drop(0)
overview.to_csv("auc_results.csv")