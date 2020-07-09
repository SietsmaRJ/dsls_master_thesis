import sys
import random
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from impute_preprocess import (preprocess, cadd_vars, impute,
                               cadd_vars_withConsequence, preprocess_withConsequence)
import argparse


def getvariants_withinrange(variants, upperBound, lowerBound):
    variants_below_upperBound = variants[variants['max_AF'] < upperBound]
    variants_withinrange = variants_below_upperBound[variants_below_upperBound['max_AF'] >= lowerBound]
    return variants_withinrange


def select_balanced_benign_dataset(clinvar_pathogenic_test1, pnv, bins=[0, 0.01, 0.05, 0.1, 0.5, 1]):
    for conse_ind, consequence in enumerate(clinvar_pathogenic_test1['Consequence'].unique()):
        print("\nConsequence:", consequence)
        clinvar_consequence = clinvar_pathogenic_test1[clinvar_pathogenic_test1['Consequence'] == consequence]
        pnv_consequence = pnv[pnv['Consequence'] == consequence]
        if clinvar_consequence.shape[0] > pnv_consequence.shape[0]:
            clinvar_consequence = clinvar_consequence.sample(pnv_consequence.shape[0], random_state=45)
            print(clinvar_consequence.shape, pnv_consequence.shape)
        clinvar_consequence_histograms, bins = np.histogram(clinvar_consequence['max_AF'], bins=bins)
        for ind in range(len(bins) - 1):
            lower_bound = bins[ind]
            upper_bound = bins[ind + 1]
            selected_pnv_all = getvariants_withinrange(pnv_consequence, upperBound=upper_bound, lowerBound=lower_bound)
            selected_pathogenic_all = getvariants_withinrange(clinvar_consequence, upperBound=upper_bound,
                                                              lowerBound=lower_bound)
            sample_num = clinvar_consequence_histograms[ind]
            if sample_num < selected_pnv_all.shape[0]:
                selected_pnv_thisrange = selected_pnv_all.sample(sample_num, random_state=45)
                selected_pathogenic_thisrange = selected_pathogenic_all
            else:
                selected_pnv_thisrange = selected_pnv_all
                selected_pathogenic_thisrange = selected_pathogenic_all.sample(selected_pnv_all.shape[0],
                                                                               random_state=45)
            print("Sampled %d variants from pnvs for %s:" % (selected_pnv_thisrange.shape[0], consequence))
            print("Sampled %d variants from Pathogenic for %s:" % (selected_pathogenic_thisrange.shape[0], consequence))
            if ind == 0:
                selected_pnv = selected_pnv_thisrange
                selected_pathogenic = selected_pathogenic_thisrange
            else:
                selected_pnv = pd.concat([selected_pnv_thisrange, selected_pnv], axis=0)
                selected_pathogenic = pd.concat([selected_pathogenic, selected_pathogenic_thisrange], axis=0)
                print(selected_pathogenic_thisrange.shape[0], selected_pathogenic.shape[0])
        pnv_consequence_histogram, _ = np.histogram(selected_pnv['max_AF'], bins=bins)
        pathogenic_histogram, _ = np.histogram(selected_pathogenic['max_AF'], bins=bins)
        print(pathogenic_histogram)
        print(pnv_consequence_histogram)
        print(bins)
        if conse_ind == 0:
            selected_pnv_allconsequences = selected_pnv
            selected_pathogenic_allconsequences = selected_pathogenic
        else:
            selected_pnv_allconsequences = pd.concat([selected_pnv, selected_pnv_allconsequences], axis=0)
            selected_pathogenic_allconsequences = pd.concat([selected_pathogenic_allconsequences,
                                                             selected_pathogenic], axis=0)
            print(selected_pathogenic_allconsequences.shape[0])
    return selected_pnv_allconsequences, selected_pathogenic_allconsequences


new_training = pd.read_csv("/home/rjsietsma/PycharmProjects/dsls_master_thesis/side_scripts/datafiles/train.txt.gz", sep='\t')
new_pathogenic = new_training[new_training['binarized_label'] == 1]
selected_pathogenic, selected_benign = select_balanced_benign_dataset(new_pathogenic,
                                                                      new_training[new_training['binarized_label'] == 0],
                                                                      bins=[0, 0.1, 1])
print('Appended')
print(selected_pathogenic.append(selected_benign).shape[0])
train_balanced, test = train_test_split(pd.concat([selected_pathogenic, selected_benign], axis=0),
                                        random_state=4, test_size=0.2)
all_variants = set(train_balanced['chr_pos_ref_alt']) | set(test['chr_pos_ref_alt'])
inbalancedsets = lambda x: True if x in all_variants else False
new_training['inbalancedsets'] = [inbalancedsets(variant) for variant in new_training['chr_pos_ref_alt']]
train = pd.concat([train_balanced, new_training[new_training['inbalancedsets'] == False]], axis=0)
print('Last train')
print(train.shape[0])

# train, test = train_test_split(new_training, test_size=0.2, random_state=4)
# train_preprocessed = preprocess(impute(train), isTrain=True)
# processed_features = []
# for col in train_preprocessed.columns:
#     for feat in cadd_vars:
#         if col == feat or col.startswith(feat):
#             if col not in processed_features:
#                 processed_features.append(col)
# basic model would stop at test AUC 0.86... for some reason
# train_balanced_preprocessed = preprocess(impute(train_balanced), isTrain=True)
# processed_features_balanced = []
# for col in train_balanced_preprocessed.columns:
#     for feat in cadd_vars:
#         if col == feat or col.startswith(feat):
#             if col not in processed_features_balanced:
#                 processed_features_balanced.append(col)


# a basic model
def basic_model():
    scale_pos_weight = train[train['binarized_label'] == 1].shape[0] / train[train['binarized_label'] == 0].shape[0]
    basic_model = xgb.XGBClassifier(max_depth=12, learning_rate=0.1, n_estimators=500, verbosity=1,
                                    objective='binary:logistic', booster='gbtree', n_jobs=8,
                                    min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                    colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                    scale_pos_weight=scale_pos_weight, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    eval_set = [(test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
    print("=====================================Started training...=============================================")
    basic_model.fit(train_preprocessed[processed_features], train_preprocessed['binarized_label'],
                    early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #        colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
    #        max_delta_step=0, max_depth=12, min_child_weight=1, missing=None,
    #        n_estimators=500, n_jobs=8, nthread=None,
    #        objective='binary:logistic', random_state=0, reg_alpha=0,
    #        reg_lambda=1, scale_pos_weight=1.0049009311769237, seed=None,
    #        silent=True, subsample=1, verbosity=1)
    pickle.dump(basic_model, open('model/different_training/basic_model.pickle.dat', 'wb'))
    return basic_model


# basic_model()

# Find the number of estimators for a high learning rate
import time
import scipy


def randomsearch():
    print("Using all the data for finetuning the model.")
    startime = time.time()
    param_dist = {
        'max_depth': scipy.stats.randint(1, 20),  # (random integer from 1 to 20)
        'learning_rate': scipy.stats.expon(scale=0.06),  # (random double from an exponential with scale 0.06)
        'n_estimators': scipy.stats.randint(100, 600),  # (random integer from 10 to 600)
    }
    model_findEstimator = xgb.XGBClassifier(verbosity=1,
                                            objective='binary:logistic', booster='gbtree', n_jobs=8,
                                            min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                            colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                            scale_pos_weight=1, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    eval_set = [(test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
    ransearch1 = RandomizedSearchCV(estimator=model_findEstimator, param_distributions=param_dist,
                                    scoring='roc_auc', n_jobs=8, iid=False, cv=5, n_iter=20)
    ransearch1.fit(train_preprocessed[processed_features], train_preprocessed['binarized_label'],
                   early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
    pickle.dump(ransearch1, open("model/paper/notBalancedTraining_randomsearch.pickle.dat", "wb"))
    print("gridsearch score:", ransearch1.grid_scores_)
    print("best parameters:", ransearch1.best_params_)
    print("best score: ", ransearch1.best_score_)
    print("Time: ", time.time() - startime)
    return ransearch1


# def randomsearch_sdCV():
#     print("Using all the data for finetuning the model.")
#     test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
#     search_models = {}
#     test_preprocessed = test_preprocessed.sample(frac=1).reset_index(drop=True)
#     for n_iter in range(10):
#         max_depth = scipy.stats.randint(1, 20).rvs()
#         learning_rate = scipy.stats.expon(scale=0.06).rvs()
#         n_estimators = scipy.stats.randint(100, 600).rvs()
#         performance_this_parameter = []
#         folds = np.array_split(test_preprocessed, 3)
#         startime = time.time()
#         print("Start training...")
#         for fold_nr, test_preprocessed_thisfold in enumerate(folds):
#             model_parameter_thisfold = xgb.XGBClassifier(max_depth=max_depth,
#                                                          learning_rate=learning_rate, n_estimators=n_estimators,
#                                                          verbosity=2,
#                                                          objective='binary:logistic', booster='gbtree', n_jobs=8,
#                                                          min_child_weight=1, max_delta_step=0, subsample=1,
#                                                          colsample_bytree=1,
#                                                          colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
#                                                          reg_lambda=1,
#                                                          scale_pos_weight=1, base_score=0.5, random_state=0)
#             train_this_fold = pd.concat(folds[:fold_nr] + folds[fold_nr + 1:] + [train_preprocessed], axis=0)
#             print("[CV%d] Training:"%fold_nr)
#             model_parameter_thisfold.fit(train_this_fold[processed_features], train_this_fold['binarized_label'])
#             print("     Time: ", time.time() - startime)
#             test_preprocessed_thisfold['probability'] = model_parameter_thisfold.predict_proba(
#                 test_preprocessed_thisfold[processed_features])[:, 1]
#             performance_fold = roc_auc_score(test_preprocessed_thisfold['binarized_label'],
#                                              test_preprocessed_thisfold['probability'])
#             performance_this_parameter.append(performance_fold)
#         model_name = "model%d" % n_iter
#         search_models[model_name] = {'Parameters': {'max_depth': max_depth, 'learning_rate': learning_rate,
#                                                     "n_estimators": n_estimators},
#                                      'Performance': performance_this_parameter}
#         print("Parameters:\n", 'max_depth', max_depth, 'learning_rate', learning_rate, "n_estimators", n_estimators)
#         print("Performances:", performance_this_parameter)
#     return search_models

#
# def gridsearch():
#     # gridsearch
#     from sklearn.model_selection import GridSearchCV
#     negative_num = float(train_preprocessed[train_preprocessed['binarized_label'] == 0].shape[0])
#     positive_num = float(train_preprocessed[train_preprocessed['binarized_label'] == 1].shape[0])
#     param_dist = {'scale_pos_weight': [1, negative_num / positive_num, positive_num / negative_num]}
#     model_findEstimator = xgb.XGBClassifier(verbosity=1, learning_rate=0.05, max_depth=14, n_estimators=553,
#                                             objective='binary:logistic', booster='gbtree', n_jobs=8,
#                                             min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
#                                             colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
#                                             base_score=0.5, random_state=0)
#     test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
#     eval_set = [(train_preprocessed[processed_features], train_preprocessed['binarized_label'], 'train'),
#                 (test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
#     gsearch = GridSearchCV(estimator=model_findEstimator, param_grid=param_dist,
#                            scoring='roc_auc', n_jobs=8, iid=False, cv=5)
#     gsearch.fit(train_preprocessed[processed_features], train_preprocessed['binarized_label'],
#                 early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
#     pickle.dump(gsearch, open("model/paper/notBalancedTraining_gridsearch.pickle.dat", "wb"))
#     print("gridsearch score:", gsearch.grid_scores_)
#     print("best parameters:", gsearch.best_params_)
#     print("best score: ", gsearch.best_score_)
#     return gsearch


# def randomsearch_downsampling():
#     startime = time.time()
#     param_dist = {
#         'max_depth': scipy.stats.randint(1, 20),  # (random integer from 1 to 20)
#         'learning_rate': scipy.stats.expon(scale=0.06),  # (random double from an exponential with scale 0.06)
#         'n_estimators': scipy.stats.randint(100, 600),  # (random integer from 10 to 600)
#     }
#     model_findEstimator = xgb.XGBClassifier(verbosity=1,
#                                             objective='binary:logistic', booster='gbtree', n_jobs=8,
#                                             min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
#                                             colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
#                                             scale_pos_weight=1, base_score=0.5, random_state=0)
#     test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features_balanced)
#     eval_set = [(test_preprocessed[processed_features_balanced], test_preprocessed['binarized_label'], 'test')]
#     ransearch1 = RandomizedSearchCV(estimator=model_findEstimator, param_distributions=param_dist,
#                                     scoring='roc_auc', n_jobs=8, iid=False, cv=5, verbose=1, n_iter=20)
#     ransearch1.fit(train_balanced_preprocessed[processed_features_balanced],
#                    train_balanced_preprocessed['binarized_label'],
#                    early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
#     pickle.dump(ransearch1, open("model/paper/downsampling_randomsearch.pickle.dat", "wb"))
#     print("gridsearch score:", ransearch1.grid_scores_)
#     print("best parameters:", ransearch1.best_params_)
#     print("best score: ", ransearch1.best_score_)
#     print("Time: ", time.time() - startime)
#     return ransearch1


# randomsearch_downsampling()


def train_model():
    best_model = xgb.XGBClassifier(learning_rate=0.095782639391349025, max_depth=17, n_estimators=451,
                                   verbosity=1, objective='binary:logistic', booster='gbtree', n_jobs=8,
                                   min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                   colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    eval_set = [(test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
    print("=====================================Started training...=============================================")
    best_model.fit(train_preprocessed[processed_features], train_preprocessed['binarized_label'],
                   early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
    print("=====================================Started training...=============================================")
    pickle.dump(best_model, open('model/different_training/best_model.pickle.dat', 'wb'))
    return best_model


# def down_sampling_best_model():
#     best_model = xgb.XGBClassifier(learning_rate=0.31163736581056845, max_depth=15, n_estimators=288,
#                                    verbosity=1, objective='binary:logistic', booster='gbtree', n_jobs=8,
#                                    min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
#                                    colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
#                                    scale_pos_weight=1, base_score=0.5, random_state=0)
#     test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features_balanced)
#     eval_set = [(test_preprocessed[processed_features_balanced], test_preprocessed['binarized_label'], 'test')]
#     print("=====================================Started training...=============================================")
#     best_model.fit(train_balanced_preprocessed[processed_features_balanced],
#                    train_balanced_preprocessed['binarized_label'],
#                    early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
#     print("=====================================Started training...=============================================")
#     pickle.dump(best_model, open('model/different_training/best_model_downsampling.pickle.dat', 'wb'))
#     return best_model


def best_model_continue_training():
    best_model = pickle.load(open('model/different_training/best_model.pickle.dat', 'rb'))
    continue_model = xgb.XGBClassifier(learning_rate=0.31163736581056845, max_depth=15, n_estimators=288,
                                   verbosity=1, objective='binary:logistic', booster='gbtree', n_jobs=8,
                                   min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                   colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    print("=====================================Started training...=============================================")
    for consequence in train_preprocessed['Consequence'].unique():
        data_consequence = train_preprocessed[train_preprocessed['Consequence'] == consequence]
        eval_set_consequence = [(test_preprocessed[processed_features][test_preprocessed['Consequence'] == consequence],
                                 test_preprocessed['binarized_label'][test_preprocessed['Consequence'] == consequence],
                                 'test')]
        continue_model.fit(data_consequence[processed_features],
                       data_consequence['binarized_label'],
                       early_stopping_rounds=15, eval_metric=["auc"],
                       eval_set=eval_set_consequence, verbose=True,
                           xgb_model=best_model.get_booster())
        pickle.dump(best_model, open('model/best_model_continue/best_model_%s.pickle.dat'%consequence, 'wb'))
    return best_model


def hq_data_finetune():
    train_hq = train_preprocessed[train_preprocessed['sample_weight'] == 1]
    print("Using all the data for finetuning the model.")
    startime = time.time()
    param_dist = {
        'max_depth': scipy.stats.randint(1, 20),  # (random integer from 1 to 20)
        'learning_rate': scipy.stats.expon(scale=0.06),  # (random double from an exponential with scale 0.06)
        'n_estimators': scipy.stats.randint(100, 600),  # (random integer from 10 to 600)
    }
    model_findEstimator = xgb.XGBClassifier(verbosity=1,
                                            objective='binary:logistic', booster='gbtree', n_jobs=8,
                                            min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                            colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                            scale_pos_weight=1, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    eval_set = [(test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
    ransearch1 = RandomizedSearchCV(estimator=model_findEstimator, param_distributions=param_dist,
                                    scoring='roc_auc', n_jobs=8, iid=False, cv=5, n_iter=20)
    ransearch1.fit(train_hq[processed_features], train_hq['binarized_label'],
                   early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True)
    pickle.dump(ransearch1, open("model/paper/xgb_hqData_randomsearch.pickle.dat", "wb"))
    print("gridsearch score:", ransearch1.grid_scores_)
    print("best parameters:", ransearch1.best_params_)
    print("best score: ", ransearch1.best_score_)
    print("Time: ", time.time() - startime)


def weightedSample_finetune():
    print("Using all the data for finetuning the model.")
    startime = time.time()
    param_dist = {
        'max_depth': scipy.stats.randint(1, 20),  # (random integer from 1 to 20)
        'learning_rate': scipy.stats.expon(scale=0.06),  # (random double from an exponential with scale 0.06)
        'n_estimators': scipy.stats.randint(100, 600),  # (random integer from 10 to 600)
    }
    model_findEstimator = xgb.XGBClassifier(verbosity=1,
                                            objective='binary:logistic', booster='gbtree', n_jobs=8,
                                            min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                            colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                                            scale_pos_weight=1, base_score=0.5, random_state=0)
    test_preprocessed = preprocess(impute(test), isTrain=False, model_features=processed_features)
    eval_set = [(test_preprocessed[processed_features], test_preprocessed['binarized_label'], 'test')]
    ransearch1 = RandomizedSearchCV(estimator=model_findEstimator, param_distributions=param_dist,
                                    scoring='roc_auc', n_jobs=8, iid=False, cv=5, n_iter=20)
    ransearch1.fit(train_preprocessed[processed_features], train_preprocessed['binarized_label'],
                   early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set, verbose=True,
                   sample_weight=train_preprocessed['sample_weight'])
    pickle.dump(ransearch1, open("model/paper/xgb_weightedSample_randomsearch.pickle.dat", "wb"))
    print("gridsearch score:", ransearch1.grid_scores_)
    print("best parameters:", ransearch1.best_params_)
    print("best score: ", ransearch1.best_score_)
    print("Time: ", time.time() - startime)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, dest="mode")
    args = parser.parse_args()
    return args


# if __name__ == '__main__':
#     mode = parse().mode
#     print(mode)
#     sys.stdout = open("out/log_%s.txt" % mode, "w")
#     if mode == "downsampling_randomsearch":
#         # _ = randomsearch_downsampling()
#         pass
#     elif mode == "downsampling_randomsearch":
#         pass
#     elif mode == "randomsearch":
#         _ = randomsearch()
#     # elif mode == "gridsearch":
#     #     _ = gridsearch()
#     elif mode == "basic":
#         _ = basic_model()
#     elif mode == "train_best":
#         _ = train_model()
#     elif mode == 'weightedSample':
#         weightedSample_finetune()
#     elif mode == 'hq_data':
#         hq_data_finetune()
#     # elif mode == "train_best_downsampling":
#     #     _ = down_sampling_best_model()
#     # elif mode == "randomsearch_sdCV":
#     #     _ = randomsearch_sdCV()
#     # elif mode == "best_model_continue":
#     #     _ = best_model_continue_training()
#     else:
#         raise IOError("Please select an available mode")
#     sys.stdout.close()
