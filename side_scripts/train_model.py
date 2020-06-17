#!/usr/bin/env python3

from impute_preprocess import preprocess, impute, cadd_vars
import pandas as pd
import xgboost as xgb
import scipy
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV

data = pd.read_csv('/groups/umcg-gcc/tmp01/umcg-rsietsma/capicev2/data/dislipid_panel.txt.gz',
                   sep='\t', compression='gzip', low_memory=False)

train, test = train_test_split(data, test_size=0.2, random_state=4)
train_preprocessed = preprocess(impute(train), isTrain=True)
processed_features = []
for col in train_preprocessed.columns:
    for feat in cadd_vars:
        if col == feat or col.startswith(feat):
            if col not in processed_features:
                processed_features.append(col)

param_dist = {
    'max_depth': scipy.stats.randint(1, 20),  # (random integer from 1 to 20)
    'learning_rate': scipy.stats.expon(scale=0.06),
    # (random double from an exponential with scale 0.06)
    'n_estimators': scipy.stats.randint(100, 600),
    # (random integer from 10 to 600)
}
print('Preparing the estimator model', flush=True)
model_findEstimator = xgb.XGBClassifier(verbosity=1,
                                        objective='binary:logistic',
                                        booster='gbtree', n_jobs=8,
                                        min_child_weight=1, max_delta_step=0,
                                        subsample=1, colsample_bytree=1,
                                        colsample_bylevel=1, colsample_bynode=1,
                                        reg_alpha=0, reg_lambda=1,
                                        scale_pos_weight=1, base_score=0.5,
                                        random_state=0)
test_preprocessed = preprocess(impute(test), isTrain=False,
                               model_features=processed_features)
eval_set = [(test_preprocessed[processed_features],
             test_preprocessed['binarized_label'], 'test')]
print('Random search initializing', flush=True)
ransearch1 = RandomizedSearchCV(estimator=model_findEstimator,
                                param_distributions=param_dist,
                                scoring='roc_auc', n_jobs=8, iid=False, cv=5,
                                n_iter=20)
print('Random search starting, please hold.', flush=True)
ransearch1.fit(train_preprocessed[processed_features],
               train_preprocessed['binarized_label'],
               early_stopping_rounds=15, eval_metric=["auc"], eval_set=eval_set,
               verbose=True,
               sample_weight=train_preprocessed['sample_weight'])
pickle.dump(ransearch1,
            open("/groups/umcg-gcc/tmp01/umcg-rsietsma/capicev2/xgb_weightedSample_randomsearch_dislipid.pickle.dat",
                 "wb"))
with open('/groups/umcg-gcc/tmp01/umcg-rsietsma/capicev2/dislipid_best_params.txt', 'w+') as export_1:
    export_1.write(ransearch1.best_params_)

with open('/groups/umcg-gcc/tmp01/umcg-rsietsma/capicev2/dislipid_grid_score.txt', 'w+') as export_2:
    export_2.write(ransearch1.grid_scores_)
