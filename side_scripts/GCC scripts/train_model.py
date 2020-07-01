#!/usr/bin/env python3

from impute_preprocess import ImputePreprocess
import pandas as pd
import xgboost as xgb
import scipy
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV


class Train:
    def __init__(self, data_loc, output_loc, verbose, default):
        self.verbose = verbose
        self._printf("Input location: " + data_loc)
        self._printf("Output location: " + output_loc)
        self.default = default
        self._printf("Default set to: " + str(self.default))
        self.data = pd.read_csv(data_loc, sep='\t',
                                compression='gzip',
                                low_memory=False,
                                verbose=self.verbose)
        self.train_set = None
        self.test_set = None
        self.processed_features = []
        self.output_loc = output_loc
        self.ransearch_output = os.path.join(self.output_loc,
                                             'xgb_ransearch.pickle.dat')
        self.optimal_model = os.path.join(self.output_loc,
                                          'xgb_optimal_model.pickle.dat')
        self.ip = ImputePreprocess(self.verbose)
        self.cadd_vars = self.ip.get_cadd_vars()
        self._prepare_data()

    def _prepare_data(self):
        train, test = train_test_split(self.data, test_size=0.2, random_state=4)
        self.train_set = self.ip.preprocess(self.ip.impute(train), isTrain=True)
        for col in self.train_set.columns:
            for feat in self.cadd_vars:
                if col == feat or col.startswith(feat):
                    if col not in self.processed_features:
                        self.processed_features.append(col)
        self.test_set = self.ip.preprocess(
            self.ip.impute(test), isTrain=False,
            model_features=self.processed_features)

    def train(self):
        param_dist = {
            'max_depth': scipy.stats.randint(1, 20),
            # (random integer from 1 to 20)
            'learning_rate': scipy.stats.expon(scale=0.06),
            # (random double from an exponential with scale 0.06)
            'n_estimators': scipy.stats.randint(100, 600),
            # (random integer from 10 to 600)
        }
        if self.verbose:
            verbosity = 1
        else:
            verbosity = 0
        self._printf('Preparing the estimator model', flush=True)
        if self.default:
            model_estimator = xgb.XGBClassifier(
                verbosity=verbosity,
                objective='binary:logistic',
                booster='gbtree', n_jobs=8,
                min_child_weight=1,
                max_delta_step=0,
                subsample=1,
                colsample_bytree=1,
                colsample_bylevel=1,
                colsample_bynode=1,
                reg_alpha=0, reg_lambda=1,
                scale_pos_weight=1,
                base_score=0.5,
                random_state=0,
                learning_rate=0.10495845238185281,
                n_estimators=422,
                max_depth=15
            )
            ransearch1 = model_estimator
        else:
            model_estimator = xgb.XGBClassifier(verbosity=verbosity,
                                                objective='binary:logistic',
                                                booster='gbtree', n_jobs=8,
                                                min_child_weight=1,
                                                max_delta_step=0,
                                                subsample=1, colsample_bytree=1,
                                                colsample_bylevel=1,
                                                colsample_bynode=1,
                                                reg_alpha=0, reg_lambda=1,
                                                scale_pos_weight=1,
                                                base_score=0.5,
                                                random_state=0)
            ransearch1 = RandomizedSearchCV(estimator=model_estimator,
                                            param_distributions=param_dist,
                                            scoring='roc_auc', n_jobs=8,
                                            cv=5,
                                            n_iter=20,
                                            verbose=verbosity)
        eval_set = [(self.test_set[self.processed_features],
                     self.test_set['binarized_label'], 'test')]
        self._printf('Random search initializing', flush=True)

        self._printf('Random search starting, please hold.', flush=True)
        ransearch1.fit(self.train_set[self.processed_features],
                       self.train_set['binarized_label'],
                       early_stopping_rounds=15,
                       eval_metric=["auc"],
                       eval_set=eval_set,
                       verbose=True,
                       sample_weight=self.train_set['sample_weight'])
        pickle.dump(ransearch1, open(self.ransearch_output, "wb"))

        if not self.default:
            pickle.dump(ransearch1.best_estimator_, open(self.optimal_model,
                                                         'wb'))

    def _printf(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class ArgumentSupporter:
    def __init__(self):
        parser = self._create_argument_parser()
        self.arguments = parser.parse_args()

    @staticmethod
    def _create_argument_parser():
        parser = argparse.ArgumentParser(
            prog="train_model.py",
            description="Python script to train new models using XGboost.")
        required = parser.add_argument_group("Required arguments")
        optional = parser.add_argument_group("Optional arguments")

        required.add_argument('-f',
                              '--file',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The location of the training file.\n'
                                   '(Must be a gzipped TSV file without index)')

        required.add_argument('-o',
                              '--output',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The output directory to put the models in.')

        optional.add_argument('-d',
                              '--default',
                              action='store_true',
                              help='Use the python3.6 model hyperparameters.')

        optional.add_argument('-v',
                              '--verbose',
                              action='store_true',
                              help='Prints messages if called.')

        return parser

    def get_argument(self, argument_key):
        """
        Method to get a command line argument.
        :param argument_key: Command line argument.
        :return: List or string.
        """
        if self.arguments is not None and argument_key in self.arguments:
            value = getattr(self.arguments, argument_key)
        else:
            value = None

        return value


def main():
    arguments = ArgumentSupporter()
    input_loc = arguments.get_argument('file')
    if isinstance(input_loc, list):
        input_loc = str(input_loc[0])
    output_loc = arguments.get_argument('output')
    if isinstance(output_loc, list):
        output_loc = str(output_loc[0])
    verbose = arguments.get_argument('verbose')
    default = arguments.get_argument('default')
    train = Train(input_loc, output_loc, verbose, default)
    train.train()


if __name__ == '__main__':
    main()
