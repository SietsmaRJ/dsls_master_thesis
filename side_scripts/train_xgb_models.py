import argparse
import pickle
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from impute_preprocess import impute, preprocess, cadd_vars
import gzip
import pandas as pd
import numpy as np


class ArgumentSupporter:
    """
    Class to handle the given command line input.
    Type python3 PreComputeCapice.py --help for more details.
    """

    def __init__(self):
        parser = self._create_argument_parser()
        self.arguments = parser.parse_args()

    @staticmethod
    def _create_argument_parser():
        parser = argparse.ArgumentParser(
            prog="train_xgb_models.py",
            description="Python script to convert RandomSearchCV optimal"
                        " parameters to a pickled XGBClassifier model.")

        required = parser.add_argument_group("Required arguments")

        required.add_argument('-i',
                              '--input',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The json of parameters.')

        required.add_argument('-o',
                              '--output',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The location of the'
                                   ' XGBClassifier pickled output.')

        required.add_argument('-f',
                              '--file',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The location of the training database.')

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


class TrainModel:
    def __init__(self, params, output_loc, training_file):
        self.params = params
        self.output_loc = output_loc
        self.train = None
        self.eval_set = None
        self.processed_feats = []
        self.prepare_input_data(training_file)

    def train_model(self):
        model = xgb.XGBClassifier(verbosity=1,
                                  objective='binary:logistic',
                                  booster='gbtree', n_jobs=8,
                                  min_child_weight=1, max_delta_step=0,
                                  subsample=1, colsample_bytree=1,
                                  colsample_bylevel=1, colsample_bynode=1,
                                  reg_alpha=0, reg_lambda=1,
                                  scale_pos_weight=1, base_score=0.5,
                                  random_state=0,
                                  max_depth=self.params['max_depth'],
                                  learning_rate=self.params['learning_rate'],
                                  n_estimators=self.params['n_estimators'])
        model.fit(self.train[self.processed_feats],
                  self.train['binarized_label'],
                  early_stopping_rounds=15,
                  eval_metric='auc',
                  eval_set=self.eval_set,
                  verbose=True,
                  sample_weight=self.train['sample_weight'])
        pickle.dump(model, open(self.output_loc, 'wb'))

    def prepare_input_data(self, training_file):
        skip_rows = 0
        for line in gzip.open(training_file):
            if line.decode().startswith("##"):
                skip_rows = 1
            break
        data = pd.read_csv(training_file, compression='gzip',
                           skiprows=skip_rows, sep='\t', low_memory=False)
        train, test = train_test_split(data, test_size=0.2, random_state=4)
        self.train = preprocess(impute(train), isTrain=True)
        for col in self.train:
            for feat in cadd_vars:
                if col == feat or col.startswith(feat):
                    if col not in self.processed_feats:
                        self.processed_feats.append(col)
        test_preprocessed = preprocess(impute(test), isTrain=False,
                                       model_features=self.processed_feats)
        self.eval_set = [(test_preprocessed[self.processed_feats],
                          test_preprocessed['binarized_label'], 'test')]


def main():
    arguments = ArgumentSupporter()
    input_json = arguments.get_argument('input')
    if isinstance(input_json, list):
        input_json = str(input_json[0])
    output_loc = arguments.get_argument('output')
    if isinstance(output_loc, list):
        output_loc = str(output_loc[0])
    training_file = arguments.get_argument('file')
    if isinstance(training_file, list):
        training_file = str(training_file[0])
    with open(input_json) as input_params:
        loaded_input_params = json.load(input_params)
    train_model = TrainModel(loaded_input_params, output_loc, training_file)
    train_model.train_model()


if __name__ == '__main__':
    main()
