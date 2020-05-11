import pickle
import pandas as pd
from impute_preprocess import impute, preprocess
import gzip
import time
import os
from pathlib import Path
import argparse
from datetime import datetime
import psutil


class CalculateCapiceScores:
    """
    Main class of the script to call all the various logger class functions and
    will process the iterative chunking processing of the CADD file.
    """
    def __init__(self, logger_instance, filepath, model_loc, output_loc,
                 batch_size):
        self.log = logger_instance
        self.filepath = filepath
        self.titles = None
        self.get_header()
        self.model = None
        self.load_model(model_loc)
        self.model_feats = None
        self.not_done = True
        self.batch_size = batch_size
        self.output_loc = output_loc
        self.utilities = Utilities()
        self.utilities.check_if_dir_exists(output_loc)

    def get_header(self):
        if not self.titles:
            with gzip.open(self.filepath, "r") as f:
                while True:
                    line = f.readline()
                    if line.decode('utf-8').startswith("#Chr"):
                        print(line.decode('utf-8'))
                        self.titles = line.decode('utf-8').strip().split("\t")
                        self.log.log(f'Title found: {self.titles}')
                        break
                    else:
                        continue

    def calculate_save_capice_score(self, batch_size, skip_rows):
        variants_df = pd.read_csv(self.filepath, sep='\t', skip_rows=skip_rows,
                                  nrows=batch_size, names=self.titles,
                                  comment='#', compression='gzip')
        if variants_df.shape[0] < batch_size:
            self.not_done = False
            self.log.log(f'Processing the last entries!'
                         f' Total variants processed: '
                         f'{skip_rows + variants_df.shape[0]}.')

        variants_df_preprocessed = preprocess(impute(variants_df),
                                              isTrain=False,
                                              model_features=self.model_feats)
        variants_df['prediction'] = self.model.predict_proba(
            variants_df_preprocessed[self.model_feats])[:, 1]
        if variants_df['prediction'].isnull().any():
            self.log.log(f'NaN encounter in chunk: {skip_rows}-{batch_size}!')
        for unique_chr in variants_df['#Chr'].unique():
            subset_variants_df = variants_df[variants_df['#Chr'] == unique_chr]
            output_dir = os.path.join(self.output_loc, f'chr{unique_chr}')
            self.utilities.check_if_dir_exists(output_dir)
            min_pos = subset_variants_df['Pos'].min()
            max_pos = subset_variants_df['Pos'].max()
            chunk = f'{unique_chr}:{min_pos}-{max_pos}'
            output_filename = f'whole_genome_SNVs_{chunk}'
            final_destination = os.path.join(output_dir, output_filename)
            self.utilities.check_if_file_exists(final_destination)
            with open(final_destination, 'a') as f:
                variants_df[['#Chr', 'Pos', 'Ref', 'Alt', 'GeneID',
                             'CCDS', 'FeatureID',
                             'prediction']].to_csv(f, sep="\t", index=False,
                                                   header=None)

    def load_model(self, model_loc):
        self.model = pickle.load(open(model_loc, "rb")).best_estimator_
        self.model_feats = self.model.get_booster().feature_names

    def calc_capice(self):
        start = 0
        first_iter = True
        start_time = time.time()
        while self.not_done:
            reset_timer = time.time()
            if reset_timer - start_time > (60 * 5) or first_iter:
                # Seconds times the amount of minutes.
                curr_time = time.time()
                time_difference = curr_time - start_time
                minutes, seconds = divmod(time_difference, 60)
                hours, minutes = divmod(minutes, 60)
                self.log.log(f'Still going for {hours} hours,'
                             f' {minutes} minutes and {seconds} seconds.')
                self.log.log(f'Memory usage: {self.log.get_ram_usage} MB.')
                self.log.log(f'Currently working on rows {start} -'
                             f' {start + self.batch_size}.')
                reset_timer = time.time()
            self.calculate_save_capice_score(self.batch_size, start)
            if first_iter:
                start += self.batch_size + 1
                first_iter = False
                exit()
            else:
                start += self.batch_size


class ArgumentSupporter:
    """
    Class to handle the given command line input.
    Type python3 step9_pre_computed_scores_snv.py --help for more details.
    """

    def __init__(self):
        parser = self._create_argument_parser()
        self.arguments = parser.parse_args()

    @staticmethod
    def _create_argument_parser():
        parser = argparse.ArgumentParser(
            prog="step9_pre_computed_scores_snv.py",
            description="Python script to calculate Pre-computed"
                        " scores for CAPICE for every given CADD variant.")
        required = parser.add_argument_group("Required arguments")
        optional = parser.add_argument_group("Optional arguments")

        required.add_argument('-f',
                              '--file',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The location of the CADD'
                                   ' annotated SNV file.')

        required.add_argument('-m',
                              '--model',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The location of the CAPICE'
                                   ' model pickled file.')

        required.add_argument('-o',
                              '--output',
                              nargs=1,
                              type=str,
                              required=True,
                              help='The output directory to put the processed'
                                   'CADD variants in.')

        optional.add_argument('-s',
                              '--batchsize',
                              nargs=1,
                              type=int,
                              default=1000,
                              required=False,
                              help='The chunksize for the script to'
                                   ' read the gzipped archive. (Default: 1000)')
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


class Logger:
    """
    Class to make a logfile on the progress being made.
    """
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.output_dir = None
        self.logfile = None
        self.utilities = Utilities()
        self.check_if_dir_exist()
        self.check_if_log_file_exist()

    def check_if_dir_exist(self):
        output_dir = os.path.join(self.root_dir, 'log_output')
        self.utilities.check_if_dir_exists(output_dir)
        self.output_dir = output_dir

    def check_if_log_file_exist(self):
        log_file_name = f'{datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")}' \
                        f'_logfile.txt'
        joined_path = os.path.join(self.output_dir, log_file_name)
        self.utilities.check_if_file_exists(joined_path)
        self.logfile = joined_path

    def get_output_dir(self):
        return self.output_dir

    @staticmethod
    def get_ram_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1000000  # Megabytes

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S_%f")
        timed_message = f'[{timestamp}]: {message}'
        with open(self.logfile, 'a') as logfile:
            logfile.write(timed_message)


class Utilities:
    @staticmethod
    def check_if_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def check_if_file_exists(filepath):
        if not os.path.isfile(filepath):
            new_file = open(filepath, 'a+')
            new_file.close()


def main():
    """
    Main method of the script. Will call the various classes.
    """
    arguments = ArgumentSupporter()
    logger = Logger()
    cadd_loc = arguments.get_argument('file')
    model_loc = arguments.get_argument('model')
    output_loc = arguments.get_argument('output')
    batch_size = arguments.get_argument('batchsize')
    if isinstance(cadd_loc, list):
        cadd_loc = str(cadd_loc[0])
    if isinstance(model_loc, list):
        model_loc = str(model_loc[0])
    if isinstance(output_loc, list):
        output_loc = str(output_loc[0])
    if isinstance(batch_size, list):
        batch_size = int(batch_size[0])
    logger.log(f'CADD file location: {cadd_loc}')
    logger.log(f'Model file location: {model_loc}')
    logger.log(f'Output directory: {output_loc}')
    logger.log(f'Batch size set to: {batch_size}')
    precompute_capice = CalculateCapiceScores(logger_instance=logger,
                                              filepath=cadd_loc,
                                              model_loc=model_loc,
                                              output_loc=output_loc,
                                              batch_size=batch_size)
    precompute_capice.calc_capice()


if __name__ == '__main__':
    main()
