import os
import pandas as pd
import numpy as np
from sklearn import metrics
import time
from multiprocessing import Process, Manager
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score,\
    f1_score


class PerGene:
    def __init__(self):
        os.chdir('/home/rjsietsma/Documents/School/Master_DSLS/'
                 'Final_Thesis/Initial_Data_exploration')
        self.execute()

    def execute(self):
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

        # var_comb = var_comb.drop(['#Chrom', 'Pos', 'Ref', 'Alt'], axis=1)
        var_comb = var_comb.drop(['Ref', 'Alt'], axis=1)

        var_comb['prediction'] = var_comb['prediction'].replace({"Pathogenic":1,"Neutral":0})

        var_comb['label'] = var_comb['label'].replace({"LP/P":1,"LB/B":0})

        overview = pd.DataFrame(columns=['Gene', 'AUC', 'FPR', 'Precision',
                                         'Recall/TPR', 'F-score', 'N_benign',
                                                          'N_malign', 'm_cat',
                                                          'n_snvs'])
        var_comb.to_csv('var_comb.csv')

        time_before_whileloop = time.time()
        total_genes = var_comb['GeneName'].unique().shape[0]
        done_genes = 0

        time_forloop_started = time.time()
        for genename in var_comb['GeneName'].unique():
            time_in_whileloop = time.time()
            if time_in_whileloop - time_before_whileloop > 10:
                print(f"I'm still running, I've been running for: "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[0])}"
                      f" minutes and "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[1])}"
                      f" seconds.")
                print(f'\n'
                      f'Done {round(done_genes/total_genes*100, ndigits=2)} %. \n')
                time_before_whileloop = time.time()
            subset_data = var_comb[var_comb['GeneName'] == genename]
            exon_subset = exons[exons['gene'] == genename]
            n_snvs = subset_data['GeneName'].count()
            y_true = np.array(subset_data['label'])
            y_pred = np.array(subset_data['probabilities'])
            has_m_cat = None
            if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
                auc = metrics.roc_auc_score(y_true, y_pred)
                precision = metrics.precision_score(y_true, y_pred)
                recall = metrics.recall_score(y_true, y_pred, zero_division=0)
                fpr = 1 - recall
                f_score = metrics.f1_score(y_true, y_pred)
                has_m_cat = 1
            else:
                auc = np.NaN
                precision = np.NaN
                recall = np.NaN
                fpr = np.NaN
                f_score = np.NaN
                has_m_cat = 0
            n_benign = subset_data[subset_data['label'] == 0]['label'].count()
            n_malign = subset_data[subset_data['label'] == 1]['label'].count()

            row = pd.DataFrame({'Gene': [genename], 'AUC': [auc], 'FPR': [fpr],
                                'Precision': [precision], 'Recall/TPR': [recall],
                                'F-score': [f_score], 'N_benign': [n_benign],
                                'N_malign': [n_malign], 'm_cat': [has_m_cat],
                                     'n_snvs':[n_snvs]})
            overview = overview.append(row, ignore_index=True)
            done_genes += 1

        overview.to_csv("auc_results.csv")


class PerSNV:
    def __init__(self):
        # Change working directory to the data files containing folder
        os.chdir("/home/rjsietsma/Documents/School/"
                 "Master_DSLS/Final_Thesis/Initial_Data_exploration")
        self.execute()

    def execute(self):
        test_dataset = pd.read_csv("test_results.txt", sep='\t', header=0)
        test_dataset['gene'] = np.NaN
        genes = pd.read_csv('ensembl_compressed.csv', header=0, index_col=0)
        time_before_whileloop = time.time()
        total_rows = genes.shape[0]
        done_rows = 0
        time_forloop_started = time.time()
        for gene in genes.iterrows():
            time_in_whileloop = time.time()
            if time_in_whileloop - time_before_whileloop > 10:
                print(f"I'm still running, I've been running for: "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[0])}"
                      f" minutes and "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[1])}"
                      f" seconds.")
                print(f'Done {round(done_rows/total_rows*100, ndigits=2)} %.')
                time_before_whileloop = time.time()
            gene = gene[1]
            subset_chr = test_dataset[test_dataset['#Chrom'] == gene['chr']]
            subset_min = subset_chr[subset_chr['Pos'] >= gene['start']]
            full_subset = subset_min[subset_min['Pos'] <= gene['stop']]
            if full_subset.shape[0] > 0:
                test_dataset.loc[full_subset.index, 'gene'] = gene['gene']
            done_rows += 1
        test_dataset.to_csv("test_results.csv")


class ApplyAUCPerGene:
    def __init__(self):
        self._execute()

    def _execute(self):
        auc_data = pd.read_csv('auc_results.csv', header=0, index_col=0)
        test_data_genes = pd.read_csv('test_results_with_genes.csv', header=0,
                                      index_col=0)
        auc_data = auc_data[['Gene', 'AUC']]
        auc_data.columns = ['gene', 'auc']
        auc_data_merged = test_data_genes.merge(auc_data, left_on=['gene'], right_on=['gene'])
        auc_data_merged.to_csv('test_results_genes_auc.csv')


class CompressGenes:
    def __init__(self):
        # Change working directory to the data files containing folder
        os.chdir("/home/rjsietsma/Documents/School/"
                 "Master_DSLS/Final_Thesis/Initial_Data_exploration")
        self._execute()

    def _execute(self):
        better_genes = pd.DataFrame(columns=['chr', 'start', 'stop', 'gene'])
        genes = pd.read_csv('Ensembl_GRCh37_ExonRegions.txt',
                            sep='\t', header=0)
        curr_gene = None
        still_gene = False
        min_curr = None
        max_curr = None
        time_before_whileloop = time.time()
        total_rows = genes.shape[0]
        done_rows = 0
        time_forloop_started = time.time()
        for gene in genes.iterrows():
            time_in_whileloop = time.time()
            if time_in_whileloop - time_before_whileloop > 10:
                print(f"I'm still running, I've been running for: "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[0])}"
                      f" minutes and "
                      f"{round(divmod(time_in_whileloop - time_forloop_started, 60)[1])}"
                      f" seconds.")
                print(f'Done {round(done_rows/total_rows*100, ndigits=2)} %.')
                time_before_whileloop = time.time()
            gene = gene[1]
            if not curr_gene:
                curr_gene = gene['gene']
            if curr_gene == gene['gene']:
                still_gene = True
            else:
                better_genes = better_genes.append(pd.DataFrame(
                    {'chr': [gene['chr']], 'start': [min_curr],
                     'stop': [max_curr], 'gene': [curr_gene]}))
                still_gene = False
                min_curr = None
                max_curr = None
                curr_gene = gene['gene']
            if not min_curr and not max_curr:
                min_curr = gene['start']
                max_curr = gene['end']

            if still_gene:
                min_row = gene['start']
                max_row = gene['end']
                if min_row < min_curr:
                    min_curr = min_row
                elif max_row < min_curr:
                    min_curr = max_row

                if max_row > max_curr:
                    max_curr = max_row
                elif min_row > max_curr:
                    max_curr = min_row
            done_rows += 1

        better_genes.to_csv("ensembl_compressed.csv")


class IntronicExonic:
    def __init__(self):
        self._execute()

    def _execute(self):
        data = pd.read_csv('test_results_genes_auc.csv', header=0, index_col=0)
        data['exonic'] = 0
        full_genes_data = pd.read_csv('Agilent_Exoom_v3_human_g1k_v37_captured.merged.genesplit.bed', sep='\t', header=0)
        reset_timer = time.time()
        total_rows = data.shape[0]
        done_rows = 0
        time_fls = time.time()
        for row in data.iterrows():
            time_ifl = time.time()
            if time_ifl - reset_timer > 10:
                print(f"I'm still running, I've been running for: "
                      f"{round(divmod(time_ifl - time_fls, 60)[0])}"
                      f" minutes and "
                      f"{round(divmod(time_ifl - time_fls, 60)[1])}"
                      f" seconds.")
                print(f'Done {round(done_rows/total_rows*100, ndigits=2)} %.')
                reset_timer = time.time()
            index = row[0]
            row = row[1]
            subset_genes_chr = full_genes_data[full_genes_data['chr'] ==
                                           row['#Chrom']]
            subset_genes_gene = subset_genes_chr[subset_genes_chr['gene'] ==
                                        row['gene']]
            subset_genes_min = subset_genes_gene[subset_genes_gene['start'] <=
                                        row['Pos']]
            subset_genes_max = subset_genes_min[subset_genes_min['end'] >=
                                        row['Pos']]
            if subset_genes_max.shape[0] > 0:
                data.loc[index, 'exonic'] = 1
            done_rows += 1
        data.to_csv('test_results_genes_auc_exonic.csv')


class MultiProcess:
    def __init__(self):
        self.data = pd.read_csv('/home/rjsietsma/Documents/School/'
                                'Master_DSLS/Final_Thesis/'
                                'Initial_Data_exploration/'
                                'full_dataset.csv',
                                header=0)
        self.execute(self.data)

    # Let's define a function that would be called by multiprocessing
    def map_optimal_auc_per_gene(self, dataset, gene,
                                 manager_list=None, ignore_warn=True):
        """
        Method to map a dataset containing the columns 'chr', 'capice',
         'label' and 'gene' to calculate an optimal AUC,
         starting from 0.02 and lowering / increasing from there,
         starting with decreasing.
        Note: method is not case sensitive

        Parameters
        ----------
            dataset: pandas.DataFrame
                A pandas dataframe containing the required columns
                 (being capice, gene and label) to calculate an AUC to.
            gene: str
                Gene name to subset the dataset to.
            manager_list: multiprocessing.Manager().list() (optional)
                For multiprocessing purposes, added the option for Manager list.
            ignore_warn: bool (optional)
                Returns None if set to true, else raises errors.

        Returns
        -------
            optimal: pandas.DataFrame
                A 1 by 8 pandas dataframe containing:
                 (1st column) the chomosome of the gene,
                 (2nd column) the gene name,
                 (3-7 columns) statistics under default cutoff,
                 (8th column) optimal CAPICE cutoff,
                 (9-13 column) statistics under optimal cutoff,
                 (14th column) the total amount of SNVs of that gene,
                 (15th column) the total amount of benign SNVs of that gene and
                 (16th column) the total amount of malignant SNVs of that gene.
                 Keep in mind that the cutoff is greater than for pathogenic.
        """

        # Check if ignore_warn is actually a boolean
        if not isinstance(ignore_warn, bool):
            raise AttributeError('ignore_warn must be a boolean.')

        # Check if input is a pandas dataframe.
        if not isinstance(dataset, pd.DataFrame):
            if not ignore_warn:
                raise AttributeError(
                    f'The input is not a pandas dataframe, instead is:'
                    f' {type(dataset)}')
            else:
                return None

        # Check if required columns are present.
        columns = [c.lower() for c in dataset.columns.tolist()]
        req_labels = ['chr', 'gene', 'capice', 'label']
        for label in req_labels:
            if label not in columns:
                if not ignore_warn:
                    raise AttributeError(f'Label {label} not found in dataset!')
                else:
                    return None

        # Reapply the column names to the columns.lower() version.
        dataset.columns = columns

        # Final check of the column types.
        req_dtypes = {'chr': np.object,
                      'label': np.int64,
                      'capice': np.float64,
                      'gene': np.object}
        for i, dtype in enumerate(dataset[req_labels].dtypes):
            label = req_labels[i]
            if dtype != req_dtypes[label]:
                if not ignore_warn:
                    raise AttributeError(f'Label {label} is an incorrect dtype,'
                                         f' expected: {req_dtypes[label]},'
                                         f' but got: {dtype}.')
                else:
                    return None

        # Finally, checking if there's only 1 gene.
        if not isinstance(gene, str):
            if not ignore_warn:
                raise AttributeError('Gene is not a string!')
            else:
                return None

        # Sub setting the data.
        dataset = dataset[dataset['gene'] == gene]

        # Checking if y_true (label) actually has multiple entries
        if dataset['label'].unique().size < 2:
            if not ignore_warn:
                raise ValueError(
                    'y_true (label) must contain at least 2 unique classes!')
            else:
                return None

        # Checking to apply the chromosome for the gene.
        exclude_genes = ['Y-RNA', 'snoU13', 'Y_RNA',
                         'SNORA43', 'SNORA73', 'U3', 'U6']
        par_gene = 'SHOX'
        if dataset['chr'].unique().shape[0] > 1:
            if gene not in exclude_genes:
                if gene == par_gene:
                    chro = 'X'
                else:
                    chro = str(dataset['chr'].value_counts().sort_values(
                        ascending=False)[0])
            else:
                chro = np.NaN
        else:
            chro = str(dataset['chr'].unique()[0])

        # Now let the fun begin.
        # Default values.
        threshold_default = 0.02
        auc_default = None
        f1_default = None
        recall_default = None
        fpr_default = None
        specificity_default = None

        # Optimal values.
        final_threshold = None
        auc_optimal = None
        f1_optimal = None
        recall_optimal = None
        fpr_optimal = None
        specificity_optimal = None

        # While loop specific settings.
        adapted_threshold = None
        stepsize = 0.001
        max_attempts = 3
        check_lower_optimum = True
        lower_attempt = 0
        check_upper_optimum = False
        upper_attempt = 0
        first_iter = True
        switch_lower_upper = False

        # While looping to find optimum.
        while check_lower_optimum or check_upper_optimum:
            if first_iter:
                adapted_threshold = threshold_default
            elif check_lower_optimum:
                adapted_threshold -= stepsize
            elif switch_lower_upper:
                adapted_threshold = threshold_default + stepsize
                switch_lower_upper = False
            else:
                adapted_threshold += stepsize
            dataset_copy = dataset.copy()
            auc, f1, recall, fpr, specificity = \
                self._calc_stats(dataset_copy, adapted_threshold)
            if first_iter:
                final_threshold = threshold_default
                auc_default = auc
                f1_default = f1
                recall_default = recall
                fpr_default = fpr
                specificity_default = specificity
                auc_optimal = auc
                f1_optimal = f1
            else:
                if f1 > f1_optimal:
                    auc_optimal = auc
                    f1_optimal = f1
                    lower_attempt = 0
                    final_threshold = adapted_threshold
                elif check_lower_optimum:
                    lower_attempt += 1
                    if lower_attempt > max_attempts:
                        check_lower_optimum = False
                        check_upper_optimum = True
                        switch_lower_upper = True
                else:
                    upper_attempt += 1
                    if upper_attempt > max_attempts:
                        check_upper_optimum = False
            first_iter = False

        # Applying optimal scores.
        recall_optimal = recall
        fpr_optimal = fpr
        specificity_optimal = specificity

        # Some extra usefull statistics.
        n_tot = dataset['label'].size
        n_benign = dataset[dataset['label'] == 0]['label'].size
        n_malign = n_tot - n_benign

        # Generating output.
        output = pd.DataFrame({'chr': chro,
                               'gene': gene,
                               'default_auc': auc_default,
                               'default_f1': f1_default,
                               'default_recall': recall_default,
                               'default_fpr': fpr_default,
                               'default_spec': specificity_default,
                               'optimal_c': final_threshold,
                               'optimal_auc': auc_optimal,
                               'optimal_f1': f1_optimal,
                               'optimal_recall': recall_optimal,
                               'optimal_fpr': fpr_optimal,
                               'optimal_spec': specificity_optimal,
                               'n_tot': n_tot,
                               'n_benign': n_benign,
                               'n_malign': n_malign}, index=[0])

        # Providing the result to the manager list.
        if manager_list is not None:
            try:
                manager_list.append(output)
            # Since manager_list is an ProxyList object, which I can't check,
            # excepting Exception.
            except Exception:
                if not ignore_warn:
                    raise TypeError("Did not provide a (manager) list!")
                else:
                    return None
        return output

    @staticmethod
    def _calc_stats(dataset, threshold):
        dataset.loc[
            dataset[dataset['capice'] > threshold].index,
            'capice'
        ] = 1
        dataset.loc[
            dataset[dataset['capice'] <= threshold].index,
            'capice'
        ] = 0
        y_true = np.array(dataset['label'])
        y_pred = np.array(dataset['capice'])
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        fpr = 1-recall
        specificity = 1-fpr
        return auc, f1, recall, fpr, specificity

    def execute(self, dataset):
        processes = []
        L = Manager().list()

        dataset.loc[
            dataset[dataset['label'] == 'LB/B'].index, 'label'
        ] = 0
        dataset.loc[
            dataset[dataset['label'] == 'LP/P'].index, 'label'
        ] = 1
        dataset['label'] = dataset['label'].astype(np.int64)

        total = dataset['gene'].unique().size
        done = 0
        reset_timer = time.time()
        for i, gene in enumerate(dataset['gene'].unique().tolist()):
            time_fls = time.time()
            if time_fls - reset_timer > 5:
                print(
                    f"Still processing, done: "
                    f"{round(done / total * 100, ndigits=2)}%")
                reset_timer = time.time()
            p = Process(target=self.map_optimal_auc_per_gene,
                        args=(dataset, gene, L,))
            p.start()
            processes.append(p)
            done += 1
        for p in processes:
            p.join()

        overview = pd.DataFrame(columns=['chr', 'gene', 'default_auc',
                                         'default_f1',
                                         'default_recall', 'default_fpr',
                                         'default_spec', 'optimal_c',
                                         'optimal_auc', 'optimal_f1',
                                         'optimal_recall', 'optimal_fpr',
                                         'optimal_spec', 'n_tot',
                                         'n_benign', 'n_malign'])

        total = len(L[:])
        done = 0
        reset_timer = time.time()
        for result in L:
            time_fls = time.time()
            if time_fls - reset_timer > 5:
                print(
                    f"Processing results, done: "
                    f"{round(done / total * 100, ndigits=2)}%")
                reset_timer = time.time()
            overview = overview.append(result, ignore_index=True)

        overview.to_csv('/home/rjsietsma/Documents/'
                        'School/Master_DSLS/Final_Thesis/'
                        'Initial_Data_exploration/'
                        'optimal_f1_full_ds_with_chr.csv',
                        index=False)


if __name__ == "__main__":
    PerGene()
