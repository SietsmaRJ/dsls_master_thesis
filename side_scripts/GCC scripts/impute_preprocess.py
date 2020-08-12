import pandas as pd
import numpy as np
import pickle


class ImputePreprocess:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def examine_nas(self, df):
        sample_num = df.shape[0]
        null_ratios = {}
        for col in df.columns:
            null_number = df[col].isnull().sum()
            if null_number > 0:
                null_ratios[col] = null_number / sample_num
                self._printf(col, null_number,
                             round(null_number / sample_num, 2))
        return null_ratios

    def replace_nas(self, df):
        for value in df.columns:
            if df[value].isna().any() and value in self._impute_values().keys():
                df[value].fillna(self.get_impute_values(value), inplace=True)
            else:
                continue
        return df

    def impute(self, df, imputed_savepath=None):
        self._printf(type(df))
        self._printf("Readin data shape: ", df.shape)
        self._printf(df.head())
        df = df.dropna(subset=self._cadd_vars(), how="all")
        self._printf("Remove samples with no parameters, shape: ", df.shape)
        func = lambda x: np.nan if pd.isnull(
            x) or x == "." or x == 'NA' else float(
            x)
        values = df["dbscSNV-rf_score"].values
        df["dbscSNV-rf_score"] = [func(item) for item in values]
        df = df.dropna(how="all")
        self._printf("Raw data loaded, shape: ", df.shape)
        self._printf("Before imputation, null ratio: \n")
        self.examine_nas(df[self._cadd_vars()])
        # save null ratios
        df = self.replace_nas(df)
        self._printf(
            "After imputation, there shouldn't be any nulls, but check below: \n")
        self.examine_nas(df)
        if imputed_savepath:
            df.to_csv(imputed_savepath, index=False)
            self._printf("Saved imputed raw file to %s" % imputed_savepath)
        return df

    def return_top10_or_less_categories(self, a_column, return_num=10):
        value_counts = a_column.value_counts()
        if len(value_counts) > return_num:
            self._printf(value_counts.index[:return_num].values)
            return value_counts.index[:return_num].values
        else:
            self._printf(value_counts.index.values)
            return value_counts.index.values

    def process_categoricalvars(self, data, feat_cadd_object, isTrain=False,
                                catFeats_levels_dict=None,
                                catFeatNames_dict=None):
        if isTrain:
            self._printf(
                "Determining feature levels from the training dataset.")
            for catFeat in catFeats_levels_dict.keys():
                featNames = self.return_top10_or_less_categories(data[catFeat],
                                                                 return_num=
                                                                 catFeats_levels_dict[
                                                                     catFeat])
                self._printf(
                    "For feature %s, saved %d levels." % (
                        catFeat, len(featNames)))
                data[catFeat] = np.where(data[catFeat].isin(featNames),
                                         data[catFeat], "other")
        else:
            self._printf("Using features from the trained model.")
            for catFeat in catFeatNames_dict.keys():
                featNames = catFeatNames_dict[catFeat]
                self._printf(
                    "For feature %s, saved %d levels." % (
                        catFeat, len(featNames)))
                data[catFeat] = np.where(data[catFeat].isin(featNames),
                                         data[catFeat], "other")
        data = pd.get_dummies(data, columns=feat_cadd_object)
        return data

    def preprocess(self, imputed_data, processed_savepath=None, isTrain=False,
                   model_path=None, model_features=None):
        feat_cadd_object = [feat for feat in
                            imputed_data.select_dtypes(include=["O"]).columns
                            if feat in self._cadd_vars()]
        print(feat_cadd_object)
        self._printf("Categorical variables", len(feat_cadd_object))
        num_samples = imputed_data.shape[0]
        self._printf("In total, there are %d samples" % num_samples)
        catFeats_levels_dict = {"Ref": 5, "Alt": 5, "Domain": 5}
        if isTrain:
            for feat in feat_cadd_object:
                if feat not in catFeats_levels_dict:
                    catFeats_levels_dict[feat] = 5
            processed_data = self.process_categoricalvars(imputed_data,
                                                          feat_cadd_object=feat_cadd_object,
                                                          isTrain=isTrain,
                                                          catFeats_levels_dict=catFeats_levels_dict)
        else:
            if model_path:
                model_features = pickle.load(
                    open(model_path, 'rb')).feature_names
            elif model_features:
                model_features = model_features
            else:
                self._printf(
                    "In testing phase, features needs to be specified or pretrained models needs to be provided...")
            catFeatNames_dict = {}
            for feature in feat_cadd_object:
                for feature_expandedname in model_features:
                    if feature in feature_expandedname:
                        expandedname = '_'.join(
                            feature_expandedname.split('_')[1:])
                        if feature in catFeatNames_dict:
                            catFeatNames_dict[feature].append(expandedname)
                        else:
                            catFeatNames_dict[feature] = [expandedname]
            processed_data = self.process_categoricalvars(imputed_data,
                                                          feat_cadd_object=feat_cadd_object,
                                                          isTrain=isTrain,
                                                          catFeatNames_dict=catFeatNames_dict)
            for col in model_features:
                if col not in processed_data:
                    processed_data[col] = 0
                    self._printf("Feature from the model not in data: ", col)
        self._printf(processed_data.shape)
        if processed_savepath:
            self._printf("Saving preprocessed data to ", processed_savepath)
            processed_data.to_csv(processed_savepath, index=False)
        return processed_data

    def preprocess_withConsequence(self, imputed_data, processed_savepath=None,
                                   isTrain=False, model_path=None):
        feat_cadd_object = [feat for feat in
                            imputed_data.select_dtypes(include=["O"]).columns
                            if feat in self._cadd_vars_with_consequence()]
        self._printf("Categorical variables", len(feat_cadd_object))
        num_samples = imputed_data.shape[0]
        self._printf("In total, there are %d samples" % num_samples)
        catFeats_levels_dict = {"Ref": 5, "Alt": 5, "Domain": 5,
                                "Consequence": 5}
        if isTrain:
            for feat in feat_cadd_object:
                if feat not in catFeats_levels_dict:
                    catFeats_levels_dict[feat] = 5
            processed_data = self.process_categoricalvars(imputed_data,
                                                          feat_cadd_object=feat_cadd_object,
                                                          isTrain=isTrain,
                                                          catFeats_levels_dict=catFeats_levels_dict)
        else:
            model_features = pickle.load(open(model_path, 'rb')).feature_names
            catFeatNames_dict = {}
            for feature in feat_cadd_object:
                for feature_expandedname in model_features:
                    if feature in feature_expandedname:
                        expandedname = '_'.join(
                            feature_expandedname.split('_')[1:])
                        if feature in catFeatNames_dict:
                            catFeatNames_dict[feature].append(expandedname)
                        else:
                            catFeatNames_dict[feature] = [expandedname]
            processed_data = self.process_categoricalvars(imputed_data,
                                                          feat_cadd_object=feat_cadd_object,
                                                          isTrain=isTrain,
                                                          catFeatNames_dict=catFeatNames_dict)
            for col in model_features:
                if col not in processed_data:
                    processed_data[col] = 0
                    self._printf("Feature from the model not in data: ", col)
        self._printf(processed_data.shape)
        if processed_savepath:
            self._printf("Saving preprocessed data to ", processed_savepath)
            processed_data.to_csv(processed_savepath, index=False)
        return processed_data

    @staticmethod
    def _cadd_vars():
        return ['Ref', 'Alt', 'Type', 'Length', 'GC', 'CpG', 'motifECount',
                'motifEScoreChng', 'motifEHIPos',
                'oAA', 'nAA', 'cDNApos', 'relcDNApos', 'CDSpos',
                'relCDSpos',
                'protPos', 'relProtPos', 'Domain', 'Dst2Splice',
                'Dst2SplType', 'minDistTSS', 'minDistTSE', 'SIFTcat',
                'SIFTval',
                'PolyPhenCat', 'PolyPhenVal', 'priPhCons',
                'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP',
                'verPhyloP',
                'bStatistic', 'targetScan', 'mirSVR-Score',
                'mirSVR-E', 'mirSVR-Aln', 'cHmmTssA', 'cHmmTssAFlnk',
                'cHmmTxFlnk',
                'cHmmTx', 'cHmmTxWk', 'cHmmEnhG',
                'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv',
                'cHmmBivFlnk',
                'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk',
                'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS',
                'TFBS',
                'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs',
                'motifDist', 'Segway', 'EncH3K27Ac', 'EncH3K4Me1',
                'EncH3K4Me3',
                'EncExp', 'EncNucleo', 'EncOCC', 'EncOCCombPVal',
                'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal',
                'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig',
                'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig',
                'EncOCmycSig',
                'Grantham', 'Dist2Mutation', 'Freq100bp',
                'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp',
                'Sngl1000bp',
                'Freq10000bp', 'Rare10000bp',
                'Sngl10000bp', 'dbscSNV-ada_score',
                'dbscSNV-rf_score']

    def get_cadd_vars(self):
        return self._cadd_vars()

    def _cadd_vars_with_consequence(self):
        return self._cadd_vars() + ['RawScore', 'PHRED', "Consequence"]

    def get_cadd_vars_with_consequence(self):
        return self._cadd_vars_with_consequence()

    @staticmethod
    def _impute_values():
        return {'Ref': 'N', 'Alt': 'N', 'Consequence': 'UNKNOWN',
                'GC': 0.42,
                'CpG': 0.02, 'motifECount': 0,
                'motifEScoreChng': 0, 'motifEHIPos': 0,
                'oAA': 'unknown',
                'nAA': 'unknown', 'cDNApos': 0,
                'relcDNApos': 0, 'CDSpos': 0, 'relCDSpos': 0,
                'protPos': 0,
                'relProtPos': 0, 'Domain': 'UD', 'Dst2Splice': 0,
                'Dst2SplType': 'unknown', 'minDistTSS': 5.5,
                'minDistTSE': 5.5,
                'SIFTcat': 'UD', 'SIFTval': 0,
                'PolyPhenCat': 'unknown', 'PolyPhenVal': 0,
                'priPhCons': 0.115,
                'mamPhCons': 0.079, 'verPhCons': 0.094,
                'priPhyloP': -0.033, 'mamPhyloP': -0.038,
                'verPhyloP': 0.017,
                'bStatistic': 800, 'targetScan': 0,
                'mirSVR-Score': 0, 'mirSVR-E': 0, 'mirSVR-Aln': 0,
                'cHmmTssA': 0.0667, 'cHmmTssAFlnk': 0.0667,
                'cHmmTxFlnk': 0.0667, 'cHmmTx': 0.0667,
                'cHmmTxWk': 0.0667,
                'cHmmEnhG': 0.0667, 'cHmmEnh': 0.0667,
                'cHmmZnfRpts': 0.0667, 'cHmmHet': 0.667,
                'cHmmTssBiv': 0.667,
                'cHmmBivFlnk': 0.0667, 'cHmmEnhBiv': 0.0667,
                'cHmmReprPC': 0.0667, 'cHmmReprPCWk': 0.0667,
                'cHmmQuies': 0.0667, 'GerpRS': 0, 'GerpRSpval': 0,
                'GerpN': 1.91, 'GerpS': -0.2, 'TFBS': 0,
                'TFBSPeaks': 0,
                'TFBSPeaksMax': 0, 'tOverlapMotifs': 0,
                'motifDist': 0, 'Segway': 'unknown', 'EncH3K27Ac': 0,
                'EncH3K4Me1': 0, 'EncH3K4Me3': 0, 'EncExp': 0,
                'EncNucleo': 0, 'EncOCC': 5, 'EncOCCombPVal': 0,
                'EncOCDNasePVal': 0, 'EncOCFairePVal': 0,
                'EncOCpolIIPVal': 0, 'EncOCctcfPVal': 0,
                'EncOCmycPVal': 0,
                'EncOCDNaseSig': 0, 'EncOCFaireSig': 0,
                'EncOCpolIISig': 0, 'EncOCctcfSig': 0,
                'EncOCmycSig': 0,
                'Grantham': 0, 'Dist2Mutation': 0,
                'Freq100bp': 0, 'Rare100bp': 0, 'Sngl100bp': 0,
                'Freq1000bp': 0, 'Rare1000bp': 0, 'Sngl1000bp': 0,
                'Freq10000bp': 0, 'Rare10000bp': 0, 'Sngl10000bp': 0,
                'dbscSNV-ada_score': 0,
                'dbscSNV-rf_score': 0}

    def get_impute_values(self, key):
        return self._impute_values()[key]

    def _printf(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
