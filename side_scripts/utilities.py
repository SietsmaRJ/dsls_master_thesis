from scipy import stats
import requests
import pandas as pd
import json
import os
from bokeh.plotting import figure, ColumnDataSource, output_file
from bokeh.models import HoverTool, WheelZoomTool, PanTool, \
    BoxZoomTool, ResetTool, SaveTool, FactorRange
from bokeh.palettes import inferno
import gzip
import numpy as np
import time
from sklearn.metrics import roc_auc_score, f1_score, precision_score, \
    recall_score

# https://gist.github.com/deekayen/4148741#file-1-1000-txt

common_used_words = []
with open('/home/rjsietsma/Documents/1-1000.txt') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        common_used_words.append(line)


# Define function to perform Shapiro-Wilk, Kolmogorov-Smirnov,
# Wilcoxon-/Mann-Whitney U -test, and a pearson correlation test.


def perform_stats(x, y):
    shapiro_x = stats.shapiro(x)[1]
    shapiro_y = stats.shapiro(y)[1]
    if shapiro_x >= 0.05:
        print("X is likely normally distributed.")
    else:
        print("X is likely not normally distributed.")
    if shapiro_y >= 0.05:
        print("Y is likely normally distributed.")
    else:
        print("Y is likely not normally distributed.")
    try:
        p_ks_xy = stats.ks_2samp(x, y)[1]
        print(f"The Kolmogorov-Smirnov test p-value: {p_ks_xy}")
    except Exception:
        print("Kolmogorov-Smirnov could not be performed!")
    try:
        p_wc_xy = stats.wilcoxon(x, y)[1]
        print(f"The Wilcoxon test p-value: {p_wc_xy}")
    except ValueError:
        p_mw_xy = stats.mannwhitneyu(x, y)[1]
        print(f"Wilcoxon could not be performed, \n"
              f"Using Mann-Whitney rank test p-value: {p_mw_xy}")
    except Exception:
        print("Neither Wilcoxon nor Mann-Whitney tests could be performed!")
    try:
        p_pears_xy = stats.pearsonr(x, y)
        print(f"The Pearson correlation: {p_pears_xy[0]},\n"
              f"p-value: {p_pears_xy[1]}")
    except Exception:
        print("Pearson correlation could not be performed!")


# Define function to calculate the Z-scores of given data.

def calc_z_scores(data):
    centered = data - data.mean(axis=0)
    return centered / centered.std(axis=0)


def get_enrichr_results(enrichr_sources, user_list_ids, number_of_hits):
    if len(enrichr_sources) == 0:
        print('You should give some sources for me to get data from.')
    else:

        # Prepare output dataframes for each source
        output_dir = './EnrichrAPIResults/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        enrichr_url = 'http://amp.pharm.mssm.edu/Enrichr/'

        for iteration, source in enumerate(enrichr_sources):
            print(f'Working on source: {source}')
            if iteration == len(enrichr_sources):
                sources_left = []
            else:
                sources_left = enrichr_sources[iteration + 1::]
            print(f'Sources left: {sources_left}')
            # Prepare the output dict
            current_source_df = pd.DataFrame(columns=['AUC',
                                                      'Rank',
                                                      'Term_Name',
                                                      'Overlapping_Genes',
                                                      'Adjusted_P-value',
                                                      'Source'])
            for (threshold, user_id) in zip(user_list_ids.keys(),
                                            user_list_ids.values()):
                print(f'\tCurrently working on threshold: {threshold}')
                req = enrichr_url + f'enrich?userListId={user_id}' \
                                    f'&backgroundType={source}'
                response = requests.get(req)
                if not response.ok:
                    print(f'Userid {user_id} (threshold: {threshold})'
                          f'received error on: {source}!')
                else:
                    enrichr_response_data = json.loads(response.text)
                    content = enrichr_response_data[source]
                    auc = threshold
                    content_df = pd.DataFrame(
                        columns=['AUC', 'Rank', 'Term_Name',
                                 'Combined_score',
                                 'Overlapping_Genes',
                                 'Adjusted_P-value',
                                 'Source'])
                    for c in content:
                        rank = c[0]
                        term_name = c[1]
                        combined_score = c[4]
                        overlapping_genes = ', '.join(c[5])
                        a_p_value = c[6]
                        local_df = pd.DataFrame(
                            {'AUC': auc, 'Rank': rank,
                             'Term_Name': term_name,
                             'Combined_score': combined_score,
                             'Overlapping_Genes': overlapping_genes,
                             'Adjusted_P-value': a_p_value,
                             'Source': source}, index=[0])
                        content_df = content_df.append(local_df,
                                                       ignore_index=True)
                    content_df = content_df.sort_values(by=['Adjusted_P-value',
                                                            'Combined_score'],
                                                        ascending=[True, False])
                    max_rows = 0
                    if content_df.shape[0] < number_of_hits:
                        max_rows = content_df.shape[0] - 1
                    else:
                        max_rows = number_of_hits
                    out_df = content_df[:max_rows]
                    out_df.drop('Combined_score', axis=1, inplace=True)
                    current_source_df = current_source_df.append(out_df,
                                                                 ignore_index=True)
            output_filename = output_dir + source + '.csv'
            current_source_df.to_csv(output_filename, index=False, sep=',')
            print(
                f'\tSource {source} has been'
                f' exported to csv (\n\t\t{output_filename}\n\t)')


def plot_results(source, n=10, print_messages=False):
    source_name = source['Source'][0]
    if not os.path.exists(f'./EnrichrAPIResults/Plots_for_{n}_categories'):
        os.mkdir(f'./EnrichrAPIResults/Plots_for_{n}_categories')
    output_file(
        filename=f'./EnrichrAPIResults/Plots_for_{n}_categories/n_{n}'
                 f'_results_from_{source_name}.html',
        title=f'Bokeh plot of {source_name} (n={n})')
    term_values = source['Term_Name'].value_counts()
    if print_messages:
        print(
            f'The most occuring term names in {source_name} '
            f'is: \n{term_values[:n]}')
    plot_source = pd.DataFrame(columns=source.columns)
    for auc in source['AUC'].unique():
        subsource = source[source['AUC'] == auc][:n]
        plot_source = plot_source.append(subsource)
    category = 'Term_Name'
    category_items = plot_source[category].unique()
    palette = inferno(len(category_items) + 1)
    colormap = dict(zip(category_items, palette))
    plot_source['color'] = plot_source[category].map(colormap)

    title = f'Term names from {source_name}, n={n}'
    source_bokeh = ColumnDataSource(plot_source)
    hover = HoverTool(
        tooltips=[('Term_name', '@Term_Name'), ('Genes', '@Overlapping_Genes')])
    tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool(),
             SaveTool()]

    p = figure(tools=tools, title=title, plot_width=2000, plot_height=1200,
               toolbar_location='right', toolbar_sticky=False, )
    if n > 5:
        p.scatter(x='AUC', y='Adjusted_P-value', source=source_bokeh, size=10,
                  color='color')
    else:
        p.scatter(x='AUC', y='Adjusted_P-value', source=source_bokeh, size=10,
                  color='color', legend=category)
    p.xaxis.axis_label = 'AUC'
    p.yaxis.axis_label = 'Adjusted P-value'
    return p


def plot_results_in_singleplot(source):
    output_file(
        filename='./EnrichrAPIResults/combined_enrichr_plot_results.html',
        title='Combined enrichr data sources plot')
    category = 'Source'
    category_items = source[category].unique()
    palette = inferno(len(category_items) + 1)
    colormap = dict(zip(category_items, palette))
    source['color'] = source[category].map(colormap)

    title = f'Combined enrichr plot results.'
    source_bokeh = ColumnDataSource(source)
    hover = HoverTool(tooltips=[('Term_name', '@Term_Name'),
                                ('Genes', '@Overlapping_Genes'),
                                ('Source', '@Source')])
    tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool(),
             SaveTool()]

    p = figure(tools=tools, title=title, plot_width=2000, plot_height=1200,
               toolbar_location='right', toolbar_sticky=False, )
    p.scatter(x='AUC', y='Adjusted_P-value', source=source_bokeh, size=10,
              color='color', legend=category)
    p.xaxis.axis_label = 'AUC'
    p.yaxis.axis_label = 'Adjusted P-value'
    return p


def plot_results_in_singleplot_iteractive_legend(source):
    output_file(filename='./EnrichrAPIResults/combined_'
                         'enrichr_plot_results_interactive_legend.html',
                title='Combined enrichr data sources plot IL')
    category = 'Source'
    category_items = source[category].unique()
    palette = inferno(len(category_items) + 1)
    colormap = dict(zip(category_items, palette))
    source['color'] = source[category].map(colormap)
    p = figure(plot_width=2000, plot_height=1200, toolbar_location='right',
               toolbar_sticky=False)
    p.title.text = f'Combined enrichr plot results.'
    for data in source[category].unique():
        subsource = ColumnDataSource(source[source[category] == data])
        p.scatter(x='AUC', y='Adjusted_P-value',
                  size=10, color='color',
                  legend_label=data,
                  source=subsource)
    p.xaxis.axis_label = 'AUC'
    p.yaxis.axis_label = 'Adjusted P-value'
    p.legend.click_policy = "hide"
    return p


def count_words_in_df(dataframe):
    word_count_df = pd.DataFrame(columns=['word', 'count', 'auc'])
    for auc in dataframe['AUC'].unique():
        word_list = []
        for source in dataframe['Source'].unique():
            subset = dataframe.where((dataframe['AUC'] == auc) &
                                     (dataframe['Source'] == source)).dropna()
            for i, row in subset.iterrows():
                term_name = row['Term_Name']
                word_list += term_name.lower().split()
        word_count_dict = {}
        for word in word_list:
            if word in common_used_words:
                continue
            if word not in word_count_dict.keys():
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1
        temp_df = pd.DataFrame(columns=word_count_df.columns)
        for key in word_count_dict.keys():
            temp_temp_df = pd.DataFrame(
                {'word': key, 'count': word_count_dict[key],
                 'auc': auc}, index=[0])
            temp_df = temp_df.append(temp_temp_df, ignore_index=True)
        word_count_df = word_count_df.append(temp_df, ignore_index=True)
    word_count_df.sort_values(by=['auc', 'count'], ascending=[True, False],
                              inplace=True, ignore_index=True)
    return word_count_df


def plot_count_results(source, item):
    if item == 'word':
        output_file(
            filename='./EnrichrAPIResults/enrichr_word_count_bokeh.html',
            title='Word count plot')
        category = 'word'
    elif item == 'gene':
        output_file(
            filename='./EnrichrAPIResults/enrichr_gene_count_bokeh.html',
            title='Gene count plot')

        category = 'gene'
    else:
        return None
    category_items = source[category].unique()
    palette = inferno(len(category_items) + 1)
    colormap = dict(zip(category_items, palette))
    source['color'] = source[category].map(colormap)

    source['auc'] = source['auc'].astype(str)

    group = source.groupby(by=['auc', category]).sum()
    group.sort_values(by=['auc', 'count'], ascending=[True, False],
                      inplace=True)

    title = f'{category} count bar plot'

    factors = group.index.tolist()
    x = group['count'].tolist()
    colors = group['color'].tolist()

    p = figure(x_range=FactorRange(*factors),
               title=title,
               plot_width=2000,
               plot_height=1200,
               toolbar_location='right',
               toolbar_sticky=False)
    p.vbar(x=factors,
           top=x,
           color=colors,
           width=0.9)
    p.add_tools(HoverTool(tooltips=[('Counts', '@top')]))
    p.xaxis.axis_label = category.capitalize()
    p.yaxis.axis_label = 'Count'
    p.xaxis.major_label_orientation = 1.5
    return p


def split_and_count_words(x, return_value='int'):
    x = x.split(', ')
    if return_value == 'int':
        return len(x)
    else:
        if not isinstance(x, list):
            x = [x]
        return x


def get_header(file_loc, start):
    header = None
    with gzip.open(file_loc) as file:
        while True:
            line = file.readline().decode('utf-8')
            if line.startswith(start):
                header = line.strip().split('\t')
                break
    return header


def genepanel_analysis(genepanels, data):
    genepanel_df = pd.DataFrame(columns=['category', 'panel', 'auc'])
    for category, panel_genes in genepanels.items():
        for panel, genes in panel_genes.items():
            subset = data[data['gene'].isin(genes)]
            x = np.array(subset['auc'])
            x_mean = x.mean()
            x_std = x.std()
            genepanel_df = genepanel_df.append(
                pd.DataFrame(
                    {
                        'category': [category],
                        'panel': [panel],
                        'auc': [x_mean],
                        'std': [x_std],
                        'n_benign': [subset['n_benign'].sum()],
                        'n_malign': [subset['n_malign'].sum()],
                        'n_tot': [subset['n_tot'].sum()],
                        'n_train': [subset['n_train'].sum()]
                    }, index=[0]
                ), ignore_index=True
            )
    mann_whitney_cats = ['two-sided', 'less', 'greater']
    return_df = pd.DataFrame(
        columns=mann_whitney_cats + ['category', 'compared_to', 'mean', 'std',
                                     'n_benign','n_malign','n_tot','n_train'])
    for category in genepanel_df['category'].unique():
        subset = genepanel_df[genepanel_df['category'] == category]
        x = np.array(subset['auc'])
        y = np.array(genepanel_df[genepanel_df['category'] != category]['auc'])
        output = {'category': ['all'],
                  'compared_to': [category],
                  'two-sided': None,
                  'less': None,
                  'greater': None,
                  'mean': [x.mean()],
                  'std': [x.std()],
                  'n_benign': [int(subset['n_benign'].sum())],
                  'n_malign': [int(subset['n_malign'].sum())],
                  'n_tot': [int(subset['n_tot'].sum())],
                  'n_train': [int(subset['n_train'].sum())]
                  }
        for alternative in mann_whitney_cats:
            output[alternative] = [stats.mannwhitneyu(
                x, y, alternative=alternative)[1]]
        return_df = return_df.append(
            pd.DataFrame(
                output, index=[0]
            ), ignore_index=True
        )
    return return_df


def analyze_auc_per_gene(dataset, output_name):
    nsd = './not_saving_directory/'
    if not os.path.exists(nsd):
        os.mkdir(nsd)
    auc_analysis_output_filename = os.path.join(nsd, output_name)
    if not os.path.isfile(auc_analysis_output_filename):
        auc_analysis = pd.DataFrame(columns=['gene', 'auc', 'f1', 'recall',
                                             'fpr', 'precision',
                                             'n_benign', 'n_malign',
                                             'n_tot', 'n_train', 'n_test'])
        dataset['label'].replace({
            'Pathogenic': 1,
            'Benign': 0,
            'Neutral': 0
        }, inplace=True)
        dataset['prediction'].replace({
            'Pathogenic': 1,
            'Benign': 0,
            'Neutral': 0
        }, inplace=True)
        n_tot_iters = dataset['GeneName'].unique().size
        done_iters = 0
        t_fls = time.time()
        for gene in dataset['GeneName'].unique():
            done_iters += 1
            t_ifl = time.time()
            if t_ifl - t_fls > 10:
                print(f'I am stilling running,'
                      f' done {round(done_iters / n_tot_iters * 100)}%')
                t_fls = time.time()
            subset = dataset[dataset['GeneName'] == gene]
            if subset['label'].unique().size > 1:
                y_true = np.array(subset['label'])
                y_pred = np.array(subset['probabilities'])
                y_pred_label = np.array(subset['prediction'])
                n_train = subset[subset['source'] == 'train'].shape[0]
                n_test = subset[subset['source'] == 'test'].shape[0]
                auc = roc_auc_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred_label)
                recall = recall_score(y_true, y_pred_label, zero_division=0)
                fpr = 1 - recall
                precision = precision_score(y_true, y_pred_label, zero_division=0)
                n_benign = y_true[y_true == 0].size
                n_malign = y_true[y_true == 1].size
                n_tot = y_true.size
            else:
                continue
            auc_analysis = auc_analysis.append(
                pd.DataFrame({
                    'gene': [gene],
                    'auc': [auc],
                    'f1': [f1],
                    'recall': [recall],
                    'fpr': [fpr],
                    'precision': [precision],
                    'n_benign': [n_benign],
                    'n_malign': [n_malign],
                    'n_tot': [n_tot],
                    'n_train': [n_train],
                    'n_test': [n_test]
                }, index=[0]), ignore_index=True)
        auc_analysis.to_csv(auc_analysis_output_filename)
    else:
        auc_analysis = pd.read_csv(auc_analysis_output_filename,
                                   index_col=0)
    return auc_analysis


def correct_threshold(train_results=None, test_results=None,
                      include_upper=True,
                      starting=0):
    thresholds = np.arange(starting, 1, 0.001)
    if train_results is not None:
        print('Doing train.')
        train_input = pd.read_csv('./datafiles/train.txt.gz',
                                  sep='\t',
                                  low_memory=False,
                                  usecols=['#Chrom', 'Pos', 'Ref', 'Alt',
                                           'label'])
        train_input.rename(columns={
            '#Chrom': 'chr',
            'Pos': 'pos',
            'Ref': 'ref',
            'Alt': 'alt'
        }, inplace=True)
        train_input['pos'] = train_input['pos'].astype(np.int64)
        data = train_results.merge(train_input)
    elif test_results is not None:
        print('Doing test.')
        test_input = pd.read_csv('./datafiles/test.txt.gz',
                                 sep='\t',
                                 low_memory=False,
                                 usecols=['#Chrom', 'Pos', 'Ref', 'Alt',
                                          'label'])
        test_input.rename(columns={
            '#Chrom': 'chr',
            'Pos': 'pos',
            'Ref': 'ref',
            'Alt': 'alt'
        }, inplace=True)
        test_input['pos'] = test_input['pos'].astype(np.int64)
        data = test_results.merge(test_input, on=['chr', 'pos', 'ref', 'alt'])
    else:
        raise AttributeError('Either train or test dataset must be supplied.')

    if data.shape[0] < 1:
        raise ValueError('Merge did not resolve in any datapoints.')

    def apply_func_thresholding(probability, loop_threshold):
        return_value = 0
        if probability > loop_threshold:
            return_value = 1
        return return_value

    data['label'].replace({'Pathogenic': 1, 'Benign': 0}, inplace=True)

    true_recall = 0
    true_threshold = 0
    precision = 0
    f1 = 0
    upper_thres = 1.1
    if include_upper:
        upper_thres = 0.96
    for threshold in thresholds:
        data['pred_label'] = data['probabilities'].apply(
            lambda i: apply_func_thresholding(i, threshold))
        y_pred = np.array(data['probabilities'])
        y_true = np.array(data['label'])
        recall = recall_score(y_pred=y_pred, y_true=y_true, zero_division=0)
        if 0.94 <= recall <= upper_thres:
            true_recall = recall
            true_threshold = threshold
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            break
    print(f"The real recall of input is: {true_recall},\n"
          f"at a threshold of: {true_threshold}.")
    print(f"Precision score: {precision}")
    print(f"F1 score: {f1}")
    return true_recall, true_threshold


def read_capice_output(capice: str):
    output = pd.read_csv(capice,
                         sep='\t',
                         low_memory=False)
    output[['chr', 'pos', 'ref', 'alt']] = output['chr_pos_ref_alt'].str.split(
        '_', expand=True)
    output['pos'] = output['pos'].astype(np.int64)
    output.drop(columns=['chr_pos_ref_alt'], inplace=True)
    return output


def auc_analysis_function(train_output: pd.DataFrame,
                          test_output: pd.DataFrame,
                          return_value=False):
    train_input = pd.read_csv('./datafiles/train.txt.gz',
                              sep='\t',
                              low_memory=False,
                              usecols=['#Chrom', 'Pos', 'Ref', 'Alt', 'label'])
    train_input.rename(columns={
        '#Chrom': 'chr',
        'Pos': 'pos',
        'Ref': 'ref',
        'Alt': 'alt'
    }, inplace=True)
    train_input['pos'] = train_input['pos'].astype(np.int64)
    test_input = pd.read_csv('./datafiles/test.txt.gz',
                             sep='\t',
                             low_memory=False,
                             usecols=['#Chrom', 'Pos', 'Ref', 'Alt', 'label'])
    test_input.rename(columns={
        '#Chrom': 'chr',
        'Pos': 'pos',
        'Ref': 'ref',
        'Alt': 'alt'
    }, inplace=True)
    test_input['pos'] = test_input['pos'].astype(np.int64)
    # First, the train dataset:
    train_merge = train_output.merge(train_input)
    train_merge['label'].replace({'Pathogenic': 1, 'Benign': 0},
                                 inplace=True)
    y_true_train = np.array(train_merge['label'])
    y_pred_train = np.array(train_merge['probabilities'])
    print(f"AUC analysis of the training dataset reveals AUC: "
          f"{roc_auc_score(y_true=y_true_train, y_score=y_pred_train)}")

    # Now the test dataset
    test_merge = test_output.merge(test_input)
    test_merge['label'].replace({'Pathogenic': 1, 'Benign': 0},
                                inplace=True)
    y_true_test = np.array(test_merge['label'])
    y_pred_test = np.array(test_merge['probabilities'])
    print(f"AUC analysis of the testing dataset reveals AUC: "
          f"{roc_auc_score(y_true=y_true_test, y_score=y_pred_test)}")

    if return_value:
        train_merge['label'].replace({1: 'Pathogenic',
                                      0: 'Benign'},
                                     inplace=True)
        test_merge['label'].replace({1: 'Pathogenic',
                                     0: 'Benign'},
                                    inplace=True)
        return train_merge, test_merge
