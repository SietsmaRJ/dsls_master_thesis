from scipy import stats
import requests
import pandas as pd
import json
import os
from bokeh.plotting import figure, ColumnDataSource, output_file
from bokeh.models import HoverTool, WheelZoomTool, PanTool,\
    BoxZoomTool, ResetTool, SaveTool, FactorRange
from bokeh.palettes import inferno

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
    p = figure(plot_width=2000, plot_height=1200,toolbar_location='right',
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
    p.legend.click_policy="hide"
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
            temp_temp_df = pd.DataFrame({'word':key, 'count': word_count_dict[key],
                                        'auc': auc}, index=[0])
            temp_df = temp_df.append(temp_temp_df, ignore_index=True)
        word_count_df = word_count_df.append(temp_df, ignore_index=True)
    word_count_df.sort_values(by=['auc','count'], ascending=[True, False],
                              inplace=True, ignore_index=True)
    return word_count_df


def plot_count_results(source, item):
    if item == 'word':
        output_file(filename='./EnrichrAPIResults/enrichr_word_count_bokeh.html',
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
