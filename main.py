import datetime as dt
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
from function_cluster import KMeansClustering
from data_loader import DataLoader
from data_labeling import raw_returns_labeling
from helper_functions import merge_small_clusters, dcor_parallel

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Create a dictionary where we define the configuration parameters
    config = dict()
    config['n_jobs'] = 10
    config['symbol'] = 'EURUSD'
    config['time_frame'] = '1d'

    # Define the direction of the extraction
    signal = 'long'
    if signal == 'longonly':
        config['signal_direction'] = 'long'
        config['type_bias'] = 'up'
    else:
        config['signal_direction'] = 'short'
        config['type_bias'] = 'down'

    # Configure the labeling
    config['bias_extraction'] = True
    config['shift'] = 1
    config['bias'] = 0.001

    # Define the duration in years and partition of the dataset
    config['years_train_valid'] = 10
    years_safe_date = 2

    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=config['years_train_valid'] * 365)
    safe_start_date = start_date - dt.timedelta(days=years_safe_date * 365)
    safe_start_date = dt.datetime.strftime(safe_start_date, "%Y-%m-%d")
    start_date = dt.datetime.strftime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strftime(end_date, "%Y-%m-%d")

    config['safe_start_date'] = safe_start_date
    config['start_date'] = start_date
    config['end_date'] = end_date

    # Load the data from the MetaTrader API
    data_loader = DataLoader()
    data_loader.load_from_MT5(symbol=config['symbol'], 
                              start_date=config['safe_start_date'],
                              end_date=config['end_date'],
                              time_frame=config['time_frame'])

    data_symbol = data_loader.get_data()
    data_symbol.drop(data_symbol.columns[4], axis=1, inplace=True)

    # Create the label to predict
    data_symbol = raw_returns_labeling(data_ohlc=data_symbol, **config)

    label_counts = data_symbol['LABEL'].value_counts()

    data_label = data_symbol['LABEL']
    data_label = data_label.replace('UP', 1).replace('DOWN', 0) \
        if signal == 'longonly' \
            else data_label.replace('UP', 0).replace('DOWN', 1)
            
    data_label = data_label[data_label.index > config['start_date']]

    data_label.value_counts().plot(kind='bar')
    plt.show()

    # Create Features
    calc_indicator = []

    for length in range(2, 100):
        indicator_value = ta.rsi(data_symbol.Close, length).rename(length)
        calc_indicator.append(indicator_value)

    # Concatenate all indicators into a DataFrame and filter dates
    data_indicator = pd.concat(calc_indicator, axis=1).dropna()
    data_indicator = data_indicator[data_indicator.index > config['start_date']]

    # Combine the indicator DataFrame with the target variable
    df_combined = data_indicator.copy()
    df_combined['target'] = data_label

    # Calculate the distance correlation matrix
    dcor_matrix = dcor_parallel(df_combined, n_jobs=10)
    data_correlation = dcor_matrix['target'].drop('target').rename('dcor')
    data_correlation.head(10)

    # Create correlation graph
    sns.clustermap(dcor_matrix.iloc[:, :-1].drop('target'), cmap='coolwarm', linewidth=1, method='ward')
    plt.show()

    data_indicator_t = data_indicator.T

    # Scale the data
    data_indicator_norm = (data_indicator_t - data_indicator_t.min()) \
            / (data_indicator_t.max() - data_indicator_t.min())

    # Apply k-means clustering and obtain cluster labels
    kmeans = KMeansClustering()
    cluster_labels = kmeans.fit_predict(data_indicator_norm)
    data_indicator_t['cluster_labels'] = cluster_labels

    # Merge small clusters
    data_indicator_t = merge_small_clusters(data_indicator_t, cluster_col='cluster_labels', min_size=3)

    count_num_clusters = data_indicator_t['cluster_labels'].value_counts()

    # Obtain the ranges of the indicator in each cluster
    range_length = []

    for item_cluster in range(len(count_num_clusters)):
        range_length.append(data_indicator_t[data_indicator_t['cluster_labels'] == item_cluster].index)

    range_length_df = pd.DataFrame(range_length)

    # Count how many elements each cluster has
    elements_cluster = range_length_df.count(axis=1)

    # Create a DataFrame that summarizes the max and min range of the indicator and 
    # number of elements in each cluster
    count_min_max_length = pd.concat([elements_cluster, range_length_df.min(axis=1), range_length_df.max(axis=1)], axis=1)
    count_min_max_length.columns = ['elements_cluster', 'min', 'max']

    # Determine the average value of the correlation with the target in each cluster
    mean_cluster_range_length = list()    
    for items_range in range_length:
        mean_corr_cluster = data_correlation.filter(items=items_range, axis=0).mean()
        mean_cluster_range_length.append(mean_corr_cluster)

    mean_cluster_range_length_df = pd.DataFrame(mean_cluster_range_length, columns=['rank_cluster_corr'])
    mean_cluster_range_length_df.sort_values(by='rank_cluster_corr', ascending=False)

    # From each cluster, select the period that has the highest correlation with the target variable
    instances = 1

    best_length_cluster_corr = list()
    for items_range in range_length:
        select_best = data_correlation.filter(items=items_range, axis=0).sort_values(ascending=False).head(instances)
        best_length_cluster_corr.append(select_best)

    best_features = pd.concat(best_length_cluster_corr)

    # Create correlation graph
    data_indicator_best = data_indicator.loc[:, best_features.index]

    corr_matrix = dcor_parallel(data_indicator_best)
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidth=1, method='ward', annot=True, fmt=".2f")
    plt.show()
