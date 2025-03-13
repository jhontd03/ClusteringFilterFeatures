import itertools
from joblib import Parallel, delayed
import dcor
from scipy.spatial.distance import squareform
import pandas as pd

def merge_small_clusters(df, cluster_col='cluster_labels', min_size=3):
    """
    Merges small clusters in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing cluster labels.
        cluster_col (str): Column name for cluster labels. Default is 'cluster_labels'.
        min_size (int): Minimum size for clusters to avoid merging. Default is 3.

    Returns:
        pd.DataFrame: DataFrame with merged clusters.
    """
    # Count the number of elements in each cluster
    cluster_counts = df[cluster_col].value_counts()
    
    # Identify clusters with fewer than min_size elements
    small_clusters = cluster_counts[cluster_counts < min_size].index.tolist()
    
    # Reassign elements of small clusters to a temporary value (-1)
    df.loc[df[cluster_col].isin(small_clusters), cluster_col] = -1
    
    # Relabel the clusters to have consecutive labels
    unique_labels = sorted(df[cluster_col].unique())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    df[cluster_col] = df[cluster_col].map(label_mapping)
    
    return df

def dcor_parallel(data: pd.DataFrame, n_jobs: int = 10) -> pd.DataFrame:
    """
    Computes distance correlation in parallel.

    Args:
        data (pd.DataFrame): DataFrame containing the data for correlation calculation.
        n_jobs (int): Number of parallel jobs to run. Default is 10.

    Returns:
        pd.DataFrame: DataFrame of distance correlations with the same columns as input.
    """
    columns = data.columns
    data_columns = [x[1].values for x in data.items()]
    pair_order_list = list(itertools.combinations(data_columns, 2))

    correlations = Parallel(n_jobs=n_jobs)(
        delayed(dcor.distance_correlation)(p[0], p[1]) 
        for p in pair_order_list
    )

    correlations = squareform(correlations)

    return pd.DataFrame(correlations, columns=columns, index=columns)