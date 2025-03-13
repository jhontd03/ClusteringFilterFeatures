import numpy as np
import pandas as pd

def raw_returns_labeling(data_ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Labels price data based on future price movements.

    Args:
        data_ohlc (pd.DataFrame): DataFrame containing OHLC (Open, High, Low, Close) price data.
        **kwargs: Additional keyword arguments.
            - bias_extraction (bool): If True, applies a threshold bias when determining labels.
            - type_bias (str): Direction of bias, either 'up' or 'down'.
            - shift (int): Number of periods to look ahead for price comparison.
            - bias (float): Threshold value for price movement (used when bias_extraction is True).

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an additional 'LABEL' column containing
        'UP' or 'DOWN' values based on the specified labeling criteria.
    
    Notes:
        - When bias_extraction is True:
          - For 'up' bias: Labels as 'UP' if future price is at least (1 + bias) times current price.
          - For 'down' bias: Labels as 'DOWN' if current price is at least (1 + bias) times future price.
        - When bias_extraction is False:
          - Simple directional comparison between current and future prices.
    """
    
    data = data_ohlc.copy(deep=True)

    if kwargs['bias_extraction']:
        if kwargs['type_bias'] == 'up':
            returns = data.Open.shift(-kwargs['shift']) / data.Open
            data['LABEL'] = np.where((returns >= 1 + kwargs['bias']), 'UP', 'DOWN')
        elif kwargs['type_bias'] == 'down':
            returns = data.Open / data.Open.shift(-kwargs['shift'])
            data['LABEL'] = np.where((returns >= 1 + kwargs['bias']), 'DOWN', 'UP')
    else:
        if kwargs['type_bias'] == 'up':
            data['LABEL'] = np.where(data.Open.shift(-kwargs['shift']) > data.Open, 'UP', 'DOWN')
        elif kwargs['type_bias'] == 'down':
            data['LABEL'] = np.where(data.Open > data.Open.shift(-kwargs['shift']), 'DOWN', 'UP')

    return data
