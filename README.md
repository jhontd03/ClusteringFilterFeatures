# Descorrelation and optimal selection of the periods of a technical indicator

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Author](#author)

## Introduction

This repository implements a clustering-based approach to optimize technical indicators for trading strategies. The main focus is on using distance correlation and clustering techniques to identify the most effective indicator parameters, particularly for RSI (Relative Strength Index) periods.

The project aims to:
- Identify optimal indicator parameters through clustering analysis
- Reduce parameter space complexity
- Evaluate indicator effectiveness using distance correlation
- Support both long and short trading strategies

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/technical-indicator-clustering.git
cd technical-indicator-clustering
pip install -r requirements.txt
```

## Requirements

The project requires Python 3.11.9 and the following key dependencies:

- MetaTrader5
- pandas
- pandas_ta
- scikit-learn
- yellowbrick
- dcor
- seaborn

For a complete list of dependencies, see `requirements.txt`.

## Usage

Here's a basic example of how to use the clustering analysis:

```python
from data_loader import DataLoader
from function_cluster import KMeansClustering

# Configure parameters
config = {
    'symbol': 'EURUSD',
    'time_frame': '1d',
    'signal_direction': 'short',
    'bias_extraction': True,
    'shift': 1,
    'bias': 0.001
}

# Load data
data_loader = DataLoader()
data_loader.load_from_MT5(symbol=config['symbol'], 
                         start_date='2014-01-01',
                         end_date='2024-01-01',
                         time_frame=config['time_frame'])

# Perform clustering analysis
kmeans = KMeansClustering()
cluster_labels = kmeans.fit_predict(data)
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── main.py                # Main script for running the analysis
├── data_loader.py         # Data loading utilities (MT5, CSV, Yahoo Finance)
├── data_labeling.py       # Price movement labeling functions
├── function_cluster.py    # Clustering implementations (KMeans, GMM, Agglomerative)
└── helper_functions.py    # Utility functions for correlation and clustering
```

## Features

- **Multiple Data Sources**:
  - MetaTrader 5 integration
  - Yahoo Finance support
  - CSV file loading

- **Clustering Methods**:
  - K-means clustering
  - Gaussian Mixture Models (GMM)
  - Agglomerative clustering

- **Technical Analysis**:
  - RSI period optimization
  - Distance correlation analysis
  - Automatic cluster size optimization

- **Trading Strategy Support**:
  - Long and short signal generation
  - Customizable bias thresholds
  - Multiple timeframe analysis

## Author

Jhon Jairo Realpe

jhon.td.03@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.
