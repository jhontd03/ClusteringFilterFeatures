from datetime import datetime, timedelta
import datetime as dt
from typing import Optional
import pandas as pd
import yfinance as yf
import MetaTrader5 as mt5

class DataLoader:
    """Handles loading and preprocessing of OHLC candlestick data."""
    
    def __init__(self):
        self.data = None
                
    def load_from_csv(self, filepath: str, date_column: str = 'Date') -> None:
        """
        Load OHLC data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            date_column: Name of the date column
        """
        try:
            self.data = pd.read_csv(filepath)
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data.set_index(date_column, inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to load data from csv file: {str(e)}")

    def load_from_yfinance(self, 
                           symbol: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           period: Optional[str] = None,
                           interval: str = '1d') -> None:
        """
        Load OHLC data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Alternative to start/end dates. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                self.data = ticker.history(period=period, interval=interval)
            else:
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
                    
                self.data = ticker.history(start=start_date, end=end_date, interval=interval)
                
            # Ensure consistent column names
            self.data.index.name = 'Date'
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            raise ValueError(f"Failed to load data from Yahoo Finance: {str(e)}")

    def load_from_MT5(self,
                      symbol: str,
                      start_date: str,
                      end_date: str = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d"),
                      time_frame: str = '1h') -> pd.DataFrame:
        
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        try:
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
            start_utc = start_date.astimezone(dt.timezone.utc)

            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
            end_utc = end_date.astimezone(dt.timezone.utc)
            time_frame = map_timeframes(time_frame)

            rates = mt5.copy_rates_range(symbol,
                                         time_frame,
                                         start_utc,
                                         end_utc)

            self.data = pd.DataFrame(rates)

            self.data['date'] = pd.to_datetime(self.data['time'], unit='s')
            self.data = self.data.set_index(self.data['date'])
            self.data = self.data.loc[:, ['open', 'high', 'low', 'close', 'spread']]
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Spread']
            mt5.shutdown()

        except Exception as e:
            raise ValueError(f"Failed to load data from Metatrader: {str(e)}")            

    def get_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.data.copy() 

def map_timeframes(time_frame: str) -> int:

    timeframe_mapping = {
        '1m': mt5.TIMEFRAME_M1,
        '2m': mt5.TIMEFRAME_M2,                        
        '3m': mt5.TIMEFRAME_M3,                        
        '4m': mt5.TIMEFRAME_M4,                        
        '5m': mt5.TIMEFRAME_M5,                        
        '6m': mt5.TIMEFRAME_M6,                        
        '10m': mt5.TIMEFRAME_M10,                       
        '12m': mt5.TIMEFRAME_M12,
        '15m': mt5.TIMEFRAME_M15,
        '20m': mt5.TIMEFRAME_M20,                       
        '30m': mt5.TIMEFRAME_M30,                       
        '1h': mt5.TIMEFRAME_H1,                          
        '2h': mt5.TIMEFRAME_H2,                          
        '3h': mt5.TIMEFRAME_H3,                          
        '4h': mt5.TIMEFRAME_H4,                          
        '6h': mt5.TIMEFRAME_H6,                          
        '8h': mt5.TIMEFRAME_H8,                          
        '12h': mt5.TIMEFRAME_H12,
        '1d': mt5.TIMEFRAME_D1,                       
        '1w': mt5.TIMEFRAME_W1,                       
        '1M': mt5.TIMEFRAME_MN1,                       
    }

    try:
        return timeframe_mapping[time_frame]
    except:
        print(f"Timeframe {time_frame} is not valid.")    