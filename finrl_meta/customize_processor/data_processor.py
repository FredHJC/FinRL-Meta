from finrl_meta.customize_processor.processor_binance import BinanceProcessor as Binance
import pandas as pd
import numpy as np
import os

class DataProcessor():
    def __init__(self, data_source):
        self.data_source = data_source
        if self.data_source =='binance':
            try:
                self.processor = Binance(data_source)
                print('Binance successfully connected')
            except:
                raise ValueError('Please input correct account info for binance!')
        else:
            raise ValueError('Data source input is NOT correct.')
    
    def download_data(self, ticker_list, start_date, end_date, 
                      time_interval) -> pd.DataFrame:
        df = self.processor.download_data(ticker_list = ticker_list, 
                                          start_date = start_date, 
                                          end_date = end_date,
                                          time_interval = time_interval)
        return df
    
    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        
        return df
    
    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df
    
    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(df,
                                                    self.tech_indicator_list,
                                                    if_vix)
        #fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        
        return price_array, tech_array, turbulence_array
    
    def run(self, ticker_list, start_date, end_date, time_interval, 
            technical_indicator_list, if_vix, cache=False):
        
        cache_csv = '_'.join(ticker_list + [self.data_source, start_date, end_date, time_interval]) + '.csv'
        cache_dir = './cache'
        cache_path = os.path.join(cache_dir, cache_csv)

        if cache and os.path.isfile(cache_path):
            print('Using cached file {}'.format(cache_path))
            self.tech_indicator_list = technical_indicator_list
            data = pd.read_csv(cache_path)
        
        else:
            data = self.download_data(ticker_list, start_date, end_date, time_interval)
            data = self.clean_data(data)
            if cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                data.to_csv(cache_path)
        data = self.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = self.add_vix(data)

        price_array, tech_array, turbulence_array = self.df_to_array(data, if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

