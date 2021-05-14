import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from get_stock_price import StockDatabase
from utils import get_previous_workday_intraday_datetime, get_next_workday_intraday_datetime
from utils import py_workdays
from portfolio.trade_transformer import DataSupplyUnit


class PriceSuppliier(metaclass=ABCMeta):
    @abstractmethod
    def reset(self, start_datetime, window):
        pass
    
    @abstractmethod
    def step(self):
        pass


class CannotGetAllDataError(Exception):
    def __init__(self, strings):
        self.strings = strings
    def __str__(self):
        return self.strings

class StockDBPriceSupplier(PriceSuppliier):
    """
    StockDatabaseに対応するPriceSupplier
    """
    def __init__(self, stock_db, ticker_names, episode_length, freq_str, interpolate=True):
        """
        stock_db: get_stock_price.StockDatabase
            利用するStockDatabaseクラス
        ticker_names: list of str
            利用する銘柄名のリスト
        episode_length: int
            エピソードの長さ
        freq_str: str
            利用する周期を表す文字列
        interpolate: bool
            補間するかどうか
        """
        self.stock_db = stock_db
        self.ticker_names = ticker_names
        self.episode_length = episode_length
        self.freq_str = freq_str        
        self.interpolate = interpolate
        
    def reset(self, start_datetime, window=np.array([0])):
        """
        Parameters
        ----------
        start_datetime: datetime.datetime 
            データ供給の開始時刻
        window: ndarray
            データ供給のウィンドウ
        
        Returns
        -------
        DatasupplyUnit
            提供する価格データ
        bool
            エピソードが終了したかどうか
        """
        # 終了時刻を求める
        # 全datetimeデータを保持
        assert 0 in window
        if not isinstance(window, np.ndarray):
            self.window = np.array(window)
        else:
            self.window = window
        
        min_window = min(self.window)
        max_window = max(self.window)
        
        if min_window <= 0:
            episode_start_datetime = get_previous_workday_intraday_datetime(start_datetime, self.freq_str, abs(min_window))
        else:
            episode_start_datetime = get_next_workday_intraday_datetime(start_datetime, self.freq_str, min_window)
            
        if self.episode_length+max_window <= 0:  # 基本的にあり得ない
            episode_end_datetime = get_previous_workday_intraday_datetime(start_datetime, self.freq_str, abs(self.episode_length+max_window))
        else:
            episode_end_datetime = get_next_workday_intraday_datetime(start_datetime, self.freq_str, self.episode_length+max_window)
        
        episode_df = self.stock_db.search_span(stock_names=self.ticker_names,
                                               start_datetime=episode_start_datetime,
                                               end_datetime=episode_end_datetime,
                                               freq_str=self.freq_str,
                                               is_end_include=True,  # 最後の値も含める
                                               to_tokyo=True,  #必ずTrueに
                                              )
        
        self.episode_df = py_workdays.extract_workdays_intraday_jp(episode_df)
        
        # 各OHLCVに対応するboolを求めておく
        column_names_array = self.episode_df.columns.values.astype(str)
        self.open_bool_array = np.char.startswith(column_names_array, "Open")
        self.high_bool_array = np.char.startswith(column_names_array, "High")
        self.low_bool_array = np.char.startswith(column_names_array, "Low")
        self.close_bool_array = np.char.startswith(column_names_array, "Close")
        self.volume_bool_array = np.char.startswith(column_names_array, "Volume")
        
        
        all_datetime_index = pd.date_range(start=episode_start_datetime,
                                           end=episode_end_datetime,
                                           freq=self.freq_str,
                                           closed="left"
                                          )
        self.all_datetime_index = py_workdays.extract_workdays_intraday_jp_index(all_datetime_index)
        
        # episode_dfの補間
        if self.interpolate:
            add_datetime_bool = ~self.all_datetime_index.isin(self.episode_df.index)
            
            # 補間を行う数が20%を越えた場合
            if (add_datetime_bool.sum()/len(self.all_datetime_index)) > 0.1:
                err_str = "Interpolate exceeds 10 % about tickers={}, datetimes[{},{}]".format(self.ticker_names,
                                                                                               episode_start_datetime,
                                                                                               episode_end_datetime)
                raise CannotGetAllDataError(err_str)
            
            add_datetime_index = self.all_datetime_index[add_datetime_bool]
            # Noneのdfを作成
            nan_df = pd.DataFrame(index=add_datetime_index, columns=self.episode_df.columns)

            # Noneのdfを追加
            self.episode_df = self.episode_df.append(nan_df)
            self.episode_df.sort_index(inplace=True)
            
            # np.nanの補間
            self.episode_df.interpolate(limit_direction="both",inplace=True)
        else:
            share_index_bool = self.all_datetime_index.isin(self.episode_df.index)
            self.all_datetime_index = self.all_datetime_index[share_index_bool]
        
        # dfをndarrayに変更
        self.episode_df_values = self.episode_df.values.astype(float)
        del self.episode_df
        
        self.all_datetime_index_values = self.all_datetime_index.to_pydatetime()
        del self.all_datetime_index
        
        # データが正しく取得できたかどうか
        if np.isnan(self.episode_df_values).sum() > 0:
            err_str = "PriceSupplier cannot get {} data  about tickers={}, datetimes[{},{}]".format(
                np.isnan(self.episode_df_values).sum(),
                self.ticker_names,
                episode_start_datetime,
                episode_end_datetime)
  
            raise CannotGetAllDataError(err_str)
        
        
        # データの取得
        self.now_index = abs(min_window)
        now_datetime = self.all_datetime_index_values[self.now_index]
        
        add_window = self.now_index + self.window
        window_data_value = self.episode_df_values[add_window,:]
        
        open_array = window_data_value[:,self.open_bool_array].T
        high_array = window_data_value[:,self.high_bool_array].T
        low_array = window_data_value[:,self.low_bool_array].T
        close_array = window_data_value[:,self.close_bool_array].T
        volume_array = window_data_value[:,self.volume_bool_array].T
        
        open_array = np.concatenate([np.ones((1, open_array.shape[1])), open_array], axis=0)
        high_array = np.concatenate([np.ones((1, high_array.shape[1])), high_array], axis=0)
        low_array = np.concatenate([np.ones((1, low_array.shape[1])), low_array], axis=0)
        close_array = np.concatenate([np.ones((1, close_array.shape[1])), close_array], axis=0)
        volume_array = np.concatenate([np.ones((1, volume_array.shape[1])), volume_array], axis=0)
        
        
        out_ticker_names = ["yen"]
        out_ticker_names.extend(self.ticker_names)
        
        out_unit = DataSupplyUnit(names=out_ticker_names,
                                  key_currency_index=0,
                                  datetime=now_datetime,
                                  window=self.window,
                                  open_array=open_array,
                                  close_array=close_array,
                                  high_array=high_array,
                                  low_array=low_array,
                                  volume_array=volume_array
                                 )
        done = False
        return out_unit, done
    
    def step(self):
        """
        Returns
        -------
        DatasupplyUnit
            提供する価格データ
        bool
            エピソードが終了したかどうか
        """
        # indexの更新
        self.now_index += 1
        now_datetime = self.all_datetime_index_values[self.now_index]
        
        add_window = self.now_index + self.window
        window_data_value = self.episode_df_values[add_window,:]
        
        open_array = window_data_value[:,self.open_bool_array].T
        high_array = window_data_value[:,self.high_bool_array].T
        low_array = window_data_value[:,self.low_bool_array].T
        close_array = window_data_value[:,self.close_bool_array].T
        volume_array = window_data_value[:,self.volume_bool_array].T
         
        open_array = np.concatenate([np.ones((1, open_array.shape[1])), open_array], axis=0)
        high_array = np.concatenate([np.ones((1, high_array.shape[1])), high_array], axis=0)
        low_array = np.concatenate([np.ones((1, low_array.shape[1])), low_array], axis=0)
        close_array = np.concatenate([np.ones((1, close_array.shape[1])), close_array], axis=0)
        volume_array = np.concatenate([np.ones((1, volume_array.shape[1])), volume_array], axis=0)
        
        out_ticker_names = ["yen"]
        out_ticker_names.extend(self.ticker_names)
        
        out_unit = DataSupplyUnit(names=out_ticker_names,
                                  key_currency_index=0,
                                  datetime=now_datetime,
                                  window=self.window,
                                  open_array=open_array,
                                  close_array=close_array,
                                  high_array=high_array,
                                  low_array=low_array,
                                  volume_array=volume_array
                                 )
        done = self.now_index >= self.episode_length
        
        return out_unit, done 

if __name__ == "__main__":
    pass