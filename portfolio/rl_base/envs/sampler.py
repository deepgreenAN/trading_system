from numpy.random import RandomState
import numpy as np
import pandas as pd
from scipy.special import softmax

import py_workdays

from utils import middle_sample_type_with_check

class ConstSamper:
    """
    定数をサンプリング値として取得するためのクラス
    """
    def __init__(self, const_object):
        """
        const_object: any
            定数としてサンプリングされる値
        """
        self.const_object = const_object
        
    def sample(self, seed=None):
        """
        seted: int
            ランダムシード
        """
        return self.const_object


class TickerSampler:
    """
    ticker_nameをサンプリングするためのクラス
    """
    def __init__(self, all_ticker_names, sampling_ticker_number):
        """
        all_ticker_names: list of str
            サンプリングを行う銘柄名のリスト
        sampling_ticker_number: int
            サンプリング数
        """
        self.all_ticker_names = all_ticker_names
        self.sampling_ticker_number = sampling_ticker_number

    def sample(self, seed=None):
        """
        Parameters
        ----------
        seed: int
            ランダムシード
            
        Returns
        -------
        list of str 
            サンプリングされた銘柄名のndarray
        datetime.datetime
        """
        random_ticker_names = RandomState(seed).choice(self.all_ticker_names, self.sampling_ticker_number, replace=False)  # 重複を許さない
        return random_ticker_names


class DatetimeSampler:
    """
    datetime.datetimeをサンプリングするためのクラス
    """
    def __init__(self, start_datetime, end_datetime, episode_length, freq_str):
        """
        start_datetime: datetime.datetime
            サンプリングする範囲の開始日時
        end_datetime: datetime.datetime
            サンプリングする範囲の終了日時
        episode_length: int
            エピソード長
        freq_str: str
            サンプリング周期を表す文字列
        """
        
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        self.freq_str = middle_sample_type_with_check(freq_str)


        all_datetime_index = pd.date_range(start=self.start_datetime,
                                           end=self.end_datetime,
                                           freq=self.freq_str,
                                           closed="left"
                                          )  
        
        self.all_datetime_value = py_workdays.extract_workdays_intraday_index(all_datetime_index).to_pydatetime()
        
        self.all_datetime_index_range = np.arange(0, len(self.all_datetime_value))
        self.episode_length = episode_length
        
    def sample(self, seed=None, window=np.array([0])):
        """
        Parameters
        ----------
        seed: int
            ランダムシード
            
        window: list of int
            ウィンドウを表すリスト
        
        Returns
        -------
        list of str
        datetime.datetime
            サンプリングされた日時
        """
        
        if not isinstance(window, np.ndarray):
            self.window = np.array(window)
        else:
            self.window = window
        
        min_window = min(self.window)
        max_window = max(self.window)
        
        random_datetime = self.all_datetime_value[RandomState(seed).choice(self.all_datetime_index_range[abs(min_window):-(max_window+self.episode_length)],1)].item()
        
        return random_datetime 


class PortfolioVectorSampler:
    """
    ポートフォリオベクトルをサンプリングするためのクラス
    """
    def __init__(self, vector_length):
        """
        vector_length: int
            ポートフォリオベクトルの長さ
        """
        self.vector_length = vector_length
    def sample(self, seed=None):
        """
        seed: int
            ランダムシード
        """
        portfolio_vector = softmax(RandomState(seed).randn(self.vector_length))
        return portfolio_vector


class MeanCostPriceSampler:
    """
    平均取得価格をサンプリングするためのクラス
    """
    def __init__(self, mean_array=None, var_array=None):
        """
        mean_array: np.ndarray
            平均ベクトル
        var_array: np.ndarray
            分散ベクトル
        """
        self.mean_array = mean_array
        self.var_array = var_array
        
    def sample(self, seed=None):
        """
        seed: int
            ランダムシード
        """
        mean_cost_price = self.mean_array + self.var_array * RandomState(seed).randn(self.vector_length)
        return mean_cost_price


class SamplerManager:
    """
    TradeEnv環境のサンプリングを担うクラス．利用するサンプリングが追加・変更された場合．こちらを変更する．
    """
    def __init__(self, 
                 ticker_names_sampler,
                 datetime_sampler,
                 portfolio_vector_sampler=ConstSamper(None),
                 mean_cost_price_array_sampler=ConstSamper(None)
                ):
        """
        ticker_names_sampler:
            銘柄名をサンプリングするクラス
        datetime_sampler:
            日時をサンプリングするクラス
        portfolio_vector_samper:
            ポートフォリオベクトルをサンプリングするクラス
        mean_cost_price_array_sampler
            平均取得価格をサンプリングするクラス
        """
        
        self.ticker_names_sampler = ticker_names_sampler
        self.datetime_sampler = datetime_sampler
        self.portfolio_vector_sampler = portfolio_vector_sampler
        self.mean_cost_price_array_sampler = mean_cost_price_array_sampler


if __name__ == "__main__":
    pass

    




