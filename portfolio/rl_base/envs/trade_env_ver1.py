from numpy.random import RandomState
import numpy as np
import pandas as pd

import py_workdays

from utils import middle_sample_type_with_check


class TickerSampler:
    """
    ticker_nameをサンプリングするためのクラス
    """
    def __init__(self, all_ticker_names, sampling_ticker_number, select_datetime=None):
        """
        all_ticker_names: list of str
            サンプリングを行う銘柄名のリスト
        sampling_ticker_number: int
            サンプリング数
        select_datetime: datetime.datetime
            指定する日時．サンプリングは行わないのでこのまま返る
        """
        self.all_ticker_names = all_ticker_names
        self.sampling_ticker_number = sampling_ticker_number
        self.select_datetime = select_datetime

    def sample(self, seed=None, *args):
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
        return random_ticker_names, self.select_datetime


class DatetimeSampler:
    """
    datetime.datetimeをサンプリングするためのクラス
    """
    def __init__(self, start_datetime, end_datetime, episode_length, freq_str, ticker_names=None):
        """
        start_datetime: datetime.datetime
            サンプリングする範囲の開始日時
        end_datetime: datetime.datetime
            サンプリングする範囲の終了日時
        episode_length: int
            エピソード長
        freq_str: str
            サンプリング周期を表す文字列
        ticker_names: list of str
            指定する銘柄名のリスト．サンプリングしないのでこのまま返る
        """
        self.ticker_names = ticker_names
        
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        self.freq_str = middle_sample_type_with_check(freq_str)


        all_datetime_index = pd.date_range(start=self.start_datetime,
                                           end=self.end_datetime,
                                           freq=self.freq_str,
                                           closed="left"
                                          )  
        
        self.all_datetime_value = py_workdays.extract_workdays_intraday_jp_index(all_datetime_index).to_pydatetime()
        
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
        
        return self.ticker_names, random_datetime


class TickerDatetimeSampler():
    """
    指定した範囲から，エピソード分利用可能なdatetimeをサンプリングするためのクラス
    """
    def __init__(self, all_ticker_names, sampling_ticker_number, start_datetime, end_datetime, episode_length, freq_str):
        """
        all_ticker_names: list of str
            サンプリングを行う銘柄名のリスト
        sampling_ticker_number: int
            銘柄名のサンプリング数
        start_datetime: datetime.datetime
            サンプリングする範囲の開始日時
        end_datetime: datetime.datetime
            サンプリングする範囲の終了日時
        episode_length: int
            エピソード長
        freq_str: str
            サンプリング周期を表す文字列
        """
        self.ticker_sampler = TickerSampler(all_ticker_names, sampling_ticker_number, select_datetime=None)
        self.datetime_sampler = DatetimeSampler(start_datetime, end_datetime, episode_length, freq_str, ticker_names=None)

        
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
            サンプリングされた銘柄名のndarray
        datetime.datetime
            サンプリングされた日時
        """
        random_ticker_names, _ = self.ticker_sampler.sample(seed=seed)
        _, random_datetime = self.datetime_sampler.sample(seed=seed, window=window)
        
        return random_ticker_names, random_datetime


class TradeEnv:
    """
    PortfolioStateを利用して基本的な売買を行う強化学習用の環境
    """
    def __init__(self, 
                 portfolio_transformer,
                 sampler,
                 window=[0],
                 fee_const=0.0025,
                ):
        """
        portfolio_transformer: PortfolioTransformer
             ポートフォリオを遷移させるTransformer
        sampler: Sampler
            銘柄と日時をサンプリングするためのサンプラー
        window: list
            PortfolioStateのウィンドウサイズ
        fee_const: float
            単位報酬・取引量当たりの手数料
        """
        self.portfolio_transformer = portfolio_transformer
        self.sampler = sampler
        self.window = window
        self.fee_const = fee_const
        
    def reset(self, seed=None):
        """
        Parameters
        ----------
        seed: int
            乱数シード
            
        Returns
        -------
        PortfolioState
            遷移したPortfolioStateのインスタンス
        float
            報酬．resetなので0
        bool
            終了を示すフラッグ
        dict
            その他の情報
        """
        random_ticker_names, random_datetime = self.sampler.sample(seed=seed, window=self.window)  # 銘柄名,日時のサンプリング
        self.portfolio_transformer.price_supplier.ticker_names = list(random_ticker_names)  # 銘柄名の変更
        portfolio_state, done = self.portfolio_transformer.reset(random_datetime, window=self.window)
        
        self.portfolio_state = portfolio_state
        
        return self.portfolio_state.copy(), 0, done, None
    
    def step(self, portfolio_vector):
        """
        Parameters
        ----------
        portfolio_vector: ndarray
            actionを意味するポートフォリオベクトル
            
        Returns
        -------
        PortfolioState
            遷移したPortfolioStateのインスタンス
        float
            報酬．
        bool
            終了を示すフラッグ
        dict
            その他の情報
        """        

        previous_portfolio_state = self.portfolio_state
        
        #状態遷移
        portfolio_state, done = self.portfolio_transformer.step(portfolio_vector)
        
        #報酬の計算
        portfolio_vector = portfolio_state.portfolio_vector
        
        price_change_ratio = portfolio_state.now_price_array / previous_portfolio_state.now_price_array  # y
        raw_reward_ratio = np.dot(portfolio_vector, price_change_ratio)  # r
        
        portfolio_change_vector = portfolio_vector - previous_portfolio_state.portfolio_vector #W_{t}-w_{t-1}
        reward = np.log(raw_reward_ratio*(1-self.fee_const*np.dot(portfolio_change_vector, portfolio_change_vector)))
        
        return portfolio_state.copy(), reward, done, None

if __name__ == "__main__":
    pass