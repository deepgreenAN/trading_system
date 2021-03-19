import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
import pandas as pd
import datetime
from collections import namedtuple
from copy import deepcopy

from pytz import timezone
from pathlib import Path

import gym
from gym import spaces, logger

import sys
sys.path.append(r"E:\システムトレード入門\trade_system_git_workspace")

from get_stock_price import StockDatabase
from utils import middle_sample_type_with_check, get_sec_from_freq, get_workdays_jp, get_previous_datetime, get_next_workday_jp
from utils import extract_workdays_intraday_jp_index, extract_workdays_intraday_jp


field_list = ["cash", "unit_number", "mean_cost_price", "all_property", "price_array"]
StockStateBase = namedtuple("StockStateBase", field_list)

class StockState(StockStateBase):
    def to_numpy(self):
        """
        ndarrayに変更する．最後に利用するのがいい？
        """
        cash_unit_mean_all_array = np.array([self.cash, self.unit_number, self.mean_cost_price, self.all_property])
        return np.concatenate([cash_unit_mean_all_array, self.price_array.copy()], axis=0)  # コピーすることに注意
    
    @classmethod
    def from_numpy(cls, stock_state_array):
        """
        ndarrayからこのクラスを作成する．基本的に使わない．
        """
        cash = stock_state_array[0]
        unit_number = stock_state_array[1]
        mean_cost_price = stock_state_array[2]
        all_property = stock_state_array[3]
        price_array = stock_state_array[4:]
        
        return cls(cash=cash, unit_number=unit_number, mean_cost_price=mean_cost_price, all_property=all_property, price_array=price_array)

    def copy(self):
        """
        deepcopyでもいいが，一応明示しておく
        """
        new_state = StockState(cash=self.cash,
                               unit_number=self.unit_number,
                               mean_cost_price=self.mean_cost_price,
                               all_property=self.all_property,
                               price_array=self.price_array.copy())
        return new_state
    
    @property
    def now_price(self):
        return self.price_array[0]


class NormalizeState():
    def __init__(self, cash_const=1, unit_const=1, price_const=1, all_property_const=1):
        self.cash_devide_const = cash_const
        self.unit_devide_const = unit_const
        self.price_devide_const = price_const
        self.all_property_const = all_property_const
        
    def __call__(self, state):
        # cash
        cash = state.cash / self.cash_devide_const
        # unit_number
        unit_number = state.unit_number / self.unit_devide_const
        # price_array
        mean_cost_price = state.mean_cost_price / self.price_devide_const
        price_array = state.price_array / self.price_devide_const
        # all_property
        all_property = state.all_property / self.all_property_const
        
        new_state = state._replace(cash=cash, 
                                   unit_number=unit_number, 
                                   mean_cost_price=mean_cost_price,
                                   all_property=all_property,
                                   price_array=price_array)  # 新しいオブジェクト 
        
        return new_state

class NormalizeReward():
    def __init__(self, reward_const=1):
        self.reward_devide_const = reward_const
        
    def __call__(self, reward):
        reward = reward / self.reward_devide_const
        
        return reward


class WindowSampler():
    """
    エピソードの開始時刻をサンプリングするサンプラー．領域内をエピソード長さで分割する訳ではなく，エピソードで利用する
    データをwindowとしてずらしながら取得する．
    """
    def __init__(self, stock_names, start_datetime, end_datetime, freq_str, end_include=False):
        """
        指定した範囲から，エピソード分利用可能なdatetimeをサンプリングするためのクラス
        """
        if isinstance(stock_names, str):  # 銘柄コードが一つの場合
            stock_names = [stock_names]
        self.stock_names = stock_names
        
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        # タイムゾーンが同じかチェック
        if self.start_datetime.tzinfo != self.end_datetime.tzinfo:
            raise Exception("The timezone of start_datetime and end_datetime must be same")
        
        self.freq_str = middle_sample_type_with_check(freq_str)
                
        stock_name_datetime_array_list = []
        
        for stock_name in stock_names:
            # 指定した範囲内，周期のdatetimeのndarrayを取得
            if end_include:
                datetime_index = pd.date_range(start=self.start_datetime,
                                               end=self.end_datetime,
                                               freq=self.freq_str
                                              )
                
            else:
                datetime_index = pd.date_range(start=self.start_datetime,
                                               end=get_previous_datetime(self.end_datetime, freq_str=self.freq_str),
                                               freq=self.freq_str
                                              )

            datetime_index = extract_workdays_intraday_jp_index(datetime_index, return_as="index")
            
            stock_name_array = np.full(len(datetime_index), stock_name).astype(object)  # object同士でstackするため
            datetime_array = datetime_index.to_pydatetime()
            stock_name_datetime_array = np.stack([stock_name_array, datetime_array], axis=1)  # [銘柄コード, datetime]のndarray
            stock_name_datetime_array_list.append(stock_name_datetime_array)
            
    
        self.all_stockname_datetime_array = np.concatenate(stock_name_datetime_array_list, axis=0)
        
    def sample(self, seed=None, remove=False):
        """
        seed: None or int
            ランダムシード
        remove: bool
            サンプリングされた値をリストから取り除くかどうあ
        """
        if seed is not None and remove:
            raise Exception("If use seed, cannot remove")
            
        random_index = RandomState(seed).randint(0, len(self.all_stockname_datetime_array))
        #ランダムに取り出す
        stock_datetime = self.all_stockname_datetime_array[random_index]
        # 取り出したものを削除
        if remove:
            self.all_stockname_datetime_array = np.delete(self.all_stockname_datetime_array,random_index,axis=0)
        return stock_datetime


class OneStockEnv(gym.Env):
    """
    一つの銘柄の株式に関する環境
    """
    def __init__(self, 
                 stock_db,
                 stock_names,
                 start_datetime,
                 end_datetime,
                 freq_str,
                 episode_length,
                 state_time_list=[0],
                 use_ohlc="Close",
                 initial_cash=1.e6,  # 100万円
                 initial_unit=50,  # 初期単元数
                 use_view=False,
                 one_unit_stocks=100,
                 max_units_number=10,
                 low_limmit=1.e3,  # 1000円
                 interpolate=True,
                 sample_remove=False,
                 action_restriction = False,
                 stay_penalty_unit_bound=20,
                 stay_penalty_cash_bound=1.e5,
                 penalty_mcp_np_diff_bound=3
                ):
        """
        stock_db: StockDataBase
            値を読み込むデータベース．1エピソード分のデータをreset事に読み込む
        stock_names: str list
            利用するデータの銘柄，複数指定した場合はreset時にランダムで決定される
        start_datetime: datetime
            利用するデータの始端
        ed_datetime: datetime
            利用するデータの終端
        freq_str: str
            サンプリング周期
        episode_length: int
            1エピソードの状態の個数
        start_time_list: int list
            状態として利用する株価の現在からのインデックスのリスト．最初の値は現在(0)でなければならない
        use_ohlc: str, defalt: "Close"
            サンプリング周期における利用するデータの指定(Open, High, Low, Close)
        initial_cash: int, defalt: 1.e6
            種銭
        use_view: bool
            reset時にviewを使うかどうか．使う場合，若干高速になるがcloseをしなければならない
        one_unit_stocks: int
            単元数．通常は100だが，独自に定義できる
        max_units_number: int, default: 10
            売買時の単元のセット数の上限・下限．
        low_limmmit: int, defalt:1.e3
            資産額の下限値．この値を下回った場合，エピソードを終了する．
        interpolate: bool
            指定した期間内のデータ中にnanがあった場合，あるいはデータが取得できなかった場合に補間をするかどうか
        sample_remove: bool
            reset時に取得するデータで一度利用したデータを取り除くかどうか
        """
        
        # データベース
        self.stock_db = stock_db
        # ビューの作成
        self.use_view = use_view
        if self.use_view:
            self.stock_view = self.stock_db.create_view(stock_names, start_datetime=start_datetime, end_datetime=end_datetime)
        
        # 利用するohlcのいずれか
        if use_ohlc not in {"Open","High","Low","Close"}:
            raise Exception("use_ohlc must be in {'Open','High','Low','Close'}")
        self.use_ohlc = use_ohlc
        
        self.freq_str = middle_sample_type_with_check(freq_str)
        # 全datetimeデータを保持
        all_datetime_index = pd.date_range(start=start_datetime,
                                           end=get_previous_datetime(end_datetime, freq_str=self.freq_str),
                                           freq=self.freq_str,
                                          )
        
        self.all_datetime_index = extract_workdays_intraday_jp_index(all_datetime_index, return_as="index")
            
        self.episode_length = episode_length
        self.initial_cash = initial_cash
        self.initial_unit = initial_unit
        self.low_limmit = low_limmit
        self.interpolate = interpolate
        self.sample_remove = sample_remove
        self.action_restriction = action_restriction
        self.stay_penalty_unit_bound = stay_penalty_unit_bound
        self.stay_penalty_cash_bound = stay_penalty_cash_bound
        self.penalty_mcp_np_diff_bound = penalty_mcp_np_diff_bound
        self.penalty_const = 1.e5
        
        # 乱数シード
        self.initial_seed_number = None
        self.mcp_seed_number = None
        
        # 報酬の重み
        self.gein_reward_weight = 0
        self.price_reward_weight = 1
        self.penalty_weight = 1
        self.all_property_reward_weight = 1
        
        if state_time_list[0] != 0:
            raise Exception("The first value of state_time_list must be 0")
        
        if state_time_list[1] != 1:
            raise Exception("The second value of state_time_list must be 1")
        
        self.state_time_list = state_time_list
        
        self.max_state_time = max(state_time_list)  # 利用する未来の状態の現在からのタイムステップ数
        last_start_datetime = self.all_datetime_index[-(self.max_state_time + self.episode_length)]  # エピソードの開始時刻の最後

        self.sampler = WindowSampler(stock_names,
                                     start_datetime=start_datetime,
                                     end_datetime=last_start_datetime,
                                     freq_str=self.freq_str,
                                     end_include=True,  # 最後も含む
                                    )
        # 状態空間と行動空間の定義
        
        min_stock_price = 0
        max_stock_price = np.finfo(np.float32).max
        
        min_cash = 0
        max_cash = np.finfo(np.float32).max
        
        min_unit = 0
        max_unit = 1.e6  # # 100万＊単元まで売買できる 
        
        state_max_list = [max_cash, max_unit, max_stock_price, max_cash]  # 現金，保有単元数，平均取得価格，全資産
        state_max_list.extend([max_stock_price]*len(self.state_time_list))  # [現在価格，指定した未来の価格]
        
        state_min_list = [min_cash, min_unit, min_stock_price, min_cash]  # 現金，保有単元数，平均取得価格，全資産
        state_min_list.extend([min_stock_price]*len(self.state_time_list))  # [現在価格，指定した未来の価格]
                
        state_max_array = np.array(state_max_list, dtype=np.float32)
        state_min_array = np.array(state_min_list, dtype=np.float32)
        
        self.one_unit_stocks = one_unit_stocks  # 独自単元．
        self.max_units_number = max_units_number  # 売買する独自単元の最大数 
        self.action_class_number = 2 * self.max_units_number + 1 # 行動数は 2 * self.max_unit_number + 1(ステイ)
        
        self.action_space = spaces.Discrete(self.action_class_number)
        self.observation_space = spaces.Box(state_min_array, state_max_array, dtype=np.float32)
        
        # 行動クラスの定義
        sell_action_array = np.arange(-self.max_units_number, 0, 1)  # 売りクラス マイナス
        stay_action_arrray = np.array([0])  # ステイクラス
        buy_action_array = np.arange(1, self.max_units_number+1, 1)  # 買いクラス　プラス
        
        self.stay_index = len(sell_action_array)  # ステイのインデックス
        self.action_class_array = np.concatenate([sell_action_array, stay_action_arrray, buy_action_array], axis=0)  # 売りクラス，ステイクラス，買いクラスの順番
        
        self.state = None
        self.steps_beyond_done = None  # エラー処理に利用
        

    def reset(self, select_datetime=None, select_stock_name=None):
        """
        環境をリセットする．エピソード開始日時と銘柄名を指定することで，特定の
        select_datetime: datetime.datetime
            エピソードの開始日時
        select_stock_name: str
            銘柄名
        """
        iter_counter = 0
        while True:  # 条件をクリアしたらreturn,それまで繰り返し
            stock_and_datetime = self.sampler.sample(remove=self.sample_remove, seed=self.initial_seed_number)
            
            if select_stock_name is None:  # 銘柄名が与えられなかった場合
                stock_name = stock_and_datetime[0]
            else:
                if select_stock_name in self.sampler.stock_names:  # 銘柄名の列
                    stock_name = select_stock_name
                else:
                    raise Exception("select_stock must be in stock_names of constructor args")
            
            if select_datetime is None:  # 日時が与えられなかった場合
                episode_start_datetime = stock_and_datetime[1]
            else:
                select_datetime_array = np.array([select_datetime])
                if np.in1d(select_datetime_array, self.sampler.all_stockname_datetime_array[:,1]).item():  # datetimeの列
                    episode_start_datetime = select_datetime
                else:
                    raise Exception("select_datetime must be in sample range of datetime")
                
            episode_start_datetime_index = np.where(self.all_datetime_index==episode_start_datetime)[0].item()  # エピソードで利用する開始時刻

            episode_end_datetime_index = episode_start_datetime_index + (self.episode_length-1) + self.max_state_time  # 状態が利用する未来のデータも含める
            episode_end_datetime = self.all_datetime_index[episode_end_datetime_index]  # エピソードで利用する終端時刻(エピソードの終了時刻ではない)

            # 利用するカラム名
            use_column_name = self.use_ohlc+"_"+stock_name
            
            # エピソード分+状態利用分のデータの取得
        
            if self.use_view:
                episode_series = self.stock_db.search_span(stock_names=stock_name, 
                                                           start_datetime=episode_start_datetime,
                                                           end_datetime=episode_end_datetime,
                                                           freq_str=self.freq_str,
                                                           is_end_include=True,  # 最後の値も含める
                                                           to_tokyo=True,  #必ずTrueに
                                                           view=self.stock_view
                                                           ).loc[:,use_column_name]
                
            else:
                episode_series = self.stock_db.search_span(stock_names=stock_name, 
                                                           start_datetime=episode_start_datetime,
                                                           end_datetime=episode_end_datetime,
                                                           freq_str=self.freq_str,
                                                           is_end_include=True,  # 最後の値も含める
                                                           to_tokyo=True,  #必ずTrueに
                                                           ).loc[:,use_column_name]
            
            # 領域を営業日日中に変更
            
            
            self.episode_series = extract_workdays_intraday_jp(episode_series, return_as="df")
            
            
            
            # 領域にnanが入っているか判定
            if self.episode_series.isnull().sum() > 0:
                IsNan = True
            else:
                IsNan = False
            
            
            # すべて取得できたか判定(データが存在しないか判定)
            part_datetime_bool = (episode_start_datetime <= self.all_datetime_index) & (self.all_datetime_index <= episode_end_datetime)  # 最後も含める
            part_datetime_index = self.all_datetime_index[part_datetime_bool].copy()
                       
            if not self.episode_series.index.equals(part_datetime_index):  # 同一判定
                IsCannotGet = True
            else:
                IsCannotGet = False
            
            iter_counter += 1
            
            # nanが入っておらず，全て取得できた場合．あるいは補間する場合，あるいは日時・もしくはstock_nameが与えられた場合
            if ((not IsNan) and (not IsCannotGet)) or self.interpolate or (select_datetime is not None or stock_name is not None):
                break

        # 線形補間
        if (self.interpolate or (select_datetime is not None or select_stock_name is not None)) and (IsNan or IsCannotGet):  # 補間の必要性がある場合

            if IsCannotGet:  # データが存在しない場合
                # part_datetime_indexに含まれて，self.episode_series.indexに含まれないdatetime
                add_datetime_indice = np.setdiff1d(part_datetime_index.to_pydatetime(), self.episode_series.index.to_pydatetime())
                add_series = pd.Series(index=add_datetime_indice,dtype=np.float64)  # nameは指定しなくてもOK, 値はnan
                new_episode_series = pd.concat([self.episode_series, add_series]).sort_index()  # 存在しないデータがnanで補間
                self.episode_series = new_episode_series
            
            # 線形補間
            self.episode_series.interpolate(limit_direction="both",inplace=True)

                
        cash = self.initial_cash  # 現金
        unit_number = self.initial_unit  # 保有単元数

        # 状態で利用する部分を取得
        #state_datetimes = [self.all_datetime_index[episode_start_datetime_index+add_index].to_pydatetime() for add_index in self.state_time_list]  # datetime.datetimeに変更
        state_datetimes = [self.all_datetime_index[episode_start_datetime_index+add_index] for add_index in self.state_time_list]  # pd.Timestampのリスト
        
        state_prices_bool_array = self.episode_series.index.isin(state_datetimes)

        state_prices_series = self.episode_series.loc[state_prices_bool_array].copy()
        state_prices_array = state_prices_series.values
        
        now_price = state_prices_array[0]  # 最初は現在の価格

        #mean_cost_price = now_price  # unit数にかかわらず，最初の平均取得価格は現在価格(unit数が0のときも含める)
        #random_index = RandomState(self.mcp_seed_number).randint(0, len(self.episode_series.index))  # 既存のseedと区別するため，+1
        #mean_cost_price = self.episode_series.loc[self.episode_series.index[random_index]]  # 平均取得価格はランダムに決める
        mean_cost_price = now_price + 5 * RandomState(self.mcp_seed_number).randn(1).item()  # intにするため
        
        all_property = cash + now_price * unit_number * self.one_unit_stocks
        
        # 現状態をアトリビュートとして保持
        self.state = StockState(cash=cash,
                                unit_number=unit_number,
                                mean_cost_price=mean_cost_price,
                                all_property=all_property,
                                price_array=state_prices_array 
                               )
        
        self.now_datetime = episode_start_datetime  # 現時間
        self.info = {"datetime":self.now_datetime,
                     "stock_name":stock_name,
                     "done_active":None,
                     "iter_counter":iter_counter,
                     "penalty":0,
                     "gein_reward":0,
                     "price_reward":0,
                     "all_property_reward":0,
                     "reward":0,
                    }
        
        self.now_index = episode_start_datetime_index  # 現時間のインデックス
        self.stock_name = stock_name  # 現銘柄コード
        self.step_counter = 1  # ステップ数のカウンタ
        done = False  # ここではもちろんFalse
        
        # エラー処理で利用
        self.steps_beyond_done = None
        
        return self.state.copy(), 0, done, self.info  # 一応stateをコピー
    
    def step(self, action):
        #assert self.action_space.contains(action), "action {} ({}) invalid".format(action, type(action))
        
        state = self.state
        
        cash = state.cash
        unit_number = state.unit_number
        mean_cost_price = state.mean_cost_price
        all_property = state.all_property
        prices = state.price_array  # 使わない？
        
        # インデックスを更新
        now_index = self.now_index + 1
        now_datetime = self.all_datetime_index[now_index].to_pydatetime()  # infoで利用
        
        state_datetimes = [self.all_datetime_index[now_index+add_index] for add_index in self.state_time_list]  # pd.Timestampのまま
        
        state_prices_bool_array = self.episode_series.index.isin(state_datetimes)

        state_prices_series = self.episode_series.loc[state_prices_bool_array].copy()
        state_prices_array = state_prices_series.values
        
        # 行動の制約条件（現金より多くかえない，保持数より多くは売れない)
        now_price = state_prices_array[0]  # state_priceの最初は必ずnowでなければならない
        cliped_cash = max(0, cash)  # 条件判定はマイナスを除く
        buy_condition_bool = (cliped_cash - now_price * self.action_class_array * self.one_unit_stocks) >= 0  # 買い条件を満たしたブール, キャッシュがマイナスの時，0にする
        buy_condition_true_max_index = buy_condition_bool.sum() - 1  # 条件を満たした最大値のインデックス
        if buy_condition_true_max_index < 0:  # -1になってしまった場合
            buy_condition_true_max_index = 0

        cliped_unit_number = max(0, unit_number)
        sell_condition_bool = (cliped_unit_number + self.action_class_array) > 0  # 売り条件を満たしたブール．unit_numberがマイナスの時，0にする
        sell_condition_true_min_index = (~sell_condition_bool).sum() - 1  # 条件を満たした最小値のインデックス
        if sell_condition_true_min_index < 0:  # -1になってしまった場合
            sell_condition_true_min_index = 0
                
        # 行動の制約を厳密化
        if self.action_restriction:
            if action < self.stay_index:  # 売りの場合
                if action < sell_condition_true_min_index:  # 行動が最小条件より下になった場合
                    action = sell_condition_true_min_index

            elif action > self.stay_index:  # 買いの場合
                if action > buy_condition_true_max_index:  # 行動が最大条件より上になった場合
                    action = buy_condition_true_max_index
                
            # それ以外(ステイ)はそのまま
        
        penalty = 0
        
        # 行動ペナルティ
        # 売買条件ペナルティ
        trading_penalty = self.get_trading_penalty(sell_condition_true_min_index, buy_condition_true_max_index, action)
        penalty += trading_penalty
        
        # ステイペナルティ
        stay_penalty = self.get_stay_penalty(cash, action, unit_number)
        penalty += stay_penalty
        
        # 平均取得価格と現在価格が近いときのペナルティ
        #mca_np_diff_penalty = self.get_mcp_np_diff_penalty(mean_cost_price, now_price)
        #penalty += mca_np_diff_penalty
        
        # マイナスペナルティ
        minus_penalty = self.get_minus_penalty(unit_number, cash, now_price)
        penalty += minus_penalty
        
        # ペナルティのチェック
        if penalty > 0:
            raise Exception("penalty must be minus")
        
        # 追加単元数
        add_unit_number = self.action_class_array[action]  # actionはint?
        
        # 状態の計算に必要な報酬
        gein_reward = self.get_gein_reward(now_price, add_unit_number)  # 株式売買による報酬
        price_reward = self.get_price_reward(now_price, mean_cost_price, add_unit_number)  # 平均取得価格による報酬
        
        # 状態の更新
        new_unit_number = unit_number + add_unit_number
        new_cash = cash + gein_reward  # 手数料はない
        
        # 平均取得価格の計算
        if add_unit_number > 0:  # 買いの場合
            num = cliped_unit_number * mean_cost_price + add_unit_number * now_price
            den = (cliped_unit_number + add_unit_number)
            new_mean_cost_price = num / den
        elif add_unit_number <= 0:  # ステイ・売りの場合
            if unit_number <= 0:  # ユニット数が0以下である場合
                new_mean_cost_price = now_price  # 現在価格とする．
            else:
                new_mean_cost_price = mean_cost_price  # 元のまま
                

        new_all_property = new_cash + now_price * new_unit_number * self.one_unit_stocks  # 全資産
        
        self.state = StockState(cash=new_cash,
                                unit_number=new_unit_number,
                                mean_cost_price=new_mean_cost_price,
                                all_property=new_all_property,
                                price_array=state_prices_array
                               )
        
        # 学習に利用する報酬
        all_property_reward = self.get_all_property_reward(all_property, new_all_property)  # 全資産による報酬
        reward = self.gein_reward_weight * gein_reward + self.price_reward_weight * price_reward +  self.penalty_weight * penalty
        reward += self.all_property_reward_weight * all_property_reward # 重み
        
        
        self.now_datetime = now_datetime
        
        self.info = {"datetime":self.now_datetime, 
                     "stock_name":self.stock_name,
                     "done_action":action,
                     "penalty":penalty,
                     "gein_reward":gein_reward,
                     "price_reward":price_reward,
                     "all_property_reward":all_property_reward,
                     "reward":reward,
                    }  # stock_nameを入れる意味ある？
        
        self.now_index = now_index
        self.step_counter += 1
        
        # doneの更新
        done = (self.step_counter >= self.episode_length) or (new_all_property < self.low_limmit)
        
        # エピソードが終了して，resetが呼ばれる前にstepが呼ばれた時のエラー判定
        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0  # 0 に初期化
                
            elif self.steps_beyond_done == 0:  # 0の時のみロギング
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You"
                    "should always call 'reset()' once you receive 'done ="
                    "True' -- any further steps are undefined behavior"
                )
                self.steps_beyond_done += 1
        
        return self.state.copy(), reward, done, self.info  # 一応stateをコピー

    def get_gein_reward(self, now_price, add_unit_number):
        """
        株式の売却による利益を意味する報酬
        """
        gein_reward = now_price * self.one_unit_stocks * (- add_unit_number) # 買いがプラスであるため，マイナスをかける．cashの計算に利用する．
        return gein_reward
    
    def get_price_reward(self, now_price, mean_cost_price, add_unit_number):
        """
        平均取得価格と現在価格とその売買に関係する報酬
        """
        price_reward = (now_price - mean_cost_price) * self.one_unit_stocks * (- add_unit_number)
        return price_reward
    
    def get_all_property_reward(self, pre_all_property, new_all_property):
        """
        全資産による報酬
        """
        all_property_reward = new_all_property - pre_all_property
        return all_property_reward
    
    def get_trading_penalty(self, sell_condition_index, buy_condition_index, action):
        """
        売買条件を制限するペナルティ．
        """
        penalty = 0
        # 売買条件ペナルティ
        if action < self.stay_index:  # 売りの場合
            if action < sell_condition_index:  # 行動が最小条件より下になった場合
                prepenalty = - (sell_condition_index - action) * self.initial_cash * 0
                assert prepenalty <= 0,"this penalty must be minus {}".format(prepenalty)
                penalty += prepenalty
            
        elif action > self.stay_index:  # 買いの場合
            if action > buy_condition_index:  # 行動が最大条件より上になった場合
                prepenalty = - (action - buy_condition_index) * self.initial_cash * 0
                assert prepenalty <= 0,"this penalty must be minus {}".format(prepenalty)
                penalty += prepenalty
        return penalty
    
    def get_stay_penalty(self, cash, action, unit_number):        
        penalty = 0
        condition_list = []
        low_unit_stay_condition = action == self.stay_index and unit_number < self.stay_penalty_unit_bound  # 行動がステイかつユニット数が閾値以下の場合
        condition_list.append(low_unit_stay_condition)
        low_cash_stay_condition = action == self.stay_index and cash < self.stay_penalty_cash_bound  # 行動がステイかつ現金が閾値以下の場合
        condition_list.append(low_cash_stay_condition)
        
        if any(condition_list):  # 条件のどれかが当てはまる場合
            #prepenalty = cash * (-0.10)  # 現金のx%のペナルティ
            #prepenalty = - self.initial_cash * 0
            prepenalty = -self.penalty_const
            penalty += prepenalty
        
        prepenalty = - cash * 0.0  # 条件に関わらないペナルティ
        penalty += prepenalty
            
        return penalty
    
    def get_mcp_np_diff_penalty(self, mean_cost_price, now_price):
        penalty = 0
        mcp_np_diff = abs(now_price-mean_cost_price)
        if mcp_np_diff < self.penalty_mcp_np_diff_bound:
            prepenalty = - self.initial_cash * 0.1
            penalty += prepenalty
        return penalty
    
    def get_minus_penalty(self, unit_number, cash, now_price):
        penalty = 0
        if unit_number < 0:
            minus_abs_unit = abs(unit_number)
            #prepenalty = - self.initial_cash * 10 * minus_abs_unit
            prepenalty = - self.penalty_const
            penalty += prepenalty
        
        if cash < 0:
            minus_abs_cash = abs(cash)
            minus_abs_cash_as_unit = minus_abs_cash / (self.one_unit_stocks * now_price)
            #prepenalty = - self.initial_cash * 10 * minus_abs_cash_as_unit
            prepenalty = -self.penalty_const
            penalty += prepenalty
        
        return penalty
            
    def close(self):
        """
        ビューのクローズとseedのリセット
        """
        if self.use_view:
            self.stock_view.close()
        self.seed()  # シードの初期化
    
    def seed(self, initial_seed=None, mcp_seed=None):
        """
        reset時のサンプリングのseedを設定する．リセットの場合，何も指定しないかNoneを指定
        """
        self.initial_seed_number = initial_seed
        self.mcp_seed_number = mcp_seed

        
    def set_weight(self, gein_reward_weight=0, price_reward_weight=1, penalty_weight=1, all_property_reward_weight=1):
        self.gein_reward_weight = gein_reward_weight
        self.price_reward_weight = price_reward_weight
        self.all_property_reward_weight = all_property_reward_weight
        self.penalty_weight = penalty_weight


def make_env():
    db_path = Path("E:/システムトレード入門/trade_system_git_workspace/db/sub_stock_db") / Path("sub_stock.db")
    stock_db = StockDatabase(str(db_path))

    jst_timezone = timezone("Asia/Tokyo")
    start_datetime = jst_timezone.localize(datetime.datetime(2020,11,1,0,0,0))
    end_datetime = jst_timezone.localize(datetime.datetime(2020,12,1,0,0,0))

    stock_names = "6502"

    initial_cash = 10.e6  # 種銭：100万円
    initial_unit = 100  # 初期単元数

    freq_str = "5T"
    episode_length = 12*5*7  # 1週間

    use_ohlc="Close"

    #state_time_list = [0,1,12,12*3,12*5,12*5*3],  # [現在，次時刻，一時間後，3時間後，5時間後(1日後), 15時間後(3日後)]
    state_time_list = [0,
                    1,
                    2,
                    6,
                    12,
                    12*2,
                    12*3,
                    12*4,
                    12*5*1,
                    12*5*2,
                    12*5*3,
                    12*5*4,
                    12*5*5,
                    ]  # 現在，5分後, 10分後, 30分後, 1時間後, 2時間後, 3時間後, 4時間後, 1日後, 2日後, 3日後, 4日後, 5日後, 6日後, 7日後

    one_unit_stocks = 20
    max_units_number = 5
    stay_penalty_unit_bound=30
    stay_penalty_cash_bound = 1.e5
    penalty_mcp_np_diff_bound = 3


    env = OneStockEnv(stock_db,
                    stock_names=stock_names,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    freq_str=freq_str,
                    episode_length=episode_length,  # 一週間
                    state_time_list=state_time_list,
                    use_ohlc=use_ohlc,  # 終値を使う
                    initial_cash=initial_cash,  # 種銭
                    initial_unit=initial_unit,
                    use_view=False,
                    one_unit_stocks=one_unit_stocks,  # 独自単元株数
                    max_units_number=max_units_number,  # 一度に売買できる独自単元数
                    low_limmit=1.e4,  # 全財産がこの値以下になれば終了
                    interpolate=True,
                    stay_penalty_unit_bound=stay_penalty_unit_bound,  # このunit数以下の場合のstayはペナルティ
                    stay_penalty_cash_bound=stay_penalty_cash_bound,  # このcash以下の場合のstayはペナルティ
                    penalty_mcp_np_diff_bound=penalty_mcp_np_diff_bound
                    )

    env.set_weight(gein_reward_weight=0,
                   price_reward_weight=0.8,
                   all_property_reward_weight=0.04942905815076451,
                   )

    return env


if __name__ == "__main__":
    from pathlib import Path
    from pytz import timezone

    from get_stock_price import StockDatabase
    
    ##########################
    ### OneStockEnv
    ##########################
    
    db_path = Path("E:/システムトレード入門/trade_system_git_workspace/db/stock_db") / Path("stock.db")
    stock_db = StockDatabase(db_path)


    jst_timezone = timezone("Asia/Tokyo")
    start_datetime = jst_timezone.localize(datetime.datetime(2020,11,1,0,0,0))
    end_datetime = jst_timezone.localize(datetime.datetime(2020,12,1,0,0,0))
    #end_datetime = get_next_workday_jp(start_datetime, days=11)  # 営業日で一週間(5日間)


    stock_names = "4755"
    #stock_names = ["6502","4755"]

    freq_str = "5T"
    episode_length = 12*5*7  # 一週間

    #state_time_list = [0,1,12,12*3,12*5,12*5*3],  # [現在，次時刻，一時間後，3時間後，5時間後(1日後), 15時間後(3日後)]
    state_time_list = [0,
                    1,
                    2,
                    6,
                    12,
                    12*2,
                    12*3,
                    12*4,
                    12*5*1,
                    12*5*2,
                    12*5*3,
                    12*5*4,
                    12*5*5,
                    ]  # 現在，5分後, 10分後, 30分後, 1時間後, 2時間後, 3時間後, 4時間後, 1日後, 2日後, 3日後, 4日後, 5日後, 6日後, 7日後



    env = OneStockEnv(stock_db,
                    stock_names=stock_names,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    freq_str="5T",
                    episode_length=episode_length,  # 一週間
                    state_time_list=state_time_list,
                    use_ohlc="Close",  # 終値を使う
                    initial_cash=1.e6,  # 種銭：100万円,
                    use_view=False,
                    one_unit_stocks=10,  # 独自単元数
                    max_units_number=10,  # 一度に変える独自単元数
                    low_limmit=1.e4,  # 全財産がこの値以下になれば終了
                    interpolate=True,
                    penalty_ratio=1.e-2
                    )


    #env.reset(select_datetime=jst_timezone.localize(datetime.datetime(2020, 11, 4, 14, 0)), select_stock_name="4755")
    env.reset()

    for i in range(100):
        #action = env.action_space.sample()
        action = 15
        observe, reward, done, info = env.step(action)

        cash = observe[0]
        unit_number = observe[1]
        now_price = observe[2]

        done_action = info["done_action"]
        print("action selected:{}, done:{}".format(action, done_action))
        print("done_buy_sell stocks:{}, deal:{}:".format(env.action_class_array[done_action]*env.one_unit_stocks, -now_price*env.action_class_array[done_action]))
        print("observe:",observe)
        print("cash:",cash)
        print("stocks:",unit_number*env.one_unit_stocks)
        print("all property:",env.all_property)
        print("#################################")