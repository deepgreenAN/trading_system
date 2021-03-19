import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
import pandas as pd
from tqdm import tqdm
import datetime

import gym
from gym import spaces, logger

import sys
sys.path.append(r"E:\システムトレード入門\trade_system_git_workspace")

from utils import middle_sample_type_with_check, get_sec_from_freq, get_workdays_jp, get_previous_datetime, get_next_workday_jp
from utils import extract_workdays_intraday_jp_index, extract_workdays_intraday_jp


class NormalizeState():
    def __init__(self, init_cash, init_cash_times=1, unit_const=1, price_const=1, replace=False):
        self.cash_devide_const = init_cash * init_cash_times
        self.unit_devide_const = unit_const
        self.price_devide_const = price_const
        
        self.replace = replace
        
    def __call__(self, state):
        if self.replace:
            state = state.copy()

        # cash
        state[0] = state[0]/self.cash_devide_const
        # unit
        state[1] = state[1]/self.unit_devide_const
        # price
        state[2:] = state[2:]/self.price_devide_const

        return state

class NormalizeReward():
    def __init__(self, init_cash, init_cash_times=1):
        self.reward_devide_const = init_cash * init_cash_times
        
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
                 use_view=False,
                 one_unit_stocks=100,
                 max_units_number=10,
                 low_limmit=1.e3,  # 1000円
                 interpolate=True,
                 sample_remove=False,
                 penalty_ratio=1.e-4
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
        self.low_limmit = low_limmit
        self.interpolate = interpolate
        self.sample_remove = sample_remove
        self.seed_number = None
        self.penalty_ratio = penalty_ratio
        
        if state_time_list[0] != 0:
            raise Exception("The first value of state_time_list must be 0")
        
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
        
        state_max_list = [max_cash, max_unit]
        state_max_list.extend([max_stock_price]*len(self.state_time_list))
        
        state_min_list = [min_cash, min_unit]
        state_min_list.extend([min_stock_price]*len(self.state_time_list))
                
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
        環境をリセットする．エピソード開始日時と銘柄名を指定することで，特定の時間のデータを環境として取得できる．
        その場合，領域内にnanが存在しても自動的に補間される．
        select_datetime: datetime.datetime
            エピソードの開始日時
        select_stock_name: str
            銘柄名
        """
        iter_counter = 0
        while True:  # 条件をクリアしたらreturn,それまで繰り返し
            stock_and_datetime = self.sampler.sample(remove=self.sample_remove, seed=self.seed_number)
            
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

                
        cash = self.initial_cash
        unit_number = 0
        cash_and_unit_array = np.array([cash, unit_number])

        # 状態で利用する部分を取得
        #state_datetimes = [self.all_datetime_index[episode_start_datetime_index+add_index].to_pydatetime() for add_index in self.state_time_list]  # datetime.datetimeに変更
        state_datetimes = [self.all_datetime_index[episode_start_datetime_index+add_index] for add_index in self.state_time_list]  # pd.Timestampのまま
        
        state_prices_bool_array = self.episode_series.index.isin(state_datetimes)

        state_prices_series = self.episode_series.loc[state_prices_bool_array].copy()
        state_prices_array = state_prices_series.values

        # 現状態をアトリビュートとして保持
        self.state = np.concatenate([cash_and_unit_array, state_prices_array], axis=0)
        self.all_property = cash + state_prices_array[0] * unit_number * self.one_unit_stocks  # 最初は現在の価格
        self.now_datetime = episode_start_datetime  # 現時間
        self.info = {"datetime":self.now_datetime, "stock_name":stock_name, "done_actine":None, "iter_counter":iter_counter, "all_property":self.all_property}
        
        self.now_index = episode_start_datetime_index  # 現時間のインデックス
        self.stock_name = stock_name  # 現銘柄コード
        self.step_counter = 1  # ステップ数のカウンタ
        done = False  # ここではもちろんFalse
        
        # エラー処理で利用
        self.steps_beyond_done = None
        
        return self.state.copy(), None, done, self.info  # なぜかagentがinplaceに変更してしまうので，stateをコピー
    
    def step(self, action):
        #assert self.action_space.contains(action), "action {} ({}) invalid".format(action, type(action))
        
        state = self.state
        
        cash = state[0]
        unit_number = state[1]
        
        prices = state[2:]  # 使わない？
        
        now_index = self.now_index + 1
        now_datetime = self.all_datetime_index[now_index].to_pydatetime()  # infoで利用
        
        #state_datetimes = [self.all_datetime_index[now_index+add_index].to_pydatetime() for add_index in self.state_time_list]
        state_datetimes = [self.all_datetime_index[now_index+add_index] for add_index in self.state_time_list]  # pd.Timestampのまま
        
        state_prices_bool_array = self.episode_series.index.isin(state_datetimes)

        state_prices_series = self.episode_series.loc[state_prices_bool_array].copy()
        state_prices_array = state_prices_series.values
        
        # 行動の制約（現金より多く変えない，保持数より多くは売れない)
        now_price = state_prices_array[0]  # state_priceの最初は必ずnowでなければならない
        buy_condition_bool = (cash - now_price * self.action_class_array * self.one_unit_stocks) >= 0  # 買い条件を満たしたブール
        # print("buy_condition_bool:",buy_condition_bool)
        buy_condition_true_max_index = buy_condition_bool.sum() - 1  # 条件を満たした最大値のインデックス
        if buy_condition_true_max_index < 0:  # -1になってしまった場合
            buy_condition_true_max_index = 0
        
        sell_condition_bool = (unit_number + self.action_class_array) > 0  # 売り条件を満たしたブール
        # print("sell_conditiion_bool:",sell_condition_bool)
        sell_condition_true_min_index = (~sell_condition_bool).sum() - 1  # 条件を満たした最小値のインデックス
        if sell_condition_true_min_index < 0:  # -1になってしまった場合
            sell_condition_true_min_index = 0
                
        if action < self.stay_index:  # 売りの場合
            if action < sell_condition_true_min_index:  # 行動が最小条件より下になった場合
                action = sell_condition_true_min_index
                
        elif action > self.stay_index:  # 買いの場合
            if action > buy_condition_true_max_index:  # 行動が最大条件より上になった場合
                action = buy_condition_true_max_index
                
        # それ以外(ステイ)はそのまま
        
        
        # rewardの計算とunit・cashの更新
        reward = now_price * self.one_unit_stocks * (- self.action_class_array[action]) # 買いがプラスであるため，
        # ペナルティの計算
        if action==self.stay_index:  # ステイしている
            penalty =  - self.penalty_ratio * cash # 現金の0.01%のペナルティ
            if (cash + reward + penalty) < 0:  # ペナルティでマイナスにならないようにする
                penalty = 0
        else:
            penalty = 0
            
        reward += penalty  # ペナルティの追加
        
        unit_number += self.action_class_array[action]  # actionはint?
        cash += reward  # 手数料はない

        self.all_property = cash + now_price * unit_number * self.one_unit_stocks  # 全資産
        
        cash_and_unit_array = np.array([cash, unit_number])
        self.state = np.concatenate([cash_and_unit_array, state_prices_array],axis=0)
        self.now_datetime = now_datetime
        
        self.info = {"datetime":self.now_datetime, "stock_name":self.stock_name, "done_action":action, "all_property":self.all_property}  # stock_nameを入れる意味ある？
        
        self.now_index = now_index
        self.step_counter += 1
        
        # doneの更新
        done = (self.step_counter >= self.episode_length) or (self.all_property < self.low_limmit)
        
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
        
        return self.state.copy(), reward, done, self.info  # なぜかagentがinplaceに変更してしまうので，stateをコピー

    def close(self):
        """
        ビューのクローズとseedのリセット
        """
        if self.use_view:
            self.stock_view.close()
        self.seed(None)  # シードの初期化
    
    def seed(self, seed=None):
        """
        reset時のサンプリングのseedを設定する．リセットの場合，何も指定しないかNoneを指定
        """
        self.seed_number = seed


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