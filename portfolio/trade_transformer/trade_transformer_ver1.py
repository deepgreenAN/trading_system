import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod

from .portfolio_restrictor import PortfolioRestrictorIdentity
from .fee_calculator import FeeCalculatorPerNumber
from .data_state_element import PortfolioState

class PortfolioTransformer:
    """
    price_supplierの提供するデータに応じてPortfolioStateを遷移させるクラス
    バックテスト・強化学習のどちらでも使えるようにする．
    """
    def __init__(self, 
                 price_supplier, 
                 portfolio_restrictor=PortfolioRestrictorIdentity(), 
                 use_ohlc="Close", 
                 initial_portfolio_vector=None, 
                 initial_all_assets=1e6, 
                 fee_calculator=FeeCalculatorPerNumber(fee_per_number=1e-3)):
        """
        price_supplier: PriceSupplier
            価格データを供給するクラス
        portfolio_restrictor: PortfolioRestrictor
            エージェントが渡すportfolio_vectorを制限するクラス
        use_ohlc: str, defalt:'Close'
            利用する価格データの指定
        initial_portfolio_vector: any, defalt:None
            初期ポートフォリオベクトル
        fee_calculator: FeeCalculator
            手数料を計算するクラス
        """
        self.price_supplier = price_supplier
        self.portfolio_restrictor = portfolio_restrictor
        self.initial_portfolio_vector = initial_portfolio_vector
        self.initial_all_assets = initial_all_assets
        self.fee_calculator = fee_calculator
    
        # 利用するohlcのいずれか
        if use_ohlc not in {"Open","High","Low","Close"}:
            raise Exception("use_ohlc must be in {'Open','High','Low','Close'}")
        
        field_name_dict = {"Open":"open_array",
                           "Close":"close_array",
                           "Low":"low_array",
                           "High":"high_array"
                          }
            
        self.use_ohlc_filed = field_name_dict[use_ohlc]
        
    def reset(self, start_datetime, window=[0]):
        """
        Parameters
        ----------
        start_datetime: datetime.datetime 
            データ供給の開始時刻
        window: ndarray
            データ供給のウィンドウ
            
        Returns
        -------
        PortfolioStat
             ポートフォリオ状態
        bool
            エピソードが終了したかどうか
        """
        initial_data_unit, done = self.price_supplier.reset(start_datetime, window)
    
        now_price_bool = initial_data_unit.window==0 
    
        if self.initial_portfolio_vector is None:
            self.initial_portfolio_vector = np.zeros(len(initial_data_unit.names))
            self.initial_portfolio_vector[initial_data_unit.key_currency_index] = 1.0
            
        else:
            assert len(initial_data_unit.names) == len(self.initial_portfolio_vector)
            assert self.initial_portfolio_vector.sum() == 1.0
            
        
        now_price_array = getattr(initial_data_unit, self.use_ohlc_filed)[:,now_price_bool].squeeze()
    
        self.portfolio_state = PortfolioState(names=initial_data_unit.names,
                                              key_currency_index=initial_data_unit.key_currency_index,
                                              window=initial_data_unit.window,
                                              datetime=initial_data_unit.datetime,
                                              price_array=getattr(initial_data_unit, self.use_ohlc_filed),
                                              volume_array=initial_data_unit.volume_array,
                                              now_price_array=now_price_array,
                                              portfolio_vector=self.initial_portfolio_vector,
                                              mean_cost_price_array=now_price_array,
                                              all_assets=self.initial_all_assets
                                             )
        
        
        return self.portfolio_state.copy(), done
    
    def step(self, action):
        """
        Parameters
        ----------
        action: ndarray
            エージェントが渡すポートフォリオベクトル
            
        Returns
        -------
        PortfolioStat
             ポートフォリオ状態
        bool
            エピソードが終了したかどうか
        """
        
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        assert (action<0).sum() == 0 and (action>1).sum() == 0
        assert abs(action.sum() - 1.0) < 1.e-5  # 大体1ならOK
        
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        previous_portfolio_state = self.portfolio_state
        supplied_data_unit, done = self.price_supplier.step()
        
        assert len(action)==len(supplied_data_unit.names)
        
        restricted_portfolio_vector = self.portfolio_restrictor.restrict(previous_portfolio_state, supplied_data_unit, action)
        
        # 全資産の変化率を求める
        now_price_bool = supplied_data_unit.window==0
        now_price_array = getattr(supplied_data_unit, self.use_ohlc_filed)[:,now_price_bool].squeeze()
        
        price_change_ratio = now_price_array / previous_portfolio_state.now_price_array
        
        all_assets_change_ratio = np.dot(restricted_portfolio_vector, price_change_ratio)
        all_assets = previous_portfolio_state.all_assets * all_assets_change_ratio
        
        # 平均取得価格を設ける
        new_numbers = all_assets*restricted_portfolio_vector/now_price_array
        pre_numbers = previous_portfolio_state.numbers
        mean_num = pre_numbers*previous_portfolio_state.now_price_array + (new_numbers - pre_numbers) * now_price_array
        mean_den = new_numbers
        
        new_numbers_near_zero_bool = new_numbers < 1  # 取り合えず1以下の場合
        mean_num[new_numbers_near_zero_bool] = 1  # 適当に1にしておく
        mean_den[new_numbers_near_zero_bool] = 1  # 適当に1にしておく
        
        mean_cost_price_array = mean_num / mean_den
        mean_cost_price_array[new_numbers_near_zero_bool] = now_price_array[new_numbers_near_zero_bool]
        
        self.portfolio_state = PortfolioState(names=supplied_data_unit.names,
                                              key_currency_index=supplied_data_unit.key_currency_index,
                                              window=supplied_data_unit.window,
                                              datetime=supplied_data_unit.datetime,
                                              price_array=getattr(supplied_data_unit, self.use_ohlc_filed),
                                              volume_array=supplied_data_unit.volume_array,
                                              now_price_array=now_price_array,
                                              portfolio_vector=restricted_portfolio_vector,
                                              mean_cost_price_array=mean_cost_price_array,
                                              all_assets=all_assets
                                             )
                 
        # 手数料の計算と更新
        all_fee = self.fee_calculator.calculate(previous_portfolio_state, self.portfolio_state)
        self.portfolio_state = self.portfolio_state._replace(all_assets=all_assets-all_fee)   
        
        return self.portfolio_state.copy(), done
        
if __name__ == "__main__":
    pass


