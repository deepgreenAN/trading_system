import numpy as np
from utils import TradeSystemBaseError

class StateTransformInvalidError(TradeSystemBaseError):
    """
    StateTransformで起きたエラー
    """
    def __init__(self, err_str=None):
        """
        err_str:
            エラーメッセージ
        """
        self.err_str = err_str
    def __str__(self):
        if self.err_str is None:
            return "Cannot get all data."
        else:
            return self.err_str


class ComposeFunction:
    """
    callabreのリスト・ディクショナリ全てを実行する．動的なアクセスのために各collableはアトリビュートとして保持する．
    """
    def __init__(self, collection):
        """
        collection: list or dict of function
            適用したいcallableなオブジェクトのリストか辞書．辞書の場合はキーはアトリビュート名とする．
        """
        if isinstance(collection, list):
            self.function_name_list = []
            for i, func in enumerate(collection):
                func_name = "func_"+str(i)
                setattr(self, func_name, func)
                self.function_name_list.append(func_name)
            
        elif isinstance(collection, dict):
            self.function_name_list = []
            for key in collection:
                setattr(self, key, collection[key])
                self.function_name_list.append(key)
                
    def __call__(self, x):
        """
        x: any
            各関数の引数
        """
        for func_name in self.function_name_list:
            x = getattr(self, func_name)(x)
            
        return x


class State2Feature:
    """
    最後に実行
    """
    def __call__(self, portfolio_state):
        price_array = portfolio_state.price_array
        price_portfolio = price_array * portfolio_state.portfolio_vector[:,None]
        price_mean_cost = price_array * portfolio_state.mean_cost_price_array[:,None]
        feature = np.stack([price_array, price_portfolio, price_mean_cost], axis=0)
        return feature


class PriceNormalizeConst:
    """
    price_arrayを定数で割る
    """
    def __init__(self, const_array=None):
        self._const_array = const_array
        
    @property
    def const_array(self):
        return self._const_array
    
    @const_array.setter
    def const_array(self, const_array):
        if const_array is not None:
            if np.isnan(const_array).sum() > 0:
                raise StateTransformInvalidError("PriceNormalizeConst.const_array cannot set array include nan.")
        else:
            raise StateTransformInvalidError("PriceNormalizeConst.const_array cannot be sestted None")
        self._const_array = const_array
        
    def __call__(self, portfolio_state):
        if portfolio_state.price_array.shape[0]!=self._const_array.shape[0]:
            err_str = "portfolio_state.price_array shape({}) and PriceNormalizeConst.const_array({})".format(portfolio_state.price_array.shape,
                                                                                                             self._const_array.shape
                                                                                                            )
            raise StateTransformInvalidError(err_str)
        
        new_price_array = portfolio_state.price_array / self._const_array[:,None]
        return portfolio_state._replace(price_array=new_price_array)

class MeanCostPriceNormalizeConst:
    """
    mean_cost_price_arrayを定数で割る
    """
    def __init__(self, const_array):
        self._const_array = const_array
        
    @property
    def const_array(self):
        return self._const_array
    
    @const_array.setter
    def const_array(self, const_array):
        if const_array is not None:
            if np.isnan(const_array).sum() > 0:
                raise StateTransformInvalidError("MeanCostPriceNormalizeConst.const_array cannot set array include nan.")
        else:
            raise StateTransformInvalidError("MeanCostPriceNormalizeConst.const_array cannot be sestted None")
        self._const_array = const_array
    
    def __call__(self, portfolio_state):
        if portfolio_state.mean_cost_price_array.shape[0]!=self._const_array.shape[0]:
            err_str = "portfolio_state.price_array shape({}) and MeanCostPriceNormalizeConst.const_array({})".format(portfolio_state.mean_cost_price_array,
                                                                                                             self._const_array.shape
                                                                                                            )
            raise StateTransformInvalidError(err_str)
        new_mean_cost_price = portfolio_state.mean_cost_price_array / self._const_array
        return portfolio_state._replace(mean_cost_price_array=new_mean_cost_price)


if __name__ == "__main__":
    pass