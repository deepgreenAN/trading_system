import numpy as np


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
        self.const_array = const_array
        
    def __call__(self, portfolio_state):
        assert portfolio_state.price_array.shape[0]==self.const_array.shape[0]
        new_price_array = portfolio_state.price_array / self.const_array[:,None]
        return portfolio_state._replace(price_array=new_price_array)


class MeanCostPriceNormalizeConst:
    """
    mean_cost_price_arrayを定数で割る
    """
    def __init__(self, const_array):
        self.const_array = const_array
    
    def __call__(self, portfolio_state):
        assert portfolio_state.mean_cost_price_array.shape[0]==self.const_array.shape[0]
        new_mean_cost_price = portfolio_state.mean_cost_price_array / self.const_array
        return portfolio_state._replace(mean_cost_price_array=new_mean_cost_price)


if __name__ == "__main__":
    pass