from collections import namedtuple
import numpy as np

field_list = ["names",  # 銘柄名
              "key_currency_index",  # 基軸通貨のインデックス
              "datetime",  # データの日時
              "window",  # データのウィンドウ
              "open_array",  # [銘柄名, ウィンドウ(時間)]に対応する始値
              "close_array",  # [銘柄名, ウィンドウ(時間)]に対応する終値
              "high_array",  # [銘柄名, ウィンドウ(時間)]に対応する高値
              "low_array",  # [銘柄名, ウィンドウ(時間)]に対応する低値
              "volume_array"  # [銘柄名, ウィンドウ(時間)]に対応する取引量
             ]

DataSupplyUnitBaseLegacy = namedtuple("DataSupplyUnitBase", field_list)


class DataSupplyUnitLegacy(DataSupplyUnitBaseLegacy):
    """
    DataSupplierによって提供されるデータクラス
    """    
    def __str__(self):
        return_str = "DataSupplyUnit( \n"
        for field_str in self._fields:
            return_str += field_str + "="
            return_str += str(getattr(self, field_str)) + "\n"
        return_str += ")"
        return return_str


field_list = ["names",  # 銘柄名
              "key_currency_index",  # 基軸通貨のインデックス
              "window",  # データのウィンドウ
              "datetime",  # データの日時
              "price_array",  # [銘柄名, ウィンドウ(時間)]に対応する現在価格
              "volume_array",  # [銘柄名, ウィンドウ(時間)]に対応する取引量
              "now_price_array",  # 銘柄名に対応する現在価格
              "portfolio_vector",  # ポートフォリオベクトル
              "mean_cost_price_array",  # 銘柄名に対応する平均取得価格
              "all_assets"  # 基軸通貨で換算した全資産
             ]

PortfolioStateBaseLegacy = namedtuple("PortfolioStateBase", field_list)


class PortfolioStateLegacy(PortfolioStateBaseLegacy):
    """
    バックテスト・強化学習で利用するTransformerが提供するデータクラス．強化学習における状態を内包する．
    """
    
    @property
    def numbers(self):
        """
        保有量のプロパティ
        """
        return self.all_assets*self.portfolio_vector/self.now_price_array
    
    def __str__(self):
        return_str = "PortfolioState( \n"
        for field_str in self._fields:
            return_str += field_str + "="
            return_str += str(getattr(self, field_str)) + "\n"
        return_str += ")"
        return return_str
    
    def copy(self):
        """
        自身のコビーを返す．ndarrayのプロパティの場合はそのコビーを保持する．
        """
        arg_dict = {}
        for field_str in self._fields:
            field_value = getattr(self, field_str)
            if isinstance(field_value, np.ndarray):
                field_value = field_value.copy()
            
            arg_dict[field_str] = field_value
        
        return PortfolioStateLegacy(**arg_dict)
    
    def partial(self, *args):
        """
        メモリ等の状況によって，自身の部分的なコビーを返す．
        引数にを耐えられなかったプロパティはNoneとなる．
        """
        arg_dict = {}
        for field_str in self._fields:
            if field_str in args:
                field_value = getattr(self, field_str)
                if isinstance(field_value, np.ndarray):
                    field_value = field_value.copy()
            else:
                field_value = None
            
            arg_dict[field_str] = field_value
            
        return PortfolioStateLegacy(**arg_dict)

if __name__ == "__main__":
    pass