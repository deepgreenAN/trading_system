import numpy as np
import datetime
import dataclasses
from dataclasses import dataclass
from collections import namedtuple

from utils import TradeSystemBaseError
from .portfolio_error import PortfolioVectorInvalidError

class UnitStateError(TradeSystemBaseError):
    """
    UnitあるいはStateに関するクラスのベースクラス
    """
    pass


class UnitStateHasNanError(UnitStateError):
    """
    DataSupplyUnitの要素の代入時にNaNをチェックする
    """
    def __init__(self, err_str=None):
        """
        err_str:
            エラーメッセージ
        """
        self.err_str = err_str
    def __str__(self):
        if self.err_str is None:
            return "This Unit has nan data"
        else:
            return self.err_str


class UnitStateHasWrongLengthError(UnitStateError):
    """
    Unitのnameを基準に特定のフィールドの長さをチェックする
    """
    def __init__(self, err_str=None):
        """
        err_str:
            エラーメッセージ
        """
        self.err_str = err_str
    def __str__(self):
        if self.err_str is None:
            return "This Unit has wrong length"
        else:
            return self.err_str


@dataclass
class DataSupplyUnit:
    """
    DataSupplierによって提供されるデータクラス
    nan_check: bool
        初期化時にnanをチェックするかどうか．デバッグ時に利用する
    length_check: bool
        初期化時にlengthをチェックするかどうか，デバッグ時に利用する．
    """
    nan_check = False
    length_check = False
    
    names: np.ndarray # 銘柄名
    key_currency_index: int  # 基軸通貨のインデックス
    datetime: datetime.datetime  # データの日時
    window: np.ndarray  # データのウィンドウ
    open_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する始値
    close_array: np.ndarray # [銘柄名, ウィンドウ(時間)]に対応する終値
    high_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する高値
    low_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する低値
    volume_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する取引量
        
    def _replace(self, **kwargs):
        """
        namedtupleとの互換性のため
        """
        return dataclasses.replace(self, **kwargs)
        
    def __post_init__(self):
        # nanが含まれるがチェック
        if DataSupplyUnit.nan_check:
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, np.ndarray):
                    if np.isnan(value).sum() > 0:
                        raise UnitStateHasNanError("This Unit has nan data about {}".format(field.name))
                    
        # 長さが適切かチェック
        if DataSupplyUnit.length_check:
            name_length = len(self.names)
            window_length = len(self.window)
            for field in dataclasses.fields(self):
                if field.name in {"open_array", "close_array", "high_array", "low_array", "volume_array"}:
                    value = getattr(self, field.name)
                    if value.shape[0]!=name_length or value.shape[1]!=window_length:
                        err_str = "This Unit has wrong legnth about {}({}) with names({}) and window({})".format(field.name,
                                                                                                                value.shape,
                                                                                                                name_length,
                                                                                                                window_length
                                                                                                               )
                        raise UnitStateHasWrongLengthError(err_str)
                        
    def __str__(self):
        return_str = "DataSupplyUnit( \n"
        for field in dataclasses.fields(self):
            return_str += field.name + "="
            return_str += str(getattr(self, field.name)) + "\n"
        return_str += ")"
        return return_str


@dataclass
class PortfolioState:
    """
    バックテスト・強化学習で利用するTransformerが提供するデータクラス．強化学習における状態を内包する．
    nan_check: bool
        初期化時にnanをチェックするかどうか．デバッグ時に利用する
    length_check: bool
        初期化時にlengthをチェックするかどうか，デバッグ時に利用する．
    """
    nan_check = False
    length_check = False
    portfoio_check = False
    
    names: np.ndarray  # 銘柄名
    key_currency_index: int  # 基軸通貨のインデックス
    window: np.ndarray  # データのウィンドウ
    datetime: datetime.datetime  # データの日時
    price_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する現在価格
    volume_array: np.ndarray  # [銘柄名, ウィンドウ(時間)]に対応する取引量
    now_price_array: np.ndarray # 銘柄名に対応する現在価格
    portfolio_vector: np.ndarray  # ポートフォリオベクトル
    mean_cost_price_array: np.ndarray  # 銘柄名に対応する平均取得価格
    all_assets: float  # 基軸通貨で換算した全資産
        
    def _replace(self, **kwargs):
        """
        namedtupleとの互換性のため，
        """
        return dataclasses.replace(self, **kwargs)
    
    def __post_init__(self):
        # nanが含まれるがチェック
        if PortfolioState.nan_check:
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                if isinstance(value, np.ndarray):
                    if np.isnan(value).sum() > 0:
                        raise UnitStateHasNanError("This State has nan data about {}".format(field.name))
                    
        # 長さが適切かチェック
        if PortfolioState.length_check and self.names is not None and self.window is not None:
            name_length = len(self.names)
            window_length = len(self.window)
            for field in dataclasses.fields(self):
                if field.name in {"price_array", "volume_array"}:
                    value = getattr(self, field.name)
                    if value is not None:
                        if value.shape[0]!=name_length or value.shape[1]!=window_length:
                            err_str = "This State has wrong legnth about {}({}) with names({}) and window({})".format(field.name,
                                                                                                                      value.shape,
                                                                                                                      name_length,
                                                                                                                      window_length
                                                                                                                     )
                            raise UnitStateHasWrongLengthError(err_str)
                elif field.name in {"now_price_array", "portfolio_vector", "mean_cost_price_array"}:
                    value = getattr(self, field.name)
                    if value is not None:
                        if len(value.shape)!=1 or value.shape[0]!=name_length:
                            err_str = "This State has wrong length about {}({}) with names({})".format(field.name,
                                                                                                       value.shape,
                                                                                                       name_length
                                                                                                      )
                            raise UnitStateHasWrongLengthError(err_str)
        
        #portfolioの和が適切かチェック
        if PortfolioState.portfoio_check:
            portfolio_vector = self.portfolio_vector
            # nanのチェック
            if portfolio_vector is not None:
                if np.isnan(portfolio_vector).sum() > 0:
                    raise PortfolioVectorInvalidError("This portfolio has nan")

                # 上限と下限のチェック
                upper_bool = portfolio_vector > 1
                lower_bool = portfolio_vector < 0
                if upper_bool.sum() > 0 or lower_bool.sum() > 0:
                    raise PortfolioVectorInvalidError("This portfolio is not in (0,1).{}".format(portfolio_vector))

                # 和のチェック
                if abs(portfolio_vector.sum() - 1) > 1.e-5:
                    raise PortfolioVectorInvalidError("The portfolio sum is must be 1. This portfolio is {}, sum is {}".format(portfolio_vector,
                                                                                                                               portfolio_vector.sum()))
                
    @property
    def numbers(self):
        """
        保有量のプロパティ
        """
        return self.all_assets*self.portfolio_vector/self.now_price_array
    
    def __str__(self):
        return_str = "PortfolioState( \n"
        for field in dataclasses.fields(self):
            return_str += field.name + "="
            return_str += str(getattr(self, field.name)) + "\n"
        return_str += ")"
        return return_str
    
    def copy(self):
        """
        自身のコビーを返す．ndarrayのプロパティの場合はそのコビーを保持する．
        """
        arg_dict = {}
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, np.ndarray):
                field_value = field_value.copy()
            
            arg_dict[field.name] = field_value
        
        return PortfolioState(**arg_dict)
    
    def partial(self, *args):
        """
        str:
            フィールド名
        メモリ等の状況によって，自身の部分的なコビーを返す．
        引数にを耐えられなかったプロパティはNoneとなる．
        """
        arg_dict = {}
        for field in dataclasses.fields(self):
            if field.name in args:
                field_value = getattr(self, field.name)
                if isinstance(field_value, np.ndarray):
                    field_value = field_value.copy()
            else:
                field_value = None
            
            arg_dict[field.name] = field_value
            
        return PortfolioState(**arg_dict)


if __name__ == "__main__":
    pass