import re
import pandas as pd

from .datetime_freq_ver1 import middle_sample_type_with_check, get_df_freq, get_sec_from_freq

# 日本株の場合
#base_ohlc_patterns = {
#    "Open":re.compile("Open_[0-9]+"),
#    "High":re.compile("High_[0-9]+"),
#    "Low":re.compile("Low_[0-9]+"),
#    "Close":re.compile("Close_[0-9]+"),
#    "Volume":re.compile("Volume_[0-9]+")
#}

# アメリカ株を含む
base_ohlc_patterns = {
    "Open":re.compile("Open_.+"),
    "High":re.compile("High_.+"),
    "Low":re.compile("Low_.+"),
    "Close":re.compile("Close_.+"),
    "Volume":re.compile("Volume_.+")
}

class ConvertFreqOHLCV():
    """
    dataframeの足をダウンサンプリングする．
    """
    def __init__(self, freq_str, ohlcv_patterns=base_ohlc_patterns):
        """
        freq_str: str
            サンプリング周期に対応する文字列
        ohlcv_patterns: dict of str:pattern
            Open,High,Low,Close,Volumeとそれに対応するパターン
        """
        freq_str = middle_sample_type_with_check(freq_str)
        self.freq_str = freq_str
        
        self.ohlcv_patterns = ohlcv_patterns
         
    def __call__(self, df):
        # dfのfrequencyの判定とアップサンプリング時のエラー
        df_freq = get_df_freq(df)
        df_freq = middle_sample_type_with_check(df_freq)
        if get_sec_from_freq(df_freq) > get_sec_from_freq(self.freq_str):  # アップサンプリングの時
            raise ValueError("this is upsampling")
        
        # ohlcvそれぞれ対応するカラムを求める(aggに渡す辞書を求める)
        column_series = pd.Series(len(df.columns)*[None], index=df.columns)

        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Open"])] = "first"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["High"])] = "max"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Low"])] = "min"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Close"])] = "last"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Volume"])] = "sum"
        agg_dict = dict(column_series)

        resampled_df = df.resample(self.freq_str, label="left", closed="left").agg(agg_dict)  # resampling
        return resampled_df


if __name__ == "__main__":
  pass